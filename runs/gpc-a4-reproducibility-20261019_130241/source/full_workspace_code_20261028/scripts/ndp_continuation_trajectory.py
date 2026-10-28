#!/usr/local/bin/python3.10
"""从已完成的 Bio+GPC-v1 E60 完整状态采集受控 continuation 轨迹。

这是新的 continuation，不回写历史实验，也不把它解释为原始 60 epoch 的中间
checkpoint。继续使用原 B0/I3 的 selector、pair-budget、CE、Triplet、Bio 和
GPC lambda=0.1；本脚本只保存 train-only 轨迹节点，不执行官方评估。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

# 老 checkpoint 的 numpy RNG pickle 使用 numpy._core 命名空间；当前验证环境
# 使用 numpy 1.23，因此在加载前提供只读兼容别名，不改写 checkpoint。
import numpy as np

import numpy.core
sys.modules.setdefault("numpy._core", numpy.core)
for _name in ("multiarray", "umath", "_multiarray_umath", "_dtype", "numerictypes", "numeric", "shape_base", "fromnumeric", "_asarray", "records", "memmap"):
    try:
        sys.modules.setdefault("numpy._core." + _name, __import__("numpy.core." + _name, fromlist=["*"]))
    except Exception:
        pass

import torch
import torch.distributed as dist
from torch.cuda import amp

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
ORIGINAL_RUN_TIME = "20261015_155822"
WORLD_SIZE = 2
STEPS_PER_EPOCH = 596
START_EPOCH = 61
TARGETS = (20, 50, 100, 200, 400)

sys.path.insert(0, str(ENGINE / "scripts"))
import gpc_v2_train as v2  # noqa: E402


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value) and value.ndim == 0:
        return value.detach().cpu().item()
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def device_guard(rank: int) -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("PPU_REAL_FORWARD_BLOCKED: 需要且只能使用两张 PPU，禁止 CPU fallback")
    torch.cuda.set_device(rank)
    name = torch.cuda.get_device_name(rank)
    if "PPU-ZW810E" not in name:
        raise RuntimeError(f"PPU 设备型号不符合预期: rank={rank}, device={name}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device(f"cuda:{rank}")


def load_resume(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def finite_parameters(model) -> bool:
    raw = model.module if isinstance(model, v2.base.DDP) else model
    # 每个节点进行完整参数有限性检查；这是止损检查，不参与梯度计算。
    return all(bool(torch.isfinite(param.detach()).all().item()) for param in raw.parameters())


def save_node(run_dir: Path, model, optimizer, scheduler, scaler, device, epoch_tag: int, global_step: int, effective_updates: int, rank: int, source_phase: str) -> dict:
    node = run_dir / f"node_update{effective_updates:04d}"
    node.mkdir(parents=True, exist_ok=True)
    state_path = node / f"rank{rank}_state.pt"
    v2.base.save_rank_state(state_path, model, optimizer, scheduler, scaler, epoch_tag, global_step, device)
    if rank == 0:
        raw = model.module if isinstance(model, v2.base.DDP) else model
        model_path = node / "model.pth"
        v2.base.save_model_checkpoint(raw, model_path)
        model_hash = sha256_file(model_path)
    else:
        model_hash = None
    holders = [model_hash]
    dist.broadcast_object_list(holders, src=0)
    dist.barrier()
    return {
        "effective_updates": effective_updates,
        "epoch_tag": epoch_tag,
        "global_step": global_step,
        "rank_state": str(state_path),
        "model": str(node / "model.pth"),
        "model_sha256": holders[0],
        "source_phase": source_phase,
        "optimizer_step_executed": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--resume-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--max-effective-updates", type=int, default=400)
    args = ap.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = device_guard(rank)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    source_state_path = args.resume_dir / f"rank{rank}_state_epoch060.pt"
    source_state_sha = sha256_file(source_state_path)
    source = load_resume(source_state_path)
    if set(source) != {"phase", "epoch", "global_step", "model", "optimizer", "scheduler", "scaler", "rng"}:
        raise RuntimeError(f"恢复状态字段不完整: {sorted(source)}")
    if int(source["epoch"]) != 60 or int(source["global_step"]) != 35760:
        raise RuntimeError(f"恢复状态不是 E60/global_step35760: {source['epoch']}/{source['global_step']}")

    class Args:
        output = OLD_OUTPUT
        run_time = ORIGINAL_RUN_TIME
        arm = "B0"
        stage = "two_seed"
        seed = args.seed
        epochs = 60
        full_data = False

    split = v2.load_split(OLD_OUTPUT)
    protocol = v2.install_protocol(Args, split, args.run_dir, None)
    base = v2.base
    p7 = v2.p7
    # install_protocol 用原始 run_time 恢复 B0 的稳定关系选择命名空间；输出目录
    # 仍是本轮独立目录。
    if p7.PHASE != f"gpc_v2_ltcc_{ORIGINAL_RUN_TIME}_two_seed_B0_seed{args.seed}":
        raise RuntimeError(f"GPC 科学命名空间异常: {p7.PHASE}")

    cfg = p7.configure()
    common_state, _init_audit = v2.b6.load_common_init()
    model, loss_fn, optimizer, scheduler, scaler = base.setup_runtime(cfg, common_state, device)
    model.module.load_state_dict(source["model"], strict=True)
    optimizer.load_state_dict(source["optimizer"])
    scheduler.load_state_dict(source["scheduler"])
    scaler.load_state_dict(source["scaler"])
    base.restore_rng(source["rng"], device)
    model.train()

    local_model_hash = v2.p7.model_parameter_hash(model)
    model_hashes = [None] * WORLD_SIZE
    dist.all_gather_object(model_hashes, local_model_hash)
    if len(set(model_hashes)) != 1:
        raise RuntimeError(f"恢复后两 rank 模型状态不一致: {model_hashes}")

    dataset = base.build_dataset(cfg)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    transform_train, _ = base.build_img_transforms(cfg)
    index_map = base.build_index_map(dataset)
    active_state = v2.b6.full_active_state(dataset)
    bank = v2.b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    if rank == 0:
        write_json(args.run_dir / "continuation_manifest.json", {
            "run_time": args.run_dir.name,
            "seed": args.seed,
            "stage": "E60_to_new_continuation",
            "source_phase": source["phase"],
            "source_state": str(args.resume_dir),
            "source_state_sha256_rank0": sha256_file(args.resume_dir / "rank0_state_epoch060.pt"),
            "source_state_sha256_rank1": sha256_file(args.resume_dir / "rank1_state_epoch060.pt"),
            "source_epoch": 60,
            "source_global_step": 35760,
            "source_scaler_scale_rank0_unknown_here": None,
            "initial_lr": [float(x["lr"]) for x in optimizer.param_groups],
            "initial_scaler_scale": float(scaler.get_scale()),
            "world_size": WORLD_SIZE,
            "train_only": True,
            "official_query_gallery_used": False,
            "cpu_model_fallback": False,
            "gpc_v1_preserved": True,
            "optimizer_step_executed": True,
            "sampler_epoch_start": START_EPOCH,
            "targets_effective_updates": [x for x in TARGETS if x <= args.max_effective_updates],
            "checkpoint_model_hash_by_rank": model_hashes,
        })

    metrics_path = args.run_dir / "trajectory_steps.csv"
    events_path = args.run_dir / "amp_events.csv"
    if rank == 0:
        mf = metrics_path.open("w", newline="", encoding="utf-8")
        ef = events_path.open("w", newline="", encoding="utf-8")
        mw = csv.DictWriter(mf, fieldnames=["batch_index", "epoch_tag", "iteration", "global_step", "effective_updates", "loss", "ce", "triplet", "bio", "gpc", "lambda_gpc", "lr", "scale_before", "scale_after", "found_inf", "step_executed", "parameter_finite", "selected_count", "active", "elapsed_seconds"])
        ew = csv.DictWriter(ef, fieldnames=["rank", "batch_index", "epoch_tag", "iteration", "global_step", "effective_updates", "scale_before", "scale_after", "found_inf", "step_executed"])
        mw.writeheader(); ew.writeheader(); mf.flush(); ef.flush()
    else:
        mf = ef = mw = ew = None

    start = time.time()
    batch_index = 0
    effective_updates = 0
    saved_nodes = []
    saved_effective: set[int] = set()
    failure = None
    last_heartbeat = 0.0
    try:
        # 同一 continuation 阶段保持恢复时的学习率；epoch_tag 只用于新的固定 sampler
        # 与原始 I3 selector 哈希，不重新调用 scheduler.step。
        while effective_updates < args.max_effective_updates:
            epoch_tag = START_EPOCH + batch_index // STEPS_PER_EPOCH
            iteration = batch_index % STEPS_PER_EPOCH + 1
            loader, plan_meta = base.make_epoch_loader(dataset, transform_train, active_state, index_map, epoch_tag, rank)
            if int(plan_meta["batches"]) != STEPS_PER_EPOCH or not plan_meta["batch_size_ok"] or not plan_meta["pair_size_ok"]:
                raise RuntimeError(f"sampler audit failed: {plan_meta}")
            iterator = iter(loader)
            for _ in range(STEPS_PER_EPOCH):
                batch = next(iterator)
                batch_index += 1
                global_step = int(source["global_step"]) + batch_index
                optimizer.zero_grad(set_to_none=True)
                result = p7.forward_batch(model, loss_fn, batch, device, path_to_index, bank, "B0", epoch_tag, rank, iteration, global_step)
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    raise RuntimeError(f"非有限总损失 at batch={batch_index}")
                scale_before = float(scaler.get_scale())
                scaler.scale(total).backward()
                scaler.step(optimizer)
                found_inf = int(base.found_inf(scaler, optimizer))
                scaler.update()
                scale_after = float(scaler.get_scale())
                flag = torch.tensor([found_inf], dtype=torch.int64, device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                min_inf = int(flag.item())
                flag2 = torch.tensor([found_inf], dtype=torch.int64, device=device)
                dist.all_reduce(flag2, op=dist.ReduceOp.MAX)
                max_inf = int(flag2.item())
                if min_inf != max_inf:
                    raise RuntimeError(f"两 rank AMP found_inf 不一致 at batch={batch_index}: min={min_inf}, max={max_inf}")
                step_executed = found_inf == 0
                if step_executed:
                    effective_updates += 1
                parameter_finite = finite_parameters(model) if step_executed else True
                if not parameter_finite:
                    raise RuntimeError(f"参数出现 NaN/Inf at batch={batch_index}")
                if rank == 0:
                    components = getattr(loss_fn, "last_components", {})
                    info = result["relation_info"]
                    mw.writerow({
                        "batch_index": batch_index, "epoch_tag": epoch_tag, "iteration": iteration, "global_step": global_step,
                        "effective_updates": effective_updates, "loss": float(result["total"].detach().float().cpu()),
                        "ce": float(components.get("ce", torch.tensor(float("nan"))).detach().float().cpu()),
                        "triplet": float(components.get("triplet", torch.tensor(float("nan"))).detach().float().cpu()),
                        "bio": float(components.get("bio", torch.tensor(float("nan"))).detach().float().cpu()),
                        "gpc": float(result["relation_raw"].detach().float().cpu()), "lambda_gpc": 0.1,
                        "lr": float(optimizer.param_groups[0]["lr"]), "scale_before": scale_before, "scale_after": scale_after,
                        "found_inf": found_inf, "step_executed": step_executed, "parameter_finite": parameter_finite,
                        "selected_count": int(info["selected_count"]), "active": bool(info["active"]), "elapsed_seconds": time.time() - start,
                    })
                    mf.flush()
                for event_rank in range(WORLD_SIZE):
                    if rank == event_rank and found_inf:
                        if rank == 0:
                            ew.writerow({"rank": rank, "batch_index": batch_index, "epoch_tag": epoch_tag, "iteration": iteration, "global_step": global_step, "effective_updates": effective_updates, "scale_before": scale_before, "scale_after": scale_after, "found_inf": found_inf, "step_executed": step_executed})
                        break
                if rank == 0 and (effective_updates in TARGETS or time.time() - last_heartbeat > 300):
                    last_heartbeat = time.time()
                    write_json(args.run_dir / "heartbeat.json", {"time": now(), "seed": args.seed, "batch_index": batch_index, "effective_updates": effective_updates, "latest_source": str(args.run_dir / "trajectory_steps.csv")})
                if effective_updates in TARGETS and effective_updates > 0 and effective_updates not in saved_effective:
                    node = save_node(args.run_dir, model, optimizer, scheduler, scaler, device, epoch_tag, global_step, effective_updates, rank, source["phase"])
                    saved_effective.add(effective_updates)
                    if rank == 0:
                        saved_nodes.append(node)
                        write_json(args.run_dir / "saved_nodes.json", saved_nodes)
                if effective_updates >= args.max_effective_updates:
                    break
                iteration += 1
            del iterator, loader
            gc.collect()
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "time": now(), "batch_index": batch_index, "effective_updates": effective_updates}
        if rank == 0:
            write_json(args.run_dir / "failure.json", failure)
        raise
    finally:
        if rank == 0:
            write_json(args.run_dir / "trajectory_summary.json", {"seed": args.seed, "effective_updates": effective_updates, "batch_index": batch_index, "saved_nodes": saved_nodes, "failure": failure, "elapsed_seconds": time.time() - start, "optimizer_step_executed": True})
            if mf is not None: mf.close()
            if ef is not None: ef.close()
        dist.barrier()
        v2.b6.cleanup_dist()
        protocol["restore"]


if __name__ == "__main__":
    main()
