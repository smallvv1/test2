#!/usr/local/bin/python3.10
"""B0 逐步训练分叉定位。

这是独立诊断 harness，不修改正式训练处理器。它从相同初始化、优化器、
GradScaler、科学随机命名空间和 train-only batch 计划出发，分别运行两条
短轨迹，并保存每一步的输入、forward、损失、梯度、AMP、optimizer 和
参数指纹。PPU 负责模型计算；CPU 只做指纹和汇总。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))

import gpc_science_rng as science  # noqa: E402
import gpc_a4_mechanism_probe as probe  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402

base = p7.base
SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
CONFIG = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen/B0_seed1234/config.yml"
COMMON_INIT = ENGINE / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth"
SCIENCE_PHASE = f"{science.SCIENCE_RNG_NAMESPACE}|{science.SCIENCE_PROTOCOL_ID}"
SEED = 1234
RANK = 0
EPOCH = 1


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def configure_science() -> None:
    p7.PHASE = SCIENCE_PHASE
    b6.PHASE = SCIENCE_PHASE
    base.PHASE = SCIENCE_PHASE
    p7.SEED = SEED
    b6.SEED = SEED
    base.SEED = SEED
    p7.WORLD_SIZE = 2
    b6.WORLD_SIZE = 2
    base.WORLD_SIZE = 2


def set_seed(seed: int) -> None:
    probe.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runtime_rng_digest() -> dict[str, Any]:
    result: dict[str, Any] = {}
    result["python"] = hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest()
    result["numpy"] = hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest()
    result["torch"] = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    try:
        states = torch.cuda.get_rng_state_all()
        result["torch_cuda"] = hashlib.sha256(
            b"".join(state.detach().cpu().numpy().tobytes() for state in states)
        ).hexdigest()
    except Exception as exc:
        result["torch_cuda"] = f"unavailable:{type(exc).__name__}"
    return result


def tensor_summary(value: Any) -> Any:
    if not torch.is_tensor(value):
        if isinstance(value, dict):
            return {str(k): tensor_summary(value[k]) for k in sorted(value, key=str)}
        if isinstance(value, (list, tuple)):
            return [tensor_summary(item) for item in value]
        return value
    detached = value.detach()
    flat = detached.float().reshape(-1)
    cpu = flat.cpu()
    if cpu.numel():
        indices = sorted(set((0, int(cpu.numel() // 2), int(cpu.numel() - 1))))
        samples = [float(cpu[index].item()) for index in indices]
        finite = bool(torch.isfinite(cpu).all().item())
        stats = {
            "mean": float(cpu.mean().item()),
            "std": float(cpu.std(unbiased=False).item()),
            "min": float(cpu.min().item()),
            "max": float(cpu.max().item()),
            "samples": samples,
            "finite": finite,
        }
    else:
        stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "samples": [], "finite": True}
    raw = detached.contiguous().cpu().numpy().tobytes()
    stats.update({"shape": list(detached.shape), "dtype": str(detached.dtype), "digest": hashlib.sha256(raw).hexdigest()})
    return stats


def scalar(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def selected_parameter_names(model: torch.nn.Module) -> list[str]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    selected: list[str] = []
    for name in names:
        lower = name.lower()
        if len(selected) < 12 or any(token in lower for token in ("head", "classifier", "bottleneck", "feature", "backbone")):
            if name not in selected:
                selected.append(name)
    for name in names[-12:]:
        if name not in selected:
            selected.append(name)
    return selected[:64]


def parameter_fingerprint(model: torch.nn.Module, names: list[str]) -> dict[str, Any]:
    lookup = dict(model.named_parameters())
    rows: dict[str, Any] = {}
    for name in names:
        parameter = lookup[name].detach()
        flat = parameter.float().reshape(-1)
        cpu = flat.cpu()
        indices = sorted(set((0, int(cpu.numel() // 2), int(cpu.numel() - 1)))) if cpu.numel() else []
        rows[name] = {
            "norm": float(torch.linalg.vector_norm(flat).cpu().item()) if cpu.numel() else 0.0,
            "absmax": float(flat.abs().max().cpu().item()) if cpu.numel() else 0.0,
            "samples": [float(cpu[i].item()) for i in indices],
        }
    return rows


def gradient_fingerprint(model: torch.nn.Module, names: list[str]) -> dict[str, Any]:
    lookup = dict(model.named_parameters())
    rows: dict[str, Any] = {}
    for name in names:
        grad = lookup[name].grad
        if grad is None:
            rows[name] = {"present": False}
            continue
        flat = grad.detach().float().reshape(-1)
        cpu = flat.cpu()
        indices = sorted(set((0, int(cpu.numel() // 2), int(cpu.numel() - 1)))) if cpu.numel() else []
        rows[name] = {
            "present": True,
            "finite": bool(torch.isfinite(flat).all().item()),
            "norm": float(torch.linalg.vector_norm(flat).cpu().item()) if cpu.numel() else 0.0,
            "absmax": float(flat.abs().max().cpu().item()) if cpu.numel() else 0.0,
            "samples": [float(cpu[i].item()) for i in indices],
        }
    return rows


def optimizer_fingerprint(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index, parameter in enumerate(optimizer.param_groups[0]["params"][:64]):
        state = optimizer.state.get(parameter, {})
        item: dict[str, Any] = {}
        for key in sorted(state):
            value = state[key]
            if torch.is_tensor(value):
                flat = value.detach().float().reshape(-1).cpu()
                indices = sorted(set((0, int(flat.numel() // 2), int(flat.numel() - 1)))) if flat.numel() else []
                item[str(key)] = {
                    "norm": float(torch.linalg.vector_norm(flat).item()) if flat.numel() else 0.0,
                    "samples": [float(flat[i].item()) for i in indices],
                }
            else:
                item[str(key)] = value
        rows[str(index)] = item
    return rows


def component_gradient_norms(model: torch.nn.Module, components: dict[str, Any], relation: torch.Tensor) -> dict[str, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    result: dict[str, float] = {}
    for name, loss in (("ce", components.get("ce")), ("triplet", components.get("triplet")),
                       ("bio", components.get("bio")), ("gpc", relation)):
        if loss is None:
            result[name] = 0.0
            continue
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        squared = None
        for gradient in gradients:
            if gradient is None:
                continue
            value = gradient.detach().float().pow(2).sum()
            squared = value if squared is None else squared + value
        result[name] = float(torch.sqrt(squared).detach().cpu()) if squared is not None else 0.0
    return result


def batch_record(batch: dict[str, Any]) -> dict[str, Any]:
    records = [
        {
            "image_path": str(batch["image_path"][index]),
            "pid": int(batch["pid"][index]),
            "clothes_id": int(batch["clothes_id"][index]),
            "augmentation_seed": int(batch["augmentation_seed"][index]),
        }
        for index in range(len(batch["pid"]))
    ]
    return {
        "plan_hash": science.list_sha256(records),
        "image_input_digest": tensor_summary(batch["image"])["digest"],
        "image_output_shape": list(batch["image"].shape),
        "augmentation_seeds": [int(value) for value in batch["augmentation_seed"]],
    }


def relation_for_batch(batch: dict[str, Any], path_to_index: dict[str, int], bank: dict[str, Any], iteration: int):
    entries = [
        {
            "pid": int(batch["pid"][index]),
            "clothes_id": int(batch["clothes_id"][index]),
            "image_path": str(batch["image_path"][index]),
        }
        for index in range(len(batch["pid"]))
    ]
    candidates = p7.candidate_lists(entries, path_to_index, bank)
    count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
    selected = science.science_selection(
        candidates["id"], count, seed=SEED, epoch=EPOCH, rank=RANK, iteration=iteration, kind="id"
    )
    return candidates, selected, int(count)


def load_runtime(device: torch.device, mode: str):
    cfg = probe.load_cfg(CONFIG)
    cfg.defrost()
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = SEED
    cfg.OUTPUT_DIR = str(ROOT / "outputs/gpc_first_divergence_20261019_145123")
    cfg.freeze()
    model, _ = probe.load_model(CONFIG, COMMON_INIT, device)
    loss_fn, center = base.make_loss(cfg, num_classes=base.NUM_CLASSES)
    optimizer, _ = base.make_optimizer(cfg, model, center)
    scheduler = base.create_scheduler(cfg, optimizer)
    if mode == "fixed_amp":
        scaler = amp.GradScaler(init_scale=1.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=100000)
    elif mode == "amp":
        scaler = amp.GradScaler(init_scale=base.INITIAL_SCALE)
    else:
        scaler = None
    return model, loss_fn, optimizer, scheduler, scaler, cfg


def run_once(device: torch.device, mode: str, label: str, steps: int, component_gradients: bool) -> dict[str, Any]:
    configure_science()
    set_seed(SEED)
    model, loss_fn, optimizer, scheduler, scaler, cfg = load_runtime(device, mode)
    names = selected_parameter_names(model)
    dataset = base.build_dataset(cfg)
    transform_train, _ = probe.build_img_transforms(cfg)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    state = probe.active_state(dataset, set(map(int, split["train_pids"])))
    index_map = base.build_index_map(dataset)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    loader, plan_meta = base.make_epoch_loader(dataset, transform_train, state, index_map, EPOCH, RANK)
    iterator = iter(loader)
    rows = []
    valid_updates = 0
    skipped_steps = 0
    for iteration in range(1, steps + 1):
        batch = next(iterator)
        batch_meta = batch_record(batch)
        candidates, selected, matched = relation_for_batch(batch, path_to_index, bank, iteration)
        relation_meta = {
            "candidate_hash": {key: science.relation_sha256(value) for key, value in candidates.items()},
            "selected_hash": science.relation_sha256(selected),
            "matched_count": matched,
            "selected_count": len(selected),
        }
        images = batch["image"].to(device, non_blocking=False)
        pids = batch["pid"].to(device, non_blocking=False)
        cams = batch["camid"].to(device, non_blocking=False)
        clothes = batch["clothes_id"].to(device, non_blocking=False)
        captions = batch["caption_feature"].to(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        writer = base.MetricCapture()
        rng_before = runtime_rng_digest()
        autocast_enabled = mode != "fp32"
        with amp.autocast(enabled=autocast_enabled):
            score, feat = model(images, cam_label=cams)
        forward_meta = {
            "score": tensor_summary(score),
            "feat": tensor_summary(feat),
        }
        base_total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=iteration)
        components = dict(getattr(loss_fn, "last_components", {}))
        z = F.normalize(feat[0].float(), dim=-1)
        values = [1.0 - torch.sum(z[i] * z[j]) for i, j, _key, _state in selected]
        relation = torch.stack(values).mean() if values else z.sum() * 0.0
        total = base_total + 0.1 * relation
        loss_meta = {
            "total": scalar(total),
            "ce": scalar(components.get("ce")),
            "triplet": scalar(components.get("triplet")),
            "bio": scalar(components.get("bio")),
            "gpc": scalar(relation),
        }
        component_gradient_meta = (
            component_gradient_norms(model, components, relation)
            if component_gradients else None
        )
        if not bool(torch.isfinite(total.detach()).item()):
            raise RuntimeError(f"非有限损失：{label} step={iteration}")
        if mode == "fp32":
            total.backward()
            gradient_meta = gradient_fingerprint(model, names)
            optimizer.step()
            found_inf = 0.0
            scale_before = None
            scale_after = None
        else:
            scale_before = float(scaler.get_scale())
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gradient_meta = gradient_fingerprint(model, names)
            scaler.step(optimizer)
            found_inf = float(base.found_inf(scaler, optimizer))
            scaler.update()
            scale_after = float(scaler.get_scale())
        skipped = bool(found_inf > 0)
        skipped_steps += int(skipped)
        valid_updates += int(not skipped)
        rng_after = runtime_rng_digest()
        rows.append({
            "step": iteration,
            "batch": batch_meta,
            "relation": relation_meta,
            "rng_before": rng_before,
            "forward": forward_meta,
            "loss": loss_meta,
            "component_gradient_norms": component_gradient_meta,
            "gradient": gradient_meta,
            "amp": {
                "mode": mode,
                "scale_before": scale_before,
                "found_inf": found_inf,
                "step_called": True,
                "skipped": skipped,
                "scale_after": scale_after,
            },
            "parameter_after": parameter_fingerprint(model, names),
            "optimizer_after": optimizer_fingerprint(optimizer),
            "rng_after": rng_after,
        })
        if iteration % 20 == 0:
            print(f"中文进度：{label} mode={mode} step={iteration}/{steps} 有效更新={valid_updates} skip={skipped_steps}", flush=True)
        del score, feat, total, base_total, relation
    torch.cuda.synchronize(device)
    return {
        "label": label,
        "mode": mode,
        "seed": SEED,
        "rank": RANK,
        "epoch": EPOCH,
        "steps": steps,
        "plan_meta": plan_meta,
        "initial_model_probe": parameter_fingerprint(model, names),
        "valid_updates": valid_updates,
        "skipped_steps": skipped_steps,
        "rows": rows,
        "cpu_model_fallback": False,
    }


def max_numeric_delta(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        return max((max_numeric_delta(left[key], right[key]) for key in left.keys() & right.keys()), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return max((max_numeric_delta(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0


def first_difference(first: dict[str, Any], second: dict[str, Any], tolerance: float) -> dict[str, Any]:
    for index, (left, right) in enumerate(zip(first["rows"], second["rows"]), start=1):
        if left["batch"]["plan_hash"] != right["batch"]["plan_hash"]:
            return {"step": index, "stage": "batch_plan", "detail": "batch plan hash differs"}
        if left["batch"]["image_input_digest"] != right["batch"]["image_input_digest"]:
            return {"step": index, "stage": "augmentation_output", "detail": "transformed image digest differs"}
        if left["relation"] != right["relation"]:
            return {"step": index, "stage": "selected_relation", "detail": "candidate or selected relation differs"}
        forward_delta = max_numeric_delta(left["forward"], right["forward"])
        if forward_delta > tolerance or left["forward"] != right["forward"]:
            return {"step": index, "stage": "forward", "max_numeric_delta": forward_delta, "exact_digest_difference": left["forward"] != right["forward"]}
        loss_delta = max_numeric_delta(left["loss"], right["loss"])
        if loss_delta > tolerance:
            return {"step": index, "stage": "loss", "max_numeric_delta": loss_delta}
        grad_delta = max_numeric_delta(left["gradient"], right["gradient"])
        if grad_delta > tolerance:
            return {"step": index, "stage": "gradient", "max_numeric_delta": grad_delta}
        if left["amp"] != right["amp"]:
            return {"step": index, "stage": "amp", "detail": "scale/found_inf/skip differs"}
        opt_delta = max_numeric_delta(left["optimizer_after"], right["optimizer_after"])
        if opt_delta > tolerance:
            return {"step": index, "stage": "optimizer", "max_numeric_delta": opt_delta}
        param_delta = max_numeric_delta(left["parameter_after"], right["parameter_after"])
        if param_delta > tolerance:
            return {"step": index, "stage": "parameter", "max_numeric_delta": param_delta}
    return {"step": None, "stage": "none", "detail": "no difference above tolerance"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--mode", choices=("amp", "fixed_amp", "fp32"), default="amp")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--component-gradients", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 预检失败：torch.cuda.is_available()=False")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not devices or any(name != "PPU-ZW810E" for name in devices):
        raise RuntimeError(f"PPU 设备不符合要求：{devices}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    first = run_once(device, args.mode, "pair_A", args.steps, args.component_gradients)
    torch.cuda.empty_cache()
    second = run_once(device, args.mode, "pair_B", args.steps, args.component_gradients)
    divergence = first_difference(first, second, tolerance=1e-4)
    payload = {
        "status": "FIRST_DIVERGENCE_FOUND" if divergence["stage"] != "none" else "NO_DIVERGENCE_ABOVE_TOLERANCE",
        "run_time": args.run_time,
        "mode": args.mode,
        "device_names": devices,
        "cpu_model_fallback": False,
        "first_divergence": divergence,
        "runs": [first, second],
        "source_sha256": {
            str(path): file_sha(path)
            for path in (
                ROOT / "scripts/gpc_science_rng.py",
                ROOT / "scripts/gpc_b0_first_divergence.py",
                ENGINE / "scripts/phase5b_id_damage_ds.py",
                ENGINE / "scripts/phase7i_relation_specificity.py",
            )
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"first_divergence_{args.mode}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "mode": args.mode, "first_divergence": divergence, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
