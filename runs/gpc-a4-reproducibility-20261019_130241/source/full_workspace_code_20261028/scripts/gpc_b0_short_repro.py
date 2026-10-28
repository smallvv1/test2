#!/usr/local/bin/python3.10
"""规范化科学随机命名空间下的 B0 短程可重复性实验。

两次运行从同一 common-init、同一 optimizer/AMP 初始状态和同一 rank0
训练计划出发，执行 120 个有效 optimizer.step。模型计算严格放在 PPU；
仅将哈希和标量统计转回 CPU。该脚本不是完整训练，也不做官方评估。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

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
SHORT_STEPS = 120


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def nested_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(b"tensor")
            digest.update(str(tuple(item.shape)).encode())
            digest.update(str(item.dtype).encode())
            digest.update(tensor_digest(item).encode())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda x: str(x)):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"seq")
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    probe.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_runtime(device: torch.device):
    cfg = probe.load_cfg(CONFIG)
    cfg.defrost()
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = SEED
    cfg.OUTPUT_DIR = str(ROOT / "outputs/gpc_science_rng_20261019_141702")
    cfg.freeze()
    model, _ = probe.load_model(CONFIG, COMMON_INIT, device)
    # The loader used by the historical engine keeps the original 77-way PID
    # head even when only the 60 development-training PIDs are active.
    loss_fn, center = base.make_loss(cfg, num_classes=base.NUM_CLASSES)
    optimizer, _center_optimizer = base.make_optimizer(cfg, model, center)
    scheduler = base.create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=base.INITIAL_SCALE)
    return model, loss_fn, optimizer, scheduler, scaler, cfg


def entry_from_batch(batch: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "pid": int(batch["pid"][index]),
        "clothes_id": int(batch["clothes_id"][index]),
        "image_path": str(batch["image_path"][index]),
    }


def relation_for_batch(batch: dict[str, Any], path_to_index: dict[str, int], bank: dict[str, Any], epoch: int, iteration: int):
    entries = [entry_from_batch(batch, index) for index in range(len(batch["pid"]))]
    candidates = p7.candidate_lists(entries, path_to_index, bank)
    count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
    selected = science.science_selection(
        candidates["id"], count, seed=SEED, epoch=epoch, rank=RANK, iteration=iteration, kind="id"
    )
    return candidates, selected, int(count)


def parameter_digest(model: torch.nn.Module) -> str:
    return nested_digest({key: value for key, value in model.state_dict().items()})


def tensor_probe(value: torch.Tensor) -> list[float]:
    cpu = value.detach().float().cpu().reshape(-1)
    if cpu.numel() == 0:
        return []
    indices = sorted(set((0, int(cpu.numel() // 2), int(cpu.numel() - 1))))
    return [float(cpu[index].item()) for index in indices]


def nested_probe(value: Any) -> list[float]:
    result: list[float] = []

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            result.extend(tensor_probe(item))
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: str(x)):
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return result


def run_once(device: torch.device, run_label: str, steps: int) -> dict[str, Any]:
    configure_science()
    set_seed(SEED)
    model, loss_fn, optimizer, scheduler, scaler, cfg = load_runtime(device)
    initial_model_hash = parameter_digest(model)
    dataset = base.build_dataset(cfg)
    transform_train, _ = probe.build_img_transforms(cfg)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    state = probe.active_state(dataset, train_pids)
    index_map = base.build_index_map(dataset)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    loader, plan_meta = base.make_epoch_loader(dataset, transform_train, state, index_map, EPOCH, RANK)
    batch_hashes = []
    relation_hashes = []
    losses = []
    amp_rows = []
    valid_updates = 0
    active_batches = 0
    skipped_steps = 0
    iterator = iter(loader)
    for iteration in range(1, steps + 1):
        batch = next(iterator)
        records = [
            {
                "image_path": str(batch["image_path"][index]),
                "pid": int(batch["pid"][index]),
                "clothes_id": int(batch["clothes_id"][index]),
                "augmentation_seed": int(batch["augmentation_seed"][index]),
            }
            for index in range(len(batch["pid"]))
        ]
        batch_hashes.append(science.list_sha256(records))
        candidates, selected, matched = relation_for_batch(batch, path_to_index, bank, EPOCH, iteration)
        relation_hashes.append({
            "candidate": {key: science.relation_sha256(value) for key, value in candidates.items()},
            "selected": science.relation_sha256(selected),
            "matched": matched,
            "selected_count": len(selected),
        })
        active_batches += int(bool(selected))
        images = batch["image"].to(device, non_blocking=False)
        pids = batch["pid"].to(device, non_blocking=False)
        cams = batch["camid"].to(device, non_blocking=False)
        clothes = batch["clothes_id"].to(device, non_blocking=False)
        captions = batch["caption_feature"].to(device, non_blocking=False)
        writer = base.MetricCapture()
        optimizer.zero_grad()
        with amp.autocast(enabled=True):
            score, feat = model(images, cam_label=cams)
        base_total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=iteration)
        z = F.normalize(feat[0].float(), dim=-1)
        values = [1.0 - torch.sum(z[i] * z[j]) for i, j, _key, _state in selected]
        relation = torch.stack(values).mean() if values else z.sum() * 0.0
        total = base_total + 0.1 * relation
        if not bool(torch.isfinite(total.detach()).item()):
            raise RuntimeError(f"非有限 total loss：run={run_label} iteration={iteration}")
        scaler.scale(total).backward()
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        found_inf = float(base.found_inf(scaler, optimizer))
        scaler.update()
        scale_after = float(scaler.get_scale())
        skipped = int(found_inf > 0)
        skipped_steps += skipped
        valid_updates += int(not skipped)
        losses.append(float(total.detach().float().cpu()))
        amp_rows.append({
            "iteration": iteration,
            "total_loss": losses[-1],
            "relation_loss": float(relation.detach().float().cpu()),
            "scale_before": scale_before,
            "scale_after": scale_after,
            "found_inf": found_inf,
            "skipped": skipped,
        })
        if iteration % 20 == 0:
            print(f"中文进度：{run_label} step={iteration}/{steps} 有效更新={valid_updates} active={active_batches} loss={losses[-1]:.6f}", flush=True)
        del score, feat, total, base_total, relation

    torch.cuda.synchronize(device)
    result = {
        "run_label": run_label,
        "seed": SEED,
        "rank": RANK,
        "epoch": EPOCH,
        "steps_requested": SHORT_STEPS,
        "valid_optimizer_updates": valid_updates,
        "amp_skipped_steps": skipped_steps,
        "active_relation_batches": active_batches,
        "plan_meta": plan_meta,
        "batch_hashes": batch_hashes,
        "relation_hashes": relation_hashes,
        "losses": losses,
        "amp_rows": amp_rows,
        "initial_model_hash": initial_model_hash,
        "final_model_hash": parameter_digest(model),
        "final_optimizer_hash": nested_digest(optimizer.state_dict()),
        "final_model_probe": nested_probe(model.state_dict()),
        "final_optimizer_probe": nested_probe(optimizer.state_dict()),
        "final_scheduler": scheduler.state_dict(),
        "final_scaler": scaler.state_dict(),
        "device": str(device),
        "cpu_model_fallback": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--steps", type=int, default=SHORT_STEPS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 预检失败：torch.cuda.is_available()=False")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not names or any(name != "PPU-ZW810E" for name in names):
        raise RuntimeError(f"PPU 设备不符合要求：{names}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    first = run_once(device, "B0_repeat_A", args.steps)
    torch.cuda.empty_cache()
    second = run_once(device, "B0_repeat_B", args.steps)
    same_batches = first["batch_hashes"] == second["batch_hashes"]
    same_relations = first["relation_hashes"] == second["relation_hashes"]
    same_losses = first["losses"] == second["losses"]
    same_model = first["final_model_hash"] == second["final_model_hash"]
    same_optimizer = first["final_optimizer_hash"] == second["final_optimizer_hash"]
    model_probe_diff = max(
        (abs(a - b) for a, b in zip(first["final_model_probe"], second["final_model_probe"])),
        default=0.0,
    )
    optimizer_probe_diff = max(
        (abs(a - b) for a, b in zip(first["final_optimizer_probe"], second["final_optimizer_probe"])),
        default=0.0,
    )
    numerical_tolerance = 1e-3
    numerical_only = (not same_model or not same_optimizer) and model_probe_diff <= numerical_tolerance and optimizer_probe_diff <= numerical_tolerance
    payload = {
        "status": "PASS" if all((same_batches, same_relations, same_losses, same_model, same_optimizer)) else ("PASS_WITH_PPU_NUMERICAL_TOLERANCE" if all((same_batches, same_relations, same_losses, numerical_only)) else "NUMERICAL_NONDETERMINISM"),
        "run_time": args.run_time,
        "science_namespace": SCIENCE_PHASE,
        "device_names": names,
        "cpu_model_fallback": False,
        "comparison": {
            "batch_hash_equal": same_batches,
            "selected_relation_hash_equal": same_relations,
            "loss_trace_equal": same_losses,
            "model_hash_equal": same_model,
            "optimizer_hash_equal": same_optimizer,
            "model_probe_max_abs_diff": model_probe_diff,
            "optimizer_probe_max_abs_diff": optimizer_probe_diff,
            "declared_numerical_tolerance": numerical_tolerance,
            "numerical_only_within_tolerance": numerical_only,
            "loss_max_abs_diff": max(abs(a - b) for a, b in zip(first["losses"], second["losses"])),
            "valid_updates_equal": first["valid_optimizer_updates"] == second["valid_optimizer_updates"],
            "amp_skip_equal": first["amp_skipped_steps"] == second["amp_skipped_steps"],
        },
        "runs": [first, second],
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (ROOT / "scripts/gpc_science_rng.py", ROOT / "scripts/gpc_b0_short_repro.py", ENGINE / "scripts/phase5b_id_damage_ds.py", ENGINE / "scripts/phase7i_relation_specificity.py")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "b0_short_reproducibility.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "comparison": payload["comparison"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    if payload["status"] not in ("PASS", "PASS_WITH_PPU_NUMERICAL_TOLERANCE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
