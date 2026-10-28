#!/usr/local/bin/python3.10
"""GPC 配对短程更新效用诊断（仅 train-only、非方法训练）。

U0 和 U1 从同一模型/optimizer/AMP 状态出发，使用完全相同的固定训练
batch；U1 只额外加入 0.1 * L_GPC。每个状态执行少量更新，再用独立的
train-only batch 评估。官方 query/gallery 和开发验证身份均不参与优化。
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
TRAIN_ROOT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen"
SCIENCE_PHASE = f"{science.SCIENCE_RNG_NAMESPACE}|{science.SCIENCE_PROTOCOL_ID}"
RANK = 0
EPOCH = 1
UPDATES = 4
EVAL_BATCHES = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_science(seed: int) -> None:
    p7.PHASE = SCIENCE_PHASE
    b6.PHASE = SCIENCE_PHASE
    base.PHASE = SCIENCE_PHASE
    p7.SEED = int(seed)
    b6.SEED = int(seed)
    base.SEED = int(seed)
    p7.WORLD_SIZE = 2
    b6.WORLD_SIZE = 2
    base.WORLD_SIZE = 2


def set_seed(seed: int) -> None:
    probe.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg():
    cfg = probe.load_cfg(CONFIG)
    cfg.defrost()
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = 1234
    cfg.OUTPUT_DIR = str(ROOT / "outputs/gpc_update_utility_20261019_141702")
    cfg.freeze()
    return cfg


def model_from_state(cfg: Any, state: dict[str, torch.Tensor], device: torch.device):
    cleaned = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    head = cleaned["head.weight"]
    model = probe.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=int(head.shape[0]), camera_num=12)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"模型 state 装载失败：missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)
    model.train()
    return model


def state_for_case(case: str, seed: int):
    if case == "initial":
        state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
        return state, None
    checkpoint = TRAIN_ROOT / f"B0_seed{seed}" / "rank0_state_epoch030.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload["model"], payload


def prepare_batches(cfg: Any, seed: int):
    configure_science(seed)
    set_seed(seed)
    dataset = base.build_dataset(cfg)
    transform_train, _ = probe.build_img_transforms(cfg)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    state = probe.active_state(dataset, train_pids)
    index_map = base.build_index_map(dataset)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    loader, loader_meta = base.make_epoch_loader(dataset, transform_train, state, index_map, EPOCH, RANK)
    batches = []
    selected = []
    for iteration, batch in enumerate(loader, start=1):
        entries = [{
            "pid": int(batch["pid"][i]),
            "clothes_id": int(batch["clothes_id"][i]),
            "image_path": str(batch["image_path"][i]),
        } for i in range(len(batch["pid"]))]
        candidates = p7.candidate_lists(entries, path_to_index, bank)
        count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
        pairs = science.science_selection(candidates["id"], count, seed=seed, epoch=EPOCH, rank=RANK, iteration=iteration, kind="id")
        if len(batches) < UPDATES + EVAL_BATCHES:
            batches.append(batch)
            selected.append({
                "iteration": iteration,
                "selected": pairs,
                "matched_count": int(count),
                "candidate_counts": {key: len(value) for key, value in candidates.items()},
            })
        if len(batches) >= UPDATES + EVAL_BATCHES:
            break
    if len(batches) < UPDATES + EVAL_BATCHES:
        raise RuntimeError("固定 train-only 诊断 batch 数量不足")
    return batches, selected, loader_meta


def batch_tensors(batch: dict[str, Any], device: torch.device):
    return (
        batch["image"].to(device, non_blocking=False),
        batch["pid"].to(device, non_blocking=False),
        batch["camid"].to(device, non_blocking=False),
        batch["clothes_id"].to(device, non_blocking=False),
        batch["caption_feature"].to(device, non_blocking=False),
    )


def relation_value(feat: Any, pairs: list[Any]):
    z = F.normalize(feat[0].float(), dim=-1)
    values = [1.0 - torch.sum(z[i] * z[j]) for i, j, _key, _state in pairs]
    relation = torch.stack(values).mean() if values else z.sum() * 0.0
    return relation, z


def base_forward(model, loss_fn, batch, device, step: int):
    images, pids, cams, clothes, captions = batch_tensors(batch, device)
    writer = base.MetricCapture()
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=cams)
    total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=step)
    components = dict(loss_fn.last_components)
    base_total = components["ce"] + components["triplet"] + components["bio"]
    return score, feat, total, base_total, components


def gradient_norm(model: torch.nn.Module) -> float:
    total = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).detach().cpu()) if total is not None else 0.0


def update_branch(cfg: Any, state: dict[str, torch.Tensor], payload: dict[str, Any] | None, batches: list[Any], selected: list[dict[str, Any]], device: torch.device, seed: int, use_gpc: bool):
    set_seed(seed)
    model = model_from_state(cfg, state, device)
    loss_fn, center = base.make_loss(cfg, num_classes=base.NUM_CLASSES)
    optimizer, _center_optimizer = base.make_optimizer(cfg, model, center)
    scheduler = base.create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=base.INITIAL_SCALE)
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
    updates = []
    for index in range(UPDATES):
        batch = batches[index]
        pairs = selected[index]["selected"]
        optimizer.zero_grad()
        score, feat, total_base, base_total, components = base_forward(model, loss_fn, batch, device, index + 1)
        relation, _z = relation_value(feat, pairs)
        total = base_total + (0.1 * relation if use_gpc else 0.0 * relation)
        if not bool(torch.isfinite(total.detach()).item()):
            raise RuntimeError("配对更新出现非有限总损失")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad = gradient_norm(model)
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        found_inf = float(base.found_inf(scaler, optimizer))
        scaler.update()
        updates.append({
            "step": index + 1,
            "base_loss": float(base_total.detach().float().cpu()),
            "total_loss": float(total.detach().float().cpu()),
            "gpc_loss": float(relation.detach().float().cpu()),
            "selected_pair_count": len(pairs),
            "gradient_norm_after_unscale": grad,
            "amp_scale_before": scale_before,
            "amp_scale_after": float(scaler.get_scale()),
            "found_inf": found_inf,
            "skipped": bool(found_inf > 0),
        })
        del score, feat, total_base, base_total, total, relation

    evaluations = []
    # The audited Bio model returns only inference features in eval() mode.
    # Keep train() mode while disabling gradients so the same score/feature
    # contract as the real training path is used for the train-only probe.
    model.train()
    with torch.no_grad():
        for index in range(UPDATES, UPDATES + EVAL_BATCHES):
            batch = batches[index]
            pairs = selected[index]["selected"]
            score, feat, _total, base_total, _components = base_forward(model, loss_fn, batch, device, 1000 + index)
            relation, z = relation_value(feat, pairs)
            pos = 1.0 - float(relation.detach().float().cpu()) if pairs else None
            id_acc = float((score["cls_score"].argmax(dim=-1) == batch["pid"].to(device)).float().mean().cpu())
            evaluations.append({
                "batch_index": index,
                "base_loss": float(base_total.detach().float().cpu()),
                "gpc_loss": float(relation.detach().float().cpu()),
                "positive_cosine_mean": pos,
                "identity_accuracy": id_acc,
                "selected_pair_count": len(pairs),
            })
    return {
        "use_gpc": use_gpc,
        "updates": updates,
        "evaluations": evaluations,
        "skipped_steps": sum(int(item["skipped"]) for item in updates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 设备不可用")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not names or any(name != "PPU-ZW810E" for name in names):
        raise RuntimeError(f"PPU 设备不符合要求：{names}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    cfg = load_cfg()
    rows = []
    for seed in (1234, 4321):
        for case in ("initial", "B0_E30"):
            state, payload = state_for_case(case, seed)
            batches, selected, loader_meta = prepare_batches(cfg, seed)
            u0 = update_branch(cfg, state, payload, batches, selected, device, seed, False)
            torch.cuda.empty_cache()
            u1 = update_branch(cfg, state, payload, batches, selected, device, seed, True)
            rows.append({
                "seed": seed,
                "state": case,
                "loader_meta": loader_meta,
                "selected_counts": [item["matched_count"] for item in selected],
                "u0": u0,
                "u1": u1,
            })
            print(f"中文进度：state={case} seed={seed} U0/U1 完成；U0 skips={u0['skipped_steps']} U1 skips={u1['skipped_steps']}", flush=True)
    # Pair-level deltas are intentionally descriptive, not a causal retrieval
    # claim. Any sign inconsistency is reported as INCONCLUSIVE.
    deltas = []
    for row in rows:
        u0_eval = row["u0"]["evaluations"]
        u1_eval = row["u1"]["evaluations"]
        for a, b in zip(u0_eval, u1_eval):
            deltas.append({
                "seed": row["seed"], "state": row["state"], "batch_index": a["batch_index"],
                "delta_base_loss_u1_minus_u0": b["base_loss"] - a["base_loss"],
                "delta_positive_cosine_u1_minus_u0": (b["positive_cosine_mean"] - a["positive_cosine_mean"]) if a["positive_cosine_mean"] is not None and b["positive_cosine_mean"] is not None else None,
                "delta_identity_accuracy_u1_minus_u0": b["identity_accuracy"] - a["identity_accuracy"],
            })
    loss_deltas = [item["delta_base_loss_u1_minus_u0"] for item in deltas]
    pos_deltas = [item["delta_positive_cosine_u1_minus_u0"] for item in deltas if item["delta_positive_cosine_u1_minus_u0"] is not None]
    stable_loss_sign = bool(loss_deltas) and all(value <= 0 for value in loss_deltas) or all(value >= 0 for value in loss_deltas)
    stable_pos_sign = bool(pos_deltas) and (all(value <= 0 for value in pos_deltas) or all(value >= 0 for value in pos_deltas))
    status = "GPC_UPDATE_UTILITY_SUPPORTED_LOCALLY" if stable_loss_sign and stable_pos_sign else "GPC_UPDATE_UTILITY_INCONCLUSIVE"
    payload = {
        "status": status,
        "run_time": args.run_time,
        "device_names": names,
        "device": "PPU-ZW810E",
        "cpu_model_fallback": False,
        "official_query_gallery_used": False,
        "development_validation_used_for_optimization": False,
        "updates_per_branch": UPDATES,
        "state_count": len(rows),
        "rows": rows,
        "paired_deltas": deltas,
        "summary": {
            "base_loss_delta_mean": sum(loss_deltas) / len(loss_deltas) if loss_deltas else None,
            "base_loss_delta_min": min(loss_deltas) if loss_deltas else None,
            "base_loss_delta_max": max(loss_deltas) if loss_deltas else None,
            "positive_cosine_delta_mean": sum(pos_deltas) / len(pos_deltas) if pos_deltas else None,
            "stable_base_loss_direction": stable_loss_sign,
            "stable_positive_cosine_direction": stable_pos_sign,
            "interpretation": "这是局部 train-only 更新诊断，不是检索因果证明；PPU 数值非确定性会降低可重复性证据强度。",
        },
        "source_sha256": {str(path): sha256_file(path) for path in (ROOT / "scripts/gpc_science_rng.py", ROOT / "scripts/gpc_update_utility_diagnostic.py", ENGINE / "scripts/phase5b_id_damage_ds.py", ENGINE / "scripts/phase7i_relation_specificity.py")},
    }
    write_json(args.output / "gpc_update_utility_diagnostic.json", payload)
    print(json.dumps({"status": status, "summary": payload["summary"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
