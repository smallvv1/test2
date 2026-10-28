#!/usr/bin/env python3
"""Small PPU calibration for the existing DIFFER+GPC dynamic-risk pilot.

The script never writes a checkpoint. It loads the recovered rank-0 training
state, reconstructs fixed LTCC train-only probe neighbourhoods, and executes
paired virtual updates. It is intentionally diagnostic: the historical GPC
selector and denominator are reproduced inline from Phase 7I, while the NDP
prototype remains in ``ndp_structure_losses.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase7i_relation_specificity as p7  # noqa: E402

base = p7.base
b6 = p7.b6

PRIOR = Path("/home/DIFFER/outputs/ndp_checkpoint_recovery/20261014_184400")
CHECKPOINT = Path("/home/output/phase7i_relation_specificity/I3_id_only_matched/model_final.pth")
RANK_STATE = Path("/home/output/phase7i_relation_specificity/I3_id_only_matched/rank0_state_epoch060.pt")
LAMBDA = 0.1
ALPHAS = (0.1, 0.25, 0.5, 1.0)
RELATION_SOURCE = ROOT / "outputs/new_method_track/phase6b_soft_relation_screen/artifacts/relation_source_rows.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _json_value(v) for k, v in row.items()})


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_payload(path: Path) -> dict[str, Any]:
    # PyTorch 2.6's safe loader does not understand the historical NumPy RNG
    # payload. The file is a locally audited rank state, so use the explicit
    # full loader after registering the legacy NumPy aliases.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    return torch.load(path, map_location="cpu", weights_only=False)


def configure() -> Any:
    cfg = b6.configure()
    cfg.defrost()
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()
    return cfg


def model_from_state(cfg: Any, state: dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    model = base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=77, camera_num=base.NUM_CAMERAS)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def snapshot_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def canonical_key(left: int, right: int) -> tuple[int, int]:
    return min(int(left), int(right)), max(int(left), int(right))


def load_targets(prior: Path, limit: int) -> list[dict[str, Any]]:
    rows = read_csv(prior / "NDP_DYNAMIC_RISK_RAW.csv")
    rows = rows[:limit]
    for row in rows:
        for k in ("candidate_index", "batch_iteration", "relation_index_in_batch", "global_left", "global_right", "pid"):
            row[k] = int(row[k])
        row["key"] = canonical_key(row["global_left"], row["global_right"])
    return rows


def load_fast_bank() -> dict[str, Any]:
    """Load only exact image-pair membership required by the I3 selector."""
    exact = set()
    with RELATION_SOURCE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            exact.add(canonical_key(int(row["index_a"]), int(row["index_b"])))
    return {"exact_lookup": exact}


def collect_batches(dataset: Any, cfg: Any, targets: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    """Find each historical relation in an independently replayed train batch.

    The historical raw file came from a prior sampler wrapper that is not part
    of the current source tree. We therefore match by canonical train-image
    pair, preserve the current Phase 7I selector, and record the actual batch
    used for this calibration rather than silently claiming sampler identity.
    """
    path_to_index = {str(item["image_path"]): i for i, item in enumerate(dataset.train)}
    bank = load_fast_bank()
    transform_train, _ = base.build_img_transforms(cfg)
    from data.dataset_loader import ImageDataset
    from torch.utils.data import DataLoader

    # The old diagnostic wrapper that produced the raw CSV is intentionally
    # not treated as a current source of truth. Build a small valid identity
    # batch around each historical pair and let the released Phase 7I
    # selector choose the relation set. This preserves the selector and the
    # original denominator while making the calibration reproducible from the
    # current repository.
    by_pid: dict[int, list[int]] = {}
    for i, item in enumerate(dataset.train):
        by_pid.setdefault(int(item["pid"]), []).append(i)
    target_indices = {i for row in targets for i in row["key"]}
    auxiliary_pairs: list[tuple[int, int]] = []
    for pid, indices in sorted(by_pid.items()):
        if any(i in target_indices for i in indices):
            continue
        same = None
        cross = None
        for a_pos, a in enumerate(indices):
            for b in indices[a_pos + 1:]:
                ca = int(dataset.train[a]["clothes_id"]); cb = int(dataset.train[b]["clothes_id"])
                key = canonical_key(a, b)
                if ca == cb and same is None:
                    same = (a, b)
                if ca != cb and key in bank["exact_lookup"] and cross is None:
                    cross = (a, b)
        if same is not None:
            auxiliary_pairs.append(same)
        if cross is not None:
            auxiliary_pairs.append(cross)
    if len(auxiliary_pairs) < 8:
        raise RuntimeError("not enough auxiliary identity pairs for diagnostic batches")

    found: dict[tuple[int, int], dict[str, Any]] = {}
    for target in targets:
        key = target["key"]
        target_pair = key
        rng = random.Random(1234 + int(target["candidate_index"]))
        selected_sources = None
        selected_meta = None
        for attempt in range(2000):
            choices = rng.sample(auxiliary_pairs, 3)
            pairs = [target_pair] + choices
            flat = [idx for pair in pairs for idx in pair]
            if len(set(flat)) != len(flat):
                continue
            sources = [dict(dataset.train[idx]) for idx in flat]
            entries = [{"pid": int(x["pid"]), "clothes_id": int(x["clothes_id"]), "image_path": str(x["image_path"])} for x in sources]
            chosen, count, info = p7.selected_relation_pairs(entries, path_to_index, bank, "I3", 60, 0, 1)
            selected_keys = [tuple(item[2]) for item in chosen["id"]]
            if key in selected_keys and count > 0:
                selected_sources = sources
                selected_meta = {"count": count, "info": info, "attempt": attempt}
                break
        if selected_sources is None:
            raise RuntimeError(f"could not construct selector-valid batch for {key}")
        loader = DataLoader(ImageDataset(selected_sources, transform=transform_train), batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        found[key] = {
            "batch": clone_batch(batch), "iteration_replayed": 1,
            "selected_index_replayed": [tuple(item[2]) for item in p7.selected_relation_pairs(
                [{"pid": int(batch["pid"][i]), "clothes_id": int(batch["clothes_id"][i]), "image_path": str(batch["image_path"][i])} for i in range(len(batch["pid"]))],
                path_to_index, bank, "I3", 60, 0, 1
            )[0]["id"]].index(key),
            "selected_count_replayed": selected_meta["count"],
            "info": selected_meta["info"], "meta": {"sampler_identity": "manual selector-valid diagnostic batch"},
        }
    return found


def selected_relation_values(
    feat: list[torch.Tensor],
    batch: dict[str, Any],
    path_to_index: dict[str, int],
    bank: dict[str, Any],
    target_key: tuple[int, int],
) -> tuple[torch.Tensor, int, list[tuple[int, int]]]:
    representation = F.normalize(feat[0].float(), dim=-1)
    entries = [
        {"pid": int(batch["pid"][i]), "clothes_id": int(batch["clothes_id"][i]), "image_path": str(batch["image_path"][i])}
        for i in range(len(batch["pid"]))
    ]
    chosen, count, _ = p7.selected_relation_pairs(entries, path_to_index, bank, "I3", 60, 0, 1)
    # The caller's batch iteration is injected by replacing this list with the
    # already replayed selected positions; this defensive path is retained for
    # unit-level use and computes the selector for iteration 1 only.
    selected = chosen["id"]
    values = torch.stack([1.0 - torch.sum(representation[i] * representation[j]) for i, j, _key, _state in selected]) if selected else representation.sum().reshape(1) * 0.0
    positions = [canonical_key(path_to_index[str(batch["image_path"][i])], path_to_index[str(batch["image_path"][j])]) for i, j, _key, _state in selected]
    if target_key not in positions:
        raise RuntimeError(f"target {target_key} not in selected batch pairs {positions}")
    return values, positions.index(target_key), positions


def make_optimizer_and_scaler(cfg: Any, model: torch.nn.Module, payload: dict[str, Any], device: torch.device):
    loss_fn, center = base.make_loss(cfg, num_classes=77)
    optimizer, _ = base.make_optimizer(cfg, model, center)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler = base.create_scheduler(cfg, optimizer)
    scheduler.load_state_dict(payload["scheduler"])
    scaler = amp.GradScaler(init_scale=1024)
    scaler.load_state_dict(payload["scaler"])
    return loss_fn, optimizer, scheduler, scaler


def reset_runtime(model: torch.nn.Module, cfg: Any, payload: dict[str, Any], device: torch.device):
    model.load_state_dict(payload["model"], strict=True)
    model.train()
    loss_fn, optimizer, scheduler, scaler = make_optimizer_and_scaler(cfg, model, payload, device)
    return loss_fn, optimizer, scheduler, scaler


def snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def load_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def tensor_norm(values: list[torch.Tensor]) -> float:
    return float(torch.sqrt(sum(torch.sum(v.float() * v.float()) for v in values)).detach().cpu()) if values else 0.0


def gpc_relation_loss(
    model: torch.nn.Module,
    loss_fn: Any,
    batch: dict[str, Any],
    device: torch.device,
    target_local_index: int,
    selected_positions: list[tuple[int, int]],
) -> dict[str, Any]:
    images = batch["image"].to(device)
    pids = batch["pid"].to(device)
    cams = batch["camid"].to(device)
    clothes = batch["clothes_id"].to(device)
    captions = batch["caption_feature"].to(device)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=cams)
    base_loss = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=base.MetricCapture(), step=35761)
    rep = F.normalize(feat[0].float(), dim=-1)
    # The selected order and count are supplied by collect_batches. Rebuild
    # local indices from the canonical image pairs, preserving i<j semantics.
    values = []
    local_pairs = []
    for key in selected_positions:
        local = None
        for i in range(len(batch["image_path"])):
            for j in range(i + 1, len(batch["image_path"])):
                pair = canonical_key(int(batch["_global_index"][i]), int(batch["_global_index"][j])) if "_global_index" in batch else None
                if pair == key:
                    local = (i, j)
                    break
            if local:
                break
        if local is None:
            raise RuntimeError(f"selected pair {key} not found in batch")
        i, j = local
        local_pairs.append(local)
        values.append(1.0 - torch.sum(rep[i] * rep[j]))
    rel_values = torch.stack(values)
    relation_full = rel_values.mean()
    relation_masked = (rel_values.sum() - rel_values[target_local_index]) / float(len(rel_values))
    return {
        "score": score,
        "feat": feat,
        "base_loss": base_loss,
        "rel_values": rel_values,
        "relation_full": relation_full,
        "relation_masked": relation_masked,
        "total_full": base_loss + LAMBDA * relation_full,
        "total_masked": base_loss + LAMBDA * relation_masked,
        "local_pairs": local_pairs,
    }


def find_local_pairs(batch: dict[str, Any], path_to_index: dict[str, int], selected_positions: list[tuple[int, int]]) -> dict[str, Any]:
    batch = clone_batch(batch)
    batch["_global_index"] = [path_to_index[str(path)] for path in batch["image_path"]]
    return batch


def optimizer_step(
    model: torch.nn.Module,
    cfg: Any,
    payload: dict[str, Any],
    batch: dict[str, Any],
    target_index: int,
    selected_positions: list[tuple[int, int]],
    device: torch.device,
    rng: dict[str, Any],
    want_gpc_gradient: bool,
    use_masked_loss: bool,
) -> dict[str, Any]:
    loss_fn, optimizer, scheduler, scaler = reset_runtime(model, cfg, payload, device)
    restore_rng(rng)
    out = gpc_relation_loss(model, loss_fn, batch, device, target_index, selected_positions)
    gpc_contribution = LAMBDA * out["rel_values"][target_index] / float(len(out["rel_values"]))
    gpc_grads = None
    if want_gpc_gradient:
        params = [p for p in model.parameters() if p.requires_grad]
        gpc_grads = torch.autograd.grad(gpc_contribution, params, retain_graph=True, allow_unused=True)
    total = out["total_masked"] if use_masked_loss else out["total_full"]
    scaler.scale(total).backward()
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    inf = float(base.found_inf(scaler, optimizer))
    scaler.update()
    scale_after = float(scaler.get_scale())
    state = snapshot_state(model)
    return {
        **out,
        "gpc_contribution": gpc_contribution,
        "gpc_grads": gpc_grads,
        "inf": inf,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "state": state,
        "loss_after_selection": total.detach().float().cpu().item(),
    }


def probe_tensors(dataset: Any, indices: list[int], transform_test: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    from data.dataset_loader import read_image
    images = []
    cams = []
    for index in indices:
        item = dataset.train[int(index)]
        images.append(transform_test(read_image(item["image_path"])))
        cams.append(int(item["camid"]))
    return torch.stack(images).to(device), torch.tensor(cams, dtype=torch.long, device=device)


def probe_metric_values(
    model: torch.nn.Module,
    probe_images: torch.Tensor,
    probe_cams: torch.Tensor,
    probe_positions: dict[int, int],
    left: int,
    right: int,
    neighbors: np.ndarray,
    neighbor_right: np.ndarray,
    require_grad: bool,
) -> dict[str, torch.Tensor]:
    with amp.autocast(enabled=True):
        output = model(probe_images, cam_label=probe_cams)
    # EVA02-Bio exposes feat[0] during training but returns the 1024-D
    # identity feature tensor directly in eval mode.
    feat = output[1][0] if isinstance(output, tuple) else output
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    z = F.normalize(feat.float(), dim=-1)
    li, ri = probe_positions[left], probe_positions[right]
    lni = torch.tensor([probe_positions[int(x)] for x in neighbors], device=z.device)
    rni = torch.tensor([probe_positions[int(x)] for x in neighbor_right], device=z.device)
    pos = torch.sum(z[li] * z[ri])
    left_neg = torch.mv(z[lni], z[li])
    right_neg = torch.mv(z[rni], z[ri])
    left_hard = pos - left_neg.max()
    right_hard = pos - right_neg.max()
    return {
        "positive_similarity": pos,
        "single_triple_margin": pos - left_neg[0],
        "hard_margin_left": left_hard,
        "average_margin_left": pos - left_neg.mean(),
        "minimum_margin_two_sides": torch.minimum(left_hard, right_hard),
        "average_margin_two_sides": (left_hard + right_hard) / 2.0,
        "hard_negative_similarity_left": left_neg.max(),
        "hard_negative_similarity_right": right_neg.max(),
        "left_margin": left_hard,
        "right_margin": right_hard,
    }


def compute_param_delta(mask_state: dict[str, torch.Tensor], full_state: dict[str, torch.Tensor], model: torch.nn.Module) -> tuple[list[torch.Tensor], list[str]]:
    delta = []
    names = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        delta.append(full_state[name] - mask_state[name])
        names.append(name)
    return delta, names


def gradient_stats(grads: tuple[torch.Tensor | None, ...], delta: list[torch.Tensor]) -> dict[str, float]:
    valid = [(g, d) for g, d in zip(grads, delta) if g is not None]
    if not valid:
        return {"grad_norm": 0.0, "delta_norm": tensor_norm(delta), "dot": 0.0, "cosine": 0.0, "coverage": 0.0}
    gs = [g.detach() for g, _ in valid]
    ds = [d.detach() for _, d in valid]
    gn = tensor_norm(gs); dn = tensor_norm(ds)
    dot = float(sum(torch.sum(g.float() * d.float()) for g, d in valid).detach().cpu())
    cosine = dot / (gn * dn) if gn > 0 and dn > 0 else 0.0
    return {"grad_norm": gn, "delta_norm": dn, "dot": dot, "cosine": cosine, "coverage": len(valid) / max(1, len(grads))}


def set_interpolated_parameters(model: torch.nn.Module, mask_state: dict[str, torch.Tensor], full_state: dict[str, torch.Tensor], alpha: float) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(mask_state[name] + float(alpha) * (full_state[name] - mask_state[name]))
        for name, b in model.named_buffers():
            # Some EVA rotary buffers are non-persistent and therefore absent
            # from state_dict; they are unchanged by the optimizer step.
            if name in mask_state:
                b.copy_(mask_state[name])


def float_value(x: torch.Tensor | float) -> float:
    return float(x.detach().float().cpu()) if torch.is_tensor(x) else float(x)


def run_one(
    model: torch.nn.Module,
    cfg: Any,
    payload: dict[str, Any],
    dataset: Any,
    target: dict[str, Any],
    found: dict[str, Any],
    path_to_index: dict[str, int],
    bank: dict[str, Any],
    neighborhoods: dict[str, np.ndarray],
    device: torch.device,
    transform_test: Any,
    rng: dict[str, Any],
    repetition: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    key = target["key"]
    batch = find_local_pairs(found["batch"], path_to_index, [])
    # Recompute selection using the replayed iteration and retain all selected
    # image pairs in their SHA-256 order.
    entries = [{"pid": int(batch["pid"][i]), "clothes_id": int(batch["clothes_id"][i]), "image_path": str(batch["image_path"][i])} for i in range(len(batch["pid"]))]
    chosen, count, info = p7.selected_relation_pairs(entries, path_to_index, bank, "I3", 60, 0, found["iteration_replayed"])
    selected_positions = [canonical_key(path_to_index[str(batch["image_path"][i])], path_to_index[str(batch["image_path"][j])]) for i, j, _key, _state in chosen["id"]]
    target_index = selected_positions.index(key)
    batch["_global_index"] = [path_to_index[str(path)] for path in batch["image_path"]]

    left, right = key
    neigh_l = neighborhoods["neighbors"][left]
    neigh_r = neighborhoods["neighbors"][right]
    probe_indices = sorted(set([left, right, *[int(x) for x in neigh_l], *[int(x) for x in neigh_r]]))
    probe_images, probe_cams = probe_tensors(dataset, probe_indices, transform_test, device)
    probe_positions = {idx: pos for pos, idx in enumerate(probe_indices)}

    # Baseline feature geometry before either virtual update.
    reset_runtime(model, cfg, payload, device)
    model.eval()
    with torch.no_grad():
        baseline_metrics = probe_metric_values(model, probe_images, probe_cams, probe_positions, left, right, neigh_l, neigh_r, False)
    model.train()

    masked = optimizer_step(model, cfg, payload, batch, target_index, selected_positions, device, rng, False, True)
    mask_state = masked["state"]
    load_state(model, mask_state)
    model.eval()
    metric_mask = probe_metric_values(model, probe_images, probe_cams, probe_positions, left, right, neigh_l, neigh_r, True)
    metric_names = ["single_triple_margin", "hard_margin_left", "average_margin_left", "minimum_margin_two_sides", "average_margin_two_sides"]
    metric_grads: dict[str, tuple[torch.Tensor | None, ...]] = {}
    params = [p for p in model.parameters() if p.requires_grad]
    for name in metric_names:
        metric_grads[name] = torch.autograd.grad(metric_mask[name], params, retain_graph=True, allow_unused=True)
    model.train()

    full = optimizer_step(model, cfg, payload, batch, target_index, selected_positions, device, rng, True, False)
    full_state = full["state"]
    delta, param_names = compute_param_delta(mask_state, full_state, model)
    gpc_stats = gradient_stats(tuple(full["gpc_grads"]), delta)

    # The difference between two SGD momentum updates cancels the old
    # momentum, weight decay and all base-loss terms. Compare it to the exact
    # weighted single-pair GPC gradient multiplied by each parameter LR.
    expected_parts = []
    actual_parts = []
    trainable_named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    for (name, _p), g in zip(trainable_named, full["gpc_grads"]):
        if g is None:
            continue
        parameter_lr = float(cfg.SOLVER.BASE_LR)
        if "bias" in name:
            parameter_lr *= float(cfg.SOLVER.BIAS_LR_FACTOR)
        if bool(cfg.SOLVER.LARGE_FC_LR) and "head_cls" in name:
            parameter_lr *= float(cfg.SOLVER.LARGE_FC_FACTOR)
        expected_parts.append(-parameter_lr * g.detach())
        actual_parts.append(full_state[name] - mask_state[name])
    expected_norm = tensor_norm(expected_parts)
    actual_norm = tensor_norm(actual_parts)
    expected_dot = float(sum(torch.sum(a.float() * b.float()) for a, b in zip(expected_parts, actual_parts)).detach().cpu()) if expected_parts else 0.0
    expected_cos = expected_dot / (expected_norm * actual_norm) if expected_norm > 0 and actual_norm > 0 else 0.0
    residual = [a - b for a, b in zip(actual_parts, expected_parts)]

    load_state(model, mask_state)
    model.eval()
    taylor_rows = []
    for alpha in ALPHAS:
        set_interpolated_parameters(model, mask_state, full_state, alpha)
        vals = probe_metric_values(model, probe_images, probe_cams, probe_positions, left, right, neigh_l, neigh_r, False)
        for metric_name in metric_names:
            actual = float_value(vals[metric_name] - metric_mask[metric_name])
            dot = float(sum(torch.sum(g.float() * d.float()) for g, d in zip(metric_grads[metric_name], delta) if g is not None).detach().cpu())
            predicted = alpha * dot
            taylor_rows.append({
                "candidate_index": target["candidate_index"], "relation_key": f"{left}:{right}", "pid": target["pid"],
                "repetition": repetition, "alpha": alpha, "metric": metric_name,
                "metric_mask": float_value(metric_mask[metric_name]), "metric_scaled": float_value(vals[metric_name]),
                "real_delta": actual, "first_order_prediction": predicted,
                "absolute_error": abs(actual - predicted), "direction_match": int((actual == 0 and predicted == 0) or (actual * predicted > 0)),
            })
    # Restore baseline model state before returning to avoid carrying BN or
    # interpolated values into the next candidate.
    load_state(model, payload["model"])
    model.train()

    row = {
        "candidate_index": target["candidate_index"], "relation_key": f"{left}:{right}", "pid": target["pid"],
        "clothes_condition": target["clothes_condition"], "clothes_left": target["clothes_left"], "clothes_right": target["clothes_right"],
        "repetition": repetition, "replayed_batch_iteration": found["iteration_replayed"],
        "historical_batch_iteration": target["batch_iteration"], "historical_relation_index": target["relation_index_in_batch"],
        "selected_count": count, "target_relation_index_replayed": target_index,
        "gpc_pair_loss": float_value(full["rel_values"][target_index]),
        "gpc_mean_full": float_value(full["relation_full"]), "gpc_mean_masked": float_value(masked["relation_masked"]),
        "base_loss_full": float_value(full["base_loss"]), "base_loss_masked": float_value(masked["base_loss"]),
        "delta_M_real": float_value(baseline_metrics["hard_margin_left"] * 0.0 + 0.0),
        "M0_hard_left": float_value(baseline_metrics["hard_margin_left"]),
        "M_mask_hard_left": float_value(metric_mask["hard_margin_left"]),
        "positive_mask": float_value(metric_mask["positive_similarity"]),
        "positive_before": float_value(baseline_metrics["positive_similarity"]),
        "hard_negative_before": float_value(baseline_metrics["hard_negative_similarity_left"]),
        "hard_negative_mask": float_value(metric_mask["hard_negative_similarity_left"]),
        "initially_correct_left": int(float_value(baseline_metrics["hard_margin_left"]) > 0),
        "initially_correct_right": int(float_value(baseline_metrics["right_margin"]) > 0),
        "full_inf": full["inf"], "masked_inf": masked["inf"], "full_scale_before": full["scale_before"], "full_scale_after": full["scale_after"],
        "masked_scale_before": masked["scale_before"], "masked_scale_after": masked["scale_after"],
        "delta_theta_norm": tensor_norm(delta), "expected_gpc_update_norm": expected_norm,
        "update_expected_cosine": expected_cos, "update_residual_norm": tensor_norm(residual),
        "update_relative_residual": tensor_norm(residual) / expected_norm if expected_norm > 0 else 0.0,
        "gpc_gradient_norm": gpc_stats["grad_norm"], "gpc_delta_dot": gpc_stats["dot"], "gpc_delta_cosine": gpc_stats["cosine"],
        "gpc_gradient_parameter_coverage": gpc_stats["coverage"],
        "gpc_contribution": float_value(full["gpc_contribution"]),
        "reversal_full_left": 0, "reversal_mask_left": 0,
    }
    # Alpha=1 main metric gives the requested real paired update influence.
    main = [x for x in taylor_rows if x["alpha"] == 1.0 and x["metric"] == "hard_margin_left"][0]
    row["delta_M_real"] = main["real_delta"]
    row["delta_M_first_order"] = main["first_order_prediction"]
    row["delta_M_first_order_error"] = main["absolute_error"]
    row["hard_negative_change"] = row["hard_negative_mask"] - row["hard_negative_before"]
    row["repeat_key"] = f"{left}:{right}"
    return row, taylor_rows, {"mask_state": mask_state, "full_state": full_state}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--repeat-limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PPU/CUDA is required; CPU fallback is forbidden")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    print(f"[NDP] device={torch.cuda.get_device_name(args.device)} index={args.device}")
    print(f"[NDP] output={args.output}")
    print(f"[NDP] prior={PRIOR}")

    cfg = configure()
    payload = load_payload(RANK_STATE)
    print("[NDP] rank_state_loaded", flush=True)
    dataset = base.build_dataset(cfg)
    print(f"[NDP] dataset_loaded train={len(dataset.train)}", flush=True)
    path_to_index = {str(item["image_path"]): i for i, item in enumerate(dataset.train)}
    bank = load_fast_bank()
    print(f"[NDP] fast_relation_bank_loaded pairs={len(bank['exact_lookup'])}", flush=True)
    all_targets = load_targets(PRIOR, 32)
    targets = all_targets[args.offset:args.offset + args.limit]
    print(f"[NDP] targets={len(targets)}")
    found = collect_batches(dataset, cfg, targets)
    print(f"[NDP] matched_batches={len(found)}")
    neighborhoods = {k: v for k, v in np.load(PRIOR / "LTCC_FIXED_NEIGHBORHOODS_K10.npz").items()}
    _, transform_test = base.build_img_transforms(cfg)
    model = model_from_state(cfg, payload["model"], device)
    # Use one current-runtime RNG snapshot for every paired branch, as in the
    # recovery experiment. The historical PPU RNG tensor is not portable.
    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    pair_rng = snapshot_rng()
    rows: list[dict[str, Any]] = []
    taylor: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    for pos, target in enumerate(targets):
        repetitions = args.repeat_count if pos < args.repeat_limit else 1
        for repetition in range(repetitions):
            print(f"[NDP] relation={pos+1}/{len(targets)} repeat={repetition+1}/{repetitions}", flush=True)
            row, trows, states = run_one(model, cfg, payload, dataset, target, found[target["key"]], path_to_index, bank, neighborhoods, device, transform_test, pair_rng, repetition)
            rows.append(row); taylor.extend(trows)
            param_rows.append({k: row[k] for k in ("candidate_index", "relation_key", "pid", "repetition", "delta_theta_norm", "expected_gpc_update_norm", "update_expected_cosine", "update_residual_norm", "update_relative_residual", "gpc_gradient_norm", "gpc_delta_dot", "gpc_delta_cosine", "gpc_gradient_parameter_coverage", "full_inf", "masked_inf", "full_scale_before", "full_scale_after", "masked_scale_before", "masked_scale_after")})
    write_csv(args.output / "CALIBRATION_RELATIONS.csv", rows)
    write_csv(args.output / "TAYLOR_APPROXIMATION_RESULTS.csv", taylor)
    write_csv(args.output / "PARAMETER_UPDATE_AUDIT.csv", param_rows)
    manifest = {
        "created_beijing": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "rank_state": str(RANK_STATE), "rank_state_sha256": sha256_file(RANK_STATE),
        "prior_output": str(PRIOR), "dataset": "LTCC train only", "seed": 1234,
        "device": torch.cuda.get_device_name(args.device), "device_index": args.device, "relations": len(targets), "repeat_count": args.repeat_count,
        "repeated_relations": min(args.repeat_limit, len(targets)), "alphas": list(ALPHAS),
        "historical_batch_identity": "pair-key matched; historical batch iteration retained for audit, not asserted identical",
        "gpc": {"selector": "Phase7I I3 SHA-256 selected ID pairs", "lambda": LAMBDA, "denominator": "original selected relation count", "base_loss_unchanged": True},
        "no_checkpoint_write": True, "query_gallery_used": False,
    }
    (args.output / "CALIBRATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[NDP] wrote {len(rows)} relation rows and {len(taylor)} Taylor rows")


if __name__ == "__main__":
    main()
