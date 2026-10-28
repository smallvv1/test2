#!/usr/bin/env python3
"""NDP risk measurability and structure-loss activation audit.

This is a diagnostic-only continuation from the recovered LTCC I3
DIFFER+GPC state.  It deliberately keeps the Phase 7I selector and the
original GPC denominator.  No checkpoint is written and no formal training
schedule is started.

The PPU experiment uses one shared full-GPC branch as the reference for the
single-relation and all-relation comparisons.  A second full-GPC branch,
started from the identical model/optimizer/RNG state, measures numerical
noise.  Thus a target requires four virtual optimizer steps: full-reference,
full-replicate, single-mask, and all-mask.
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
import ndp_dynamic_calibration as cal  # noqa: E402
from ndp_structure_losses import ndp_margin_loss, ndp_negative_intrusion_loss  # noqa: E402

base = p7.base
LAMBDA_GPC = 0.1
PRIOR = cal.PRIOR
CHECKPOINT = cal.CHECKPOINT
RANK_STATE = cal.RANK_STATE


def json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().float().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_value(v) for k, v in row.items()})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in ("torch_cpu", "torch_cuda"):
        value = state[key]
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def float_value(x: torch.Tensor | float) -> float:
    return float(x.detach().float().cpu()) if torch.is_tensor(x) else float(x)


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


def clone_batch_with_index(batch: dict[str, Any], path_to_index: dict[str, int]) -> dict[str, Any]:
    out = cal.clone_batch(batch)
    out["_global_index"] = [path_to_index[str(path)] for path in out["image_path"]]
    return out


def selected_positions(batch: dict[str, Any], path_to_index: dict[str, int], bank: dict[str, Any], iteration: int) -> tuple[list[tuple[int, int]], int, dict[str, Any]]:
    entries = [
        {"pid": int(batch["pid"][i]), "clothes_id": int(batch["clothes_id"][i]), "image_path": str(batch["image_path"][i])}
        for i in range(len(batch["pid"]))
    ]
    chosen, count, info = p7.selected_relation_pairs(entries, path_to_index, bank, "I3", 60, 0, iteration)
    positions = [cal.canonical_key(path_to_index[str(batch["image_path"][i])], path_to_index[str(batch["image_path"][j])]) for i, j, _key, _state in chosen["id"]]
    return positions, int(count), info


def optimizer_step_mode(
    model: torch.nn.Module,
    cfg: Any,
    payload: dict[str, Any],
    batch: dict[str, Any],
    target_index: int,
    selected: list[tuple[int, int]],
    device: torch.device,
    rng: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """One paired virtual update: full, single-mask, or all-mask."""
    loss_fn, optimizer, scheduler, scaler = cal.reset_runtime(model, cfg, payload, device)
    cal.restore_rng(rng)
    out = cal.gpc_relation_loss(model, loss_fn, batch, device, target_index, selected)
    if mode == "full":
        total = out["total_full"]
    elif mode == "single_mask":
        total = out["total_masked"]
    elif mode == "all_mask":
        # Keep the relation graph evaluated, but remove every GPC contribution.
        # This is an overall-GPC diagnostic and is not a single-relation result.
        total = out["base_loss"] + LAMBDA_GPC * out["rel_values"].sum() * 0.0
    else:
        raise ValueError(mode)
    scaler.scale(total).backward()
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    inf = float(base.found_inf(scaler, optimizer))
    scaler.update()
    state = cal.snapshot_state(model)
    return {
        "mode": mode,
        "state": state,
        "loss": float_value(total),
        "base_loss": float_value(out["base_loss"]),
        "gpc_mean": float_value(out["relation_full"]),
        "target_pair_loss": float_value(out["rel_values"][target_index]),
        "pair_count": int(len(out["rel_values"])),
        "inf": inf,
        "scale_before": scale_before,
        "scale_after": float(scaler.get_scale()),
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def probe_features(model: torch.nn.Module, images: torch.Tensor, cams: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), amp.autocast(enabled=True):
        output = model(images, cam_label=cams)
    feat = output[1][0] if isinstance(output, tuple) else output
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    return F.normalize(feat.float(), dim=-1)


def local_probe(
    model: torch.nn.Module,
    images: torch.Tensor,
    cams: torch.Tensor,
    positions: dict[int, int],
    left: int,
    right: int,
    neighbors_left: np.ndarray,
    neighbors_right: np.ndarray,
    pool_left: np.ndarray,
) -> tuple[dict[str, float], torch.Tensor]:
    z = probe_features(model, images, cams)
    li, ri = positions[left], positions[right]
    lni = torch.tensor([positions[int(x)] for x in neighbors_left], device=z.device)
    rni = torch.tensor([positions[int(x)] for x in neighbors_right], device=z.device)
    pool = torch.tensor([positions[int(x)] for x in pool_left], device=z.device)
    pos = torch.sum(z[li] * z[ri])
    left_scores = torch.mv(z[lni], z[li])
    right_scores = torch.mv(z[rni], z[ri])
    pool_scores = torch.mv(z[pool], z[li])
    hard = pos - left_scores.max()
    top_k = min(10, len(pool_left))
    top_order = torch.argsort(pool_scores, descending=True)[:top_k].detach().cpu().numpy().tolist()
    top_set = {int(pool_left[x]) for x in top_order}
    positive_rank = int((pool_scores > pos).sum().detach().cpu().item()) + 1
    return {
        "positive_similarity": float_value(pos),
        "single_triple_margin": float_value(pos - left_scores[0]),
        "hard_margin_left": float_value(hard),
        "average_margin_left": float_value(pos - left_scores.mean()),
        "minimum_margin_two_sides": float_value(torch.minimum(hard, pos - right_scores.max())),
        "average_margin_two_sides": float_value((hard + pos - right_scores.max()) / 2.0),
        "hard_negative_similarity_left": float_value(left_scores.max()),
        "hard_negative_similarity_right": float_value(right_scores.max()),
        "positive_rank_against_fixed_pool": positive_rank,
        "initially_correct_fixed_pool": int(bool((pool_scores < pos).all().detach().cpu())),
        "top10_size": len(top_set),
    }, z


def buffer_diff(baseline: dict[str, torch.Tensor], state: dict[str, torch.Tensor], model: torch.nn.Module) -> tuple[float, int, float]:
    names = {name for name, _ in model.named_buffers()}
    diffs = [(name, (state[name].float() - baseline[name].float()).abs()) for name in names if name in state and name in baseline]
    max_abs = max((float(x.max().cpu()) for _, x in diffs if x.numel()), default=0.0)
    changed = sum(int(bool((x > 0).any().cpu())) for _, x in diffs)
    running = [x for name, x in diffs if "running_" in name]
    running_max = max((float(x.max().cpu()) for x in running if x.numel()), default=0.0)
    return max_abs, changed, running_max


def parameter_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], model: torch.nn.Module) -> tuple[float, float, float]:
    values = [(b[name].float() - a[name].float()) for name, p in model.named_parameters() if p.requires_grad]
    norm = math.sqrt(sum(float((x * x).sum().cpu()) for x in values)) if values else 0.0
    max_abs = max((float(x.abs().max().cpu()) for x in values if x.numel()), default=0.0)
    return norm, max_abs, len(values)


def make_probe(dataset: Any, target_key: tuple[int, int], neighborhoods: dict[str, np.ndarray], transform_test: Any, device: torch.device) -> dict[str, Any]:
    left, right = target_key
    neigh_l = neighborhoods["neighbors"][left]
    neigh_r = neighborhoods["neighbors"][right]
    pool_l = neighborhoods.get("pool", neighborhoods["neighbors"])[left]
    pos = int(neighborhoods["positive_partner"][left])
    indices = sorted(set([left, right, pos, *[int(x) for x in neigh_l], *[int(x) for x in neigh_r], *[int(x) for x in pool_l]]))
    images, cams = cal.probe_tensors(dataset, indices, transform_test, device)
    return {"indices": indices, "images": images, "cams": cams, "positions": {x: i for i, x in enumerate(indices)}, "left": left, "right": right, "pos": pos, "neigh_l": neigh_l, "neigh_r": neigh_r, "pool_l": pool_l}


def run_paired(args: argparse.Namespace, output: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("PPU/CUDA is required; CPU model fallback is forbidden")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    cfg = cal.configure()
    payload = cal.load_payload(RANK_STATE)
    dataset = base.build_dataset(cfg)
    path_to_index = {str(x["image_path"]): i for i, x in enumerate(dataset.train)}
    bank = cal.load_fast_bank()
    targets = cal.load_targets(PRIOR, args.limit)[args.offset:args.offset + args.limit]
    found = cal.collect_batches(dataset, cfg, targets)
    neighborhoods = {k: v for k, v in np.load(PRIOR / "LTCC_FIXED_NEIGHBORHOODS_K10.npz").items()}
    _, transform_test = base.build_img_transforms(cfg)
    model = cal.model_from_state(cfg, payload["model"], device)
    baseline_state = clone_state(payload["model"])
    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    pair_rng = cal.snapshot_rng()
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for pos, target in enumerate(targets):
        key = target["key"]
        found_item = found[key]
        batch = clone_batch_with_index(found_item["batch"], path_to_index)
        selected, count, info = selected_positions(batch, path_to_index, bank, found_item["iteration_replayed"])
        target_index = selected.index(key)
        probe = make_probe(dataset, key, neighborhoods, transform_test, device)
        model.load_state_dict(baseline_state, strict=True); model.eval()
        base_metrics, base_z = local_probe(model, probe["images"], probe["cams"], probe["positions"], probe["left"], probe["right"], probe["neigh_l"], probe["neigh_r"], probe["pool_l"])
        # All four branches reset model, optimizer, scheduler, scaler and RNG.
        full = optimizer_step_mode(model, cfg, payload, batch, target_index, selected, device, pair_rng, "full")
        full_rep = optimizer_step_mode(model, cfg, payload, batch, target_index, selected, device, pair_rng, "full")
        single = optimizer_step_mode(model, cfg, payload, batch, target_index, selected, device, pair_rng, "single_mask")
        all_mask = optimizer_step_mode(model, cfg, payload, batch, target_index, selected, device, pair_rng, "all_mask")
        branch_metrics: dict[str, dict[str, float]] = {}
        branch_top: dict[str, set[int]] = {}
        for name, result in (("full", full), ("full_replicate", full_rep), ("single_mask", single), ("all_mask", all_mask)):
            model.load_state_dict(result["state"], strict=True); model.eval()
            metrics, _z = local_probe(model, probe["images"], probe["cams"], probe["positions"], probe["left"], probe["right"], probe["neigh_l"], probe["neigh_r"], probe["pool_l"])
            branch_metrics[name] = metrics
            # Reconstruct top-10 from a second deterministic feature call.
            pool_indices = torch.tensor([probe["positions"][int(x)] for x in probe["pool_l"]], device=device)
            li = probe["positions"][probe["left"]]
            z = probe_features(model, probe["images"], probe["cams"])
            scores = torch.mv(z[pool_indices], z[li])
            order = torch.argsort(scores, descending=True)[:min(10, len(probe["pool_l"]))].detach().cpu().numpy().tolist()
            branch_top[name] = {int(probe["pool_l"][i]) for i in order}
        noise_top_jaccard = len(branch_top["full"] & branch_top["full_replicate"]) / max(1, len(branch_top["full"] | branch_top["full_replicate"]))
        common = {"candidate_index": target["candidate_index"], "relation_key": f"{key[0]}:{key[1]}", "pid": target["pid"], "clothes_condition": target["clothes_condition"], "selected_count": count, "target_relation_index": target_index, "gpc_target_loss": full["target_pair_loss"], "gpc_mean_full": full["gpc_mean"], "base_loss": full["base_loss"], "batch_sampler_identity": found_item["meta"]["sampler_identity"]}
        metric_names = ["hard_margin_left", "minimum_margin_two_sides", "average_margin_two_sides", "positive_similarity", "hard_negative_similarity_left", "single_triple_margin", "positive_rank_against_fixed_pool", "initially_correct_fixed_pool"]
        for metric in metric_names:
            base_v = base_metrics[metric]
            for comparison, label in (("full_replicate", "A_noise_full_minus_full"), ("single_mask", "B_full_minus_single"), ("all_mask", "C_full_minus_all")):
                rows.append({**common, "comparison": label, "metric": metric, "baseline": base_v, "full": branch_metrics["full"][metric], "comparison_value": branch_metrics[comparison][metric], "delta_full_minus_comparison": branch_metrics["full"][metric] - branch_metrics[comparison][metric], "delta_comparison_minus_baseline": branch_metrics[comparison][metric] - base_v, "full_inf": full["inf"], "comparison_inf": (full_rep if comparison == "full_replicate" else single if comparison == "single_mask" else all_mask)["inf"], "full_scale_before": full["scale_before"], "comparison_scale_before": (full_rep if comparison == "full_replicate" else single if comparison == "single_mask" else all_mask)["scale_before"], "rng_hash": state_hash(pair_rng)})
        for comparison, label in (("full_replicate", "A_noise"), ("single_mask", "B_single_relation"), ("all_mask", "C_all_GPC")):
            branch = full_rep if comparison == "full_replicate" else single if comparison == "single_mask" else all_mask
            pnorm, pmax, pcount = parameter_diff(branch["state"], full["state"], model)
            bmax, bchanged, brunning = buffer_diff(baseline_state, branch["state"], model)
            rows.append({**common, "comparison": label, "metric": "fixed_pool_top10_jaccard", "baseline": 1.0, "full": 1.0, "comparison_value": len(branch_top["full"] & branch_top[comparison]) / max(1, len(branch_top["full"] | branch_top[comparison])), "delta_full_minus_comparison": noise_top_jaccard if comparison == "full_replicate" else 1.0 - len(branch_top["full"] & branch_top[comparison]) / max(1, len(branch_top["full"] | branch_top[comparison])), "delta_comparison_minus_baseline": 0.0, "full_inf": full["inf"], "comparison_inf": branch["inf"], "rng_hash": state_hash(pair_rng)})
            audit_rows.append({**common, "comparison": label, "parameter_diff_norm": pnorm, "parameter_diff_max_abs": pmax, "trainable_parameter_tensor_count": pcount, "buffer_diff_max_abs_from_checkpoint": bmax, "buffer_tensors_changed": bchanged, "bn_running_buffer_diff_max_abs": brunning, "full_loss": full["loss"], "comparison_loss": branch["loss"], "full_base_loss": full["base_loss"], "comparison_base_loss": branch["base_loss"], "full_gpc_mean": full["gpc_mean"], "comparison_gpc_mean": branch["gpc_mean"], "full_inf": full["inf"], "comparison_inf": branch["inf"], "full_scale_after": full["scale_after"], "comparison_scale_after": branch["scale_after"], "rng_hash": state_hash(pair_rng), "selector_count": count, "selector_info": json.dumps(info, sort_keys=True, default=str)})
        # Restore before the next target; branch state is never used as a new checkpoint.
        model.load_state_dict(baseline_state, strict=True); model.train()
        print(f"[NDP-MEAS] {pos + 1}/{len(targets)} {key} count={count} A_noise={noise_top_jaccard:.3f}", flush=True)
    cfg_info = {
        "device": torch.cuda.get_device_name(args.device), "device_index": args.device,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "rank_state": str(RANK_STATE), "rank_state_sha256": sha256_file(RANK_STATE),
        "dataset": "LTCC train only", "train_size": len(dataset.train), "targets": len(targets),
        "gpc_selector": "Phase7I I3 SHA-256", "gpc_lambda": LAMBDA_GPC, "gpc_denominator": "original selected relation count",
        "optimizer": str(cfg.SOLVER.OPTIMIZER_NAME), "momentum": float(cfg.SOLVER.MOMENTUM), "weight_decay": float(cfg.SOLVER.WEIGHT_DECAY),
        "base_lr": float(cfg.SOLVER.BASE_LR), "clip_grad": str(getattr(cfg.SOLVER, "CLIP_GRAD", "not configured")),
        "same_update_pair": "full vs full_replicate from identical model/optimizer/RNG; full reference reused for B/C",
        "historical_batch_identity": "pair-key matched selector-valid diagnostic batch; not claimed identical to old sampler batch",
        "no_checkpoint_write": True, "query_gallery_labels_used": False,
    }
    return rows, cfg_info, audit_rows


def offline_ndp_analysis(output: Path, paired_rows: list[dict[str, Any]]) -> None:
    ref = np.load(PRIOR / "LTCC_TRAIN_FEATURES_E60_I3.npy", mmap_mode="r")
    neigh = {k: v for k, v in np.load(PRIOR / "LTCC_FIXED_NEIGHBORHOODS_K10.npz").items()}
    rows: list[dict[str, Any]] = []
    partner = neigh["positive_partner"].astype(np.int64)
    first_neg = neigh["neighbors"][:, 0].astype(np.int64)
    pos_sim = np.sum(ref * ref[partner], axis=1)
    neg_sim = np.sum(ref * ref[first_neg], axis=1)
    margins = pos_sim - neg_sim
    hard_neg = np.max(np.sum(ref[:, None, :] * ref[neigh["neighbors"]], axis=2), axis=1)
    hard_margin = pos_sim - hard_neg
    for name, values in (("reference_first_triple_margin", margins), ("reference_hard_margin", hard_margin), ("reference_positive_similarity", pos_sim), ("reference_first_negative_similarity", neg_sim)):
        for q in (0.0, 0.01, 0.05, 0.1, 0.2):
            rows.append({"section": "reference_distribution", "metric": name, "threshold": q, "count": len(values), "mean": float(np.mean(values)), "std": float(np.std(values)), "p01": float(np.quantile(values, .01)), "p05": float(np.quantile(values, .05)), "p50": float(np.quantile(values, .50)), "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)), "reliable_count_if_margin": int(np.sum(values > q)) if "margin" in name else ""})
    for tau in (0.0, 0.01, 0.05, 0.1, 0.2):
        rows.append({"section": "reliable_triplet_count", "metric": "first_triple_margin_gt_tau", "threshold": tau, "count": int(np.sum(margins > tau)), "mean": "", "std": "", "p01": "", "p05": "", "p50": "", "p95": "", "p99": "", "reliable_count_if_margin": int(np.sum(margins > tau))})
    prior_rows = cal.read_csv(PRIOR.parent.parent / "ndp_design_pilot" / "20261014_201552" / "NDP_PILOT_RESULTS.csv")
    for row in prior_rows:
        if row["group"] not in ("N1_MARGIN", "N2_NEGATIVE"):
            continue
        ref_m = float(row["ref_margin"]); cur_m = float(row["current_margin_before"])
        rows.append({"section": "prior_ndp_pilot", "metric": row["group"], "threshold": 0.01, "count": 1, "mean": float(row["ndp_loss"]), "std": "", "p01": "", "p05": "", "p50": "", "p95": "", "p99": "", "reliable_count_if_margin": int(float(row["ndp_violating_count"])), "ref_margin": ref_m, "current_margin": cur_m, "margin_drop_ref_minus_current": ref_m - cur_m, "ndp_grad_norm": float(row["ndp_grad_norm"]), "ndp_active_count": int(row["ndp_active_count"])})
    # The new baseline rows include the same fixed first neighbour used by N2.
    first_by_key: dict[str, dict[str, Any]] = {}
    for row in paired_rows:
        if row["comparison"] == "A_noise" and row["metric"] == "hard_margin_left":
            first_by_key[row["relation_key"]] = row
    for key, row in first_by_key.items():
        left = int(key.split(":")[0]); p = int(neigh["positive_partner"][left]); n = int(neigh["neighbors"][left, 0])
        rpos = float(pos_sim[left]); rneg = float(neg_sim[left])
        # baseline single_triple_margin is pos_cur - neg_cur for first fixed neighbour
        cur_single = next(x for x in paired_rows if x["relation_key"] == key and x["comparison"] == "A_noise" and x["metric"] == "single_triple_margin")["baseline"]
        cur_pos = next(x for x in paired_rows if x["relation_key"] == key and x["comparison"] == "A_noise" and x["metric"] == "positive_similarity")["baseline"]
        cur_neg = float(cur_pos) - float(cur_single)
        rows.append({"section": "new_current_baseline", "metric": "fixed_first_triple", "threshold": 0.01, "count": 1, "mean": float(cur_single), "std": "", "p01": "", "p05": "", "p50": "", "p95": "", "p99": "", "reliable_count_if_margin": int(rpos > 0), "relation_key": key, "anchor": left, "positive": p, "negative": n, "ref_margin": rpos - rneg, "current_margin": float(cur_single), "margin_drop_ref_minus_current": (rpos - rneg) - float(cur_single), "ref_negative_similarity": rneg, "current_negative_similarity": cur_neg, "negative_increase_ref_to_current": cur_neg - rneg})
    write_csv(output / "NDP_INACTIVATION_ANALYSIS.csv", rows)


def controlled_activation(output: Path) -> None:
    torch.manual_seed(7)
    ref_a = F.normalize(torch.randn(4, 16), dim=1)
    ref_p = F.normalize(torch.randn(4, 16), dim=1)
    ref_n = F.normalize(torch.randn(4, 16), dim=1)
    mask = torch.ones(4, dtype=torch.bool)
    cases = []
    for name, cur_p, cur_n, kind in (
        ("no_damage", ref_p.clone(), ref_n.clone(), "none"),
        ("margin_damage", ref_p - 0.8 * ref_a, ref_n.clone(), "margin"),
        ("negative_intrusion", ref_p.clone(), ref_n + 0.8 * ref_a, "negative"),
    ):
        cur_a = ref_a.clone().requires_grad_(True)
        cur_p = cur_p.clone().requires_grad_(True)
        cur_n = cur_n.clone().requires_grad_(True)
        lm, im = ndp_margin_loss(cur_a, cur_p, cur_n, ref_a, ref_p, ref_n, epsilon=0.01, reliable_mask=mask)
        cur_a.grad = cur_p.grad = cur_n.grad = None
        ln, inn = ndp_negative_intrusion_loss(cur_a, cur_n, ref_a, ref_n, epsilon_negative=0.01, reliable_mask=mask)
        (lm + ln).backward()
        cases.append({"case": name, "damage_type": kind, "margin_loss": float(lm.detach()), "negative_loss": float(ln.detach()), "margin_active": int(im["violating_count"]), "negative_active": int(inn["violating_count"]), "combined_grad_norm": float(torch.sqrt(sum((x.grad.float() ** 2).sum() for x in (cur_a, cur_p, cur_n))).detach()), "finite": int(all(bool(torch.isfinite(x.grad).all()) for x in (cur_a, cur_p, cur_n)))})
    write_csv(output / "CONTROLLED_NDP_ACTIVATION.csv", cases)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--postprocess-existing", action="store_true", help="reuse PPU raw paired output; do not load the model")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[NDP-MEAS] output={args.output}", flush=True)
    if args.postprocess_existing:
        rows = cal.read_csv(args.output / "PAIRED_METRICS_LONG.csv")
        audits = cal.read_csv(args.output / "PARAMETER_STATE_AUDIT.csv")
        offline_ndp_analysis(args.output, rows)
        controlled_activation(args.output)
        write_csv(args.output / "A_NUMERICAL_NOISE.csv", [r for r in rows if r["comparison"] in ("A_noise", "A_noise_full_minus_full")])
        write_csv(args.output / "B_SINGLE_RELATION_IMPACT.csv", [r for r in rows if r["comparison"] in ("B_single_relation", "B_full_minus_single")])
        write_csv(args.output / "C_OVERALL_GPC_IMPACT.csv", [r for r in rows if r["comparison"] in ("C_all_GPC", "C_full_minus_all")])
        print(f"[NDP-MEAS] postprocess complete metric_rows={len(rows)} audit_rows={len(audits)}", flush=True)
        return
    rows, cfg_info, audits = run_paired(args, args.output)
    # Split the common long-form table into the three requested experiment files.
    write_csv(args.output / "PAIRED_METRICS_LONG.csv", rows)
    write_csv(args.output / "A_NUMERICAL_NOISE.csv", [r for r in rows if r["comparison"] in ("A_noise", "A_noise_full_minus_full")])
    write_csv(args.output / "B_SINGLE_RELATION_IMPACT.csv", [r for r in rows if r["comparison"] in ("B_single_relation", "B_full_minus_single")])
    write_csv(args.output / "C_OVERALL_GPC_IMPACT.csv", [r for r in rows if r["comparison"] in ("C_all_GPC", "C_full_minus_all")])
    write_csv(args.output / "PARAMETER_STATE_AUDIT.csv", audits)
    offline_ndp_analysis(args.output, rows)
    controlled_activation(args.output)
    manifest = {"created_beijing": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(), "phase": "NDP-RISK-MEASURABILITY", **cfg_info, "limit": args.limit, "offset": args.offset, "files": ["A_NUMERICAL_NOISE.csv", "B_SINGLE_RELATION_IMPACT.csv", "C_OVERALL_GPC_IMPACT.csv", "NDP_INACTIVATION_ANALYSIS.csv", "CONTROLLED_NDP_ACTIVATION.csv", "PARAMETER_STATE_AUDIT.csv"], "formal_training": False}
    (args.output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"[NDP-MEAS] complete metric_rows={len(rows)} audit_rows={len(audits)}", flush=True)


if __name__ == "__main__":
    main()
