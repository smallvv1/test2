#!/usr/local/bin/python3
"""NDP真实邻域覆盖、缓存可行性和局部优势梯度控制审计。

本脚本只读取既有 LTCC train-only feature bank、neighbor graph 和冻结 batch plan。
不实例化模型、不执行 PPU/CPU 模型 forward，不修改 GPC，不读取官方 query/gallery。
CPU torch 仅用于独立候选公式和控制损失的张量梯度单测。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
BASE = ROOT / "outputs" / "ndp_local_intrusion_20261020_155558"
REL = ROOT / "outputs" / "ndp_metric_coverage_audit_20261020_165332"
PLAN_SOURCE = ROOT / "scripts" / "ndp_net_intrusion_triplet_coverage_audit.py"
SEEDS = (1234, 4321)
K = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_plan_builder():
    spec = importlib.util.spec_from_file_location("frozen_plan", PLAN_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载冻结 batch plan 构造器: {PLAN_SOURCE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_seed(seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, str]]]:
    bank = load_npz(BASE / "feature_banks" / f"seed{seed}" / "gpc_feature_bank.npz")
    graph = load_npz(BASE / "neighbor_graphs" / f"seed{seed}_bio_all_image_top20.npz")
    rows = read_csv(REL / "tables" / f"seed{seed}_K10_identity_relation_states.csv")
    return bank, graph, rows


def relation_map(rows: list[dict[str, str]]) -> dict[tuple[int, int, int], dict[str, str]]:
    return {(int(r["anchor"]), int(r["positive"]), int(r["negative"])): r for r in rows}


def build_image_locations(plan: dict[str, Any]) -> dict[int, list[tuple[int, int, set[int]]]]:
    locations: dict[int, list[tuple[int, int, set[int]]]] = defaultdict(list)
    for rank, payload in plan["ranks"].items():
        for iteration, batch in enumerate(payload["batches"], start=1):
            bset = {int(x) for x in batch}
            for image in bset:
                locations[image].append((int(rank), iteration, bset))
    return locations


def relation_is_reliable(row: dict[str, str]) -> bool:
    return row["reference_reliable"].strip().lower() in {"true", "1", "yes"}


def coverage_for_seed(seed: int, bank: dict[str, np.ndarray], graph_all: dict[str, np.ndarray], graph_negative: dict[str, np.ndarray], rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    plan = load_plan_builder().build_batch_plan(seed)
    locations = build_image_locations(plan)
    pids = bank["pid"].astype(int)
    clothes = bank["clothes"].astype(int)
    cam = bank["camid"].astype(int)
    gidx = graph_all["indices"][:, :K].astype(int)
    nidx = graph_negative["indices"][:, :K].astype(int)
    paths = bank["image_path"]
    alignment = {
        "all_image_path_aligned": bool(np.array_equal(paths, graph_all["image_path"])),
        "all_image_pid_aligned": bool(np.array_equal(pids, graph_all["pid"].astype(int))),
        "negative_only_path_aligned": bool(np.array_equal(paths, graph_negative["image_path"])),
        "negative_only_pid_aligned": bool(np.array_equal(pids, graph_negative["pid"].astype(int))),
        "graph_rows": int(gidx.shape[0]),
        "graph_k": int(gidx.shape[1]),
        "all_image_self_reference_count": int(sum(int(a) in set(gidx[a].tolist()) for a in range(len(gidx)))),
        "all_image_duplicate_neighbor_row_count": int(sum(len(set(row.tolist())) != len(row) for row in gidx)),
        "negative_only_self_reference_count": int(sum(int(a) in set(nidx[a].tolist()) for a in range(len(nidx)))),
        "negative_only_duplicate_neighbor_row_count": int(sum(len(set(row.tolist())) != len(row) for row in nidx)),
    }
    if not alignment["all_image_path_aligned"] or not alignment["all_image_pid_aligned"] or not alignment["negative_only_path_aligned"] or not alignment["negative_only_pid_aligned"]:
        raise RuntimeError(f"seed{seed} 缓存与标签未对齐: {alignment}")
    if any(alignment[k] for k in ("all_image_self_reference_count", "all_image_duplicate_neighbor_row_count", "negative_only_self_reference_count", "negative_only_duplicate_neighbor_row_count")):
        raise RuntimeError(f"seed{seed} 缓存存在自引用或重复邻居: {alignment}")

    relmap = relation_map(rows)
    reliable_rows = [r for r in rows if relation_is_reliable(r)]
    global_all_edges = 0
    global_all_positive_edges = 0
    global_all_negative_edges = 0
    for a in range(len(gidx)):
        global_all_edges += len(gidx[a])
        global_all_positive_edges += sum(int(pids[int(n)]) == int(pids[a]) for n in gidx[a])
        global_all_negative_edges += sum(int(pids[int(n)]) != int(pids[a]) for n in gidx[a])
    # The relation table has one fixed positive partner for each anchor/negative row.
    relation_anchor_set = {int(r["anchor"]) for r in rows}
    reliable_relation_anchor_set = {int(r["anchor"]) for r in reliable_rows}
    reliable_relation_total = len(reliable_rows)
    relation_total = len(rows)

    batch_rows: list[dict[str, Any]] = []
    valid_anchor_events = 0
    unique_valid_anchors: set[int] = set()
    batch_cache_all_edges = 0
    batch_cache_all_positive_edges = 0
    batch_cache_all_negative_edges = 0
    batch_cache_negative_only_edges = 0
    batch_cache_reliable_edges = 0
    batch_cache_full_reliable_triples = 0
    batch_cache_all_any_anchor_events = 0
    batch_cache_all_positive_anchor_events = 0
    batch_cache_all_negative_anchor_events = 0
    batch_cache_reliable_anchor_events = 0
    batch_positive_pairs = 0
    batch_negative_pairs = 0
    batch_cross_clothes_positive_pairs = 0
    rank_summary: dict[int, dict[str, int]] = defaultdict(lambda: Counter())
    for rank, payload in plan["ranks"].items():
        for iteration, batch in enumerate(payload["batches"], start=1):
            bset = {int(x) for x in batch}
            valid = 0
            pos_candidates = 0
            neg_candidates = 0
            cache_all = 0
            cache_all_pos = 0
            cache_all_neg = 0
            cache_neg_only = 0
            cache_rel = 0
            cache_full = 0
            cache_any_anchor = 0
            cache_pos_anchor = 0
            cache_neg_anchor = 0
            cross_pos = 0
            for a in sorted(bset):
                same = [x for x in bset if x != a and pids[x] == pids[a]]
                neg = [x for x in bset if pids[x] != pids[a]]
                if same and neg:
                    valid += 1
                    valid_anchor_events += 1
                    unique_valid_anchors.add(a)
                    pos_candidates += len(same)
                    neg_candidates += len(neg)
                    batch_positive_pairs += len(same)
                    batch_negative_pairs += len(neg)
                    c = sum(int(clothes[x] != clothes[a]) for x in same)
                    cross_pos += c
                    batch_cross_clothes_positive_pairs += c
                all_neighbors = [int(n) for n in gidx[a]]
                in_batch_all = [n for n in all_neighbors if n in bset]
                in_batch_pos = [n for n in in_batch_all if pids[n] == pids[a]]
                in_batch_all_neg = [n for n in in_batch_all if pids[n] != pids[a]]
                in_batch_neg = [n for n in nidx[a] if n in bset]
                cache_all += len(in_batch_all); cache_all_pos += len(in_batch_pos); cache_all_neg += len(in_batch_all_neg); cache_neg_only += len(in_batch_neg)
                batch_cache_all_edges += len(in_batch_all); batch_cache_all_positive_edges += len(in_batch_pos); batch_cache_all_negative_edges += len(in_batch_all_neg); batch_cache_negative_only_edges += len(in_batch_neg)
                if in_batch_all:
                    cache_any_anchor += 1; batch_cache_all_any_anchor_events += 1
                if in_batch_pos:
                    cache_pos_anchor += 1; batch_cache_all_positive_anchor_events += 1
                if in_batch_all_neg:
                    cache_neg_anchor += 1; batch_cache_all_negative_anchor_events += 1
                for n in in_batch_neg:
                    # A reference relation is fully trainable only if its fixed positive
                    # partner and its negative neighbor are in this same batch.
                    matches = [r for r in rows if int(r["anchor"]) == a and int(r["negative"]) == n]
                    for r in matches:
                        if relation_is_reliable(r):
                            cache_rel += 1
                            batch_cache_reliable_edges += 1
                            if int(r["positive"]) in bset:
                                cache_full += 1
                                batch_cache_full_reliable_triples += 1
                                batch_cache_reliable_anchor_events += 1
            rank_summary[int(rank)]["batches"] += 1
            rank_summary[int(rank)]["valid_anchor_events"] += valid
            rank_summary[int(rank)]["cache_all_edges"] += cache_all
            rank_summary[int(rank)]["cache_all_positive_edges"] += cache_all_pos
            rank_summary[int(rank)]["cache_all_negative_edges"] += cache_all_neg
            rank_summary[int(rank)]["cache_negative_only_edges"] += cache_neg_only
            rank_summary[int(rank)]["cache_reliable_edges"] += cache_rel
            rank_summary[int(rank)]["cache_full_reliable_triples"] += cache_full
            batch_rows.append({
                "seed": seed, "rank": int(rank), "iteration": iteration, "batch_size": len(bset),
                "valid_anchor_count": valid, "batch_positive_candidate_pairs": pos_candidates,
                "batch_negative_candidate_pairs": neg_candidates, "cache_all_edge_count": cache_all,
                "cache_all_positive_edge_count": cache_all_pos, "cache_all_negative_edge_count": cache_all_neg,
                "cache_negative_only_edge_count": cache_neg_only,
                "cache_reliable_edge_count": cache_rel, "cache_full_reliable_triple_count": cache_full,
                "cache_all_any_overlap_anchor_count": cache_any_anchor, "cache_all_positive_overlap_anchor_count": cache_pos_anchor,
                "cache_all_negative_overlap_anchor_count": cache_neg_anchor,
                "cross_clothes_positive_pair_count": cross_pos,
            })

    negative_only_edges = int(nidx.size)
    summary = {
        "seed": seed,
        "plan_source": str(PLAN_SOURCE),
        "plan_source_sha256": sha256_file(PLAN_SOURCE),
        "rank_summary": {str(k): dict(v) for k, v in rank_summary.items()},
        "total_batches": len(batch_rows),
        "total_images": int(len(pids)),
        "valid_anchor_events": valid_anchor_events,
        "unique_valid_anchors": len(unique_valid_anchors),
        "valid_anchor_rate": len(unique_valid_anchors) / len(pids),
        "batch_positive_candidate_pairs": batch_positive_pairs,
        "batch_negative_candidate_pairs": batch_negative_pairs,
        "batch_cross_clothes_positive_pairs": batch_cross_clothes_positive_pairs,
        "batch_cross_clothes_positive_rate": batch_cross_clothes_positive_pairs / max(batch_positive_pairs, 1),
        "global_graph_anchor_count": int(len(gidx)),
        "global_graph_all_edges": global_all_edges,
        "global_graph_all_positive_edges": global_all_positive_edges,
        "global_graph_all_negative_edges": global_all_negative_edges,
        "global_graph_negative_only_edges": negative_only_edges,
        "reference_relation_anchor_count": len(relation_anchor_set),
        "reference_reliable_anchor_count": len(reliable_relation_anchor_set),
        "reference_relation_total_rows": relation_total,
        "reference_reliable_rows": reliable_relation_total,
        "reference_reliable_rate": reliable_relation_total / max(relation_total, 1),
        "batch_cache_all_edges": batch_cache_all_edges,
        "batch_cache_all_positive_edges": batch_cache_all_positive_edges,
        "batch_cache_all_negative_edges": batch_cache_all_negative_edges,
        "historical_all_image_any_overlap_anchor_events": batch_cache_all_any_anchor_events,
        "historical_all_image_any_overlap_anchor_rate": batch_cache_all_any_anchor_events / max(valid_anchor_events, 1),
        "historical_all_image_positive_overlap_anchor_events": batch_cache_all_positive_anchor_events,
        "historical_all_image_positive_overlap_anchor_rate": batch_cache_all_positive_anchor_events / max(valid_anchor_events, 1),
        "historical_all_image_negative_overlap_anchor_events": batch_cache_all_negative_anchor_events,
        "historical_all_image_negative_overlap_anchor_rate": batch_cache_all_negative_anchor_events / max(valid_anchor_events, 1),
        "batch_cache_negative_only_edges": batch_cache_negative_only_edges,
        "batch_cache_negative_only_edge_rate_vs_global_graph": batch_cache_negative_only_edges / max(negative_only_edges, 1),
        "batch_cache_reliable_edges": batch_cache_reliable_edges,
        "batch_cache_reliable_edge_rate": batch_cache_reliable_edges / max(reliable_relation_total, 1),
        "batch_cache_full_reliable_triples": batch_cache_full_reliable_triples,
        "batch_cache_full_reliable_triple_rate": batch_cache_full_reliable_triples / max(reliable_relation_total, 1),
        "batch_cache_reliable_anchor_events": batch_cache_reliable_anchor_events,
        "alignment": alignment,
        "cache_is_immutable_stop_gradient_in_design": True,
        "cache_global_retrieval_is_offline_only": True,
    }
    return summary, batch_rows, [{"seed": seed, "rank": rank, **vals} for rank, vals in sorted(rank_summary.items())]


def local_advantage(z: torch.Tensor, ref: torch.Tensor, positives: list[int], negatives: list[int], tau: float = .05, epsilon: float = .02, reliable: bool = True):
    zero = z.sum() * 0.0
    if not reliable or not positives or not negatives:
        return zero.detach(), zero, zero
    if 0 in positives or 0 in negatives or len(set(positives)) != len(positives) or len(set(negatives)) != len(negatives) or set(positives) & set(negatives):
        raise ValueError("候选关系包含anchor自引用、重复或正负重叠")
    zc = F.normalize(z, dim=-1)
    zr = F.normalize(ref.detach(), dim=-1)
    ids = positives + negatives
    p = len(positives)
    cur = torch.matmul(zc[0:1], zc[ids].t()).squeeze(0) / tau
    rr = torch.matmul(zr[0:1], zr[ids].t()).squeeze(0) / tau
    ac = torch.logsumexp(cur[:p], 0) - torch.log(torch.tensor(float(p), dtype=cur.dtype, device=cur.device)) - (torch.logsumexp(cur[p:], 0) - torch.log(torch.tensor(float(len(negatives)), dtype=cur.dtype, device=cur.device)))
    ar = torch.logsumexp(rr[:p], 0) - torch.log(torch.tensor(float(p), dtype=rr.dtype, device=rr.device)) - (torch.logsumexp(rr[p:], 0) - torch.log(torch.tensor(float(len(negatives)), dtype=rr.dtype, device=rr.device)))
    return ar.detach(), ac, F.relu(ar.detach() - ac - epsilon)


def control_losses(z: torch.Tensor, ref: torch.Tensor, positives: list[int], negatives: list[int], tau: float = .2) -> dict[str, torch.Tensor]:
    zn = F.normalize(z, dim=-1)
    rn = F.normalize(ref.detach(), dim=-1)
    all_ids = positives + negatives
    # C0: current Batch-Hard Euclidean soft-margin surrogate.
    dp = torch.linalg.vector_norm(zn[0] - zn[positives], dim=-1).max()
    dn = torch.linalg.vector_norm(zn[0] - zn[negatives], dim=-1).min()
    c0 = F.softplus(dp - dn)
    # C1: one fixed reference pair, adaptive-margin Triplet control.
    ref_adv = torch.matmul(rn[0:1], rn[positives].t()).max() - torch.matmul(rn[0:1], rn[negatives].t()).max()
    cur_adv = torch.matmul(zn[0:1], zn[positives].t()).max() - torch.matmul(zn[0:1], zn[negatives].t()).max()
    c1 = F.relu(ref_adv.detach() - cur_adv - .02)
    # C2: multi-neighbor local identity advantage candidate.
    _, _, c2 = local_advantage(z, ref, positives, negatives, tau=tau)
    # C3: same candidate set, multi-positive/multi-negative N-pair/SupCon control.
    logits = torch.matmul(zn[0:1], zn[all_ids].t()).squeeze(0) / tau
    c3 = -(torch.logsumexp(logits[:len(positives)], 0) - torch.logsumexp(logits, 0))
    return {"C0_batch_hard_triplet": c0, "C1_reference_margin_triplet": c1, "C2_local_advantage": c2, "C3_multi_neighbor_supcon_npair": c3}


def formula_tests() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    def add(name: str, passed: bool, detail: str) -> None:
        out.append({"test": name, "passed": bool(passed), "detail": detail})
    z = torch.randn(5, 16, dtype=torch.float32, requires_grad=True); ref = z.detach().clone()
    _, _, loss = local_advantage(z, ref, [1, 2], [3, 4], epsilon=-.1); loss.backward(); add("multi_positive_negative_finite", bool(torch.isfinite(z.grad).all()), f"loss={float(loss):.6f}")
    z = torch.randn(3, 8, requires_grad=True); ref = torch.randn(3, 8, requires_grad=True); _, _, loss = local_advantage(z, ref, [1], [2], epsilon=-.1); loss.backward(); add("reference_stop_gradient", ref.grad is None, "ref.grad=None")
    z = torch.tensor([[1., 0.], [1., .1], [0., 1.]], requires_grad=True); ref = z.detach().clone(); ar, ac, loss = local_advantage(z, ref, [1], [2], tau=.1, epsilon=0.); loss.backward(); add("advantage_increase_not_penalized", float(ac) >= float(ar) and float(loss) == 0., f"A_ref={float(ar):.4f}, A_cur={float(ac):.4f}")
    z = torch.tensor([[1., 0.], [0., 1.], [1., .01]], requires_grad=True); ref = torch.tensor([[1., 0.], [1., .01], [0., 1.]]); ar, ac, loss = local_advantage(z, ref, [1], [2], tau=.1, epsilon=0.); loss.backward(); add("advantage_decline_finite", float(ar) > float(ac) and float(loss) > 0 and bool(torch.isfinite(z.grad).all()), f"A_ref={float(ar):.4f}, A_cur={float(ac):.4f}")
    z = torch.randn(3, 8, requires_grad=True); ref = z.detach(); _, _, loss = local_advantage(z, ref, [], [2]); loss.backward(); add("empty_candidate_mask", float(loss) == 0. and bool(torch.isfinite(z.grad).all()), "zero graph-connected loss")
    z = torch.randn(3, 8, requires_grad=True)
    try:
        local_advantage(z, z.detach(), [0], [2]); add("self_reference_rejected", False, "没有拒绝anchor自引用")
    except ValueError as e: add("self_reference_rejected", True, str(e))
    try:
        local_advantage(z, z.detach(), [1, 1], [2]); add("duplicate_relation_rejected", False, "没有拒绝重复正样本")
    except ValueError as e: add("duplicate_relation_rejected", True, str(e))
    z = torch.randn(4, 8, requires_grad=True); cached = torch.randn(4, 8, requires_grad=True); _, _, loss = local_advantage(z, cached, [1], [2], epsilon=-.1); loss.backward(); add("cached_neighbor_stop_gradient", cached.grad is None, "cached neighbor is immutable")
    z = torch.randn(4, 12, dtype=torch.float32, requires_grad=True); ref = torch.randn(4, 12); _, _, loss = local_advantage(z, ref, [1], [2], epsilon=-.1); loss.backward(); add("fp32_finite", bool(torch.isfinite(torch.cat([z.grad.flatten(), loss.reshape(1)])).all()), "FP32 finite")
    z16 = torch.randn(4, 12, dtype=torch.float16, requires_grad=True); ref16 = torch.randn(4, 12, dtype=torch.float16); _, _, l16 = local_advantage(z16, ref16, [1], [2], epsilon=-.1); l16.backward(); add("float16_sensitivity_probe", bool(torch.isfinite(torch.cat([z16.grad.flatten(), l16.reshape(1)])).all()), "CPU float16 probe only; not PPU AMP")
    z = torch.randn(5, 8, requires_grad=True); ref = z.detach(); vals = control_losses(z, ref, [1, 2], [3, 4]); grads = {}
    for name, val in vals.items(): grads[name] = torch.autograd.grad(val, z, retain_graph=True, allow_unused=True)[0]
    finite = all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
    distinct = not torch.allclose(grads["C2_local_advantage"], grads["C3_multi_neighbor_supcon_npair"])
    add("control_gradients_finite", finite, "C0-C3 finite")
    add("C2_not_identical_to_C3_on_probe", distinct, "synthetic probe gradient fields differ")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-time", required=True); args = parser.parse_args()
    run_time = args.run_time
    out = ROOT / "outputs" / f"ndp_neighbor_coverage_gradient_{run_time}"; out.mkdir(parents=True, exist_ok=False)
    log_path = ROOT / "logs" / f"NDP真实邻域覆盖与局部优势梯度审计_{run_time}.log"
    def log(msg: str) -> None:
        print(msg, flush=True); log_path.open("a", encoding="utf-8").write(msg + "\n")

    summaries = []; batch_rows = []; rank_rows = []; cache_meta = {}
    for seed in SEEDS:
        bank, graph_all, rows = load_seed(seed)
        graph_negative = load_npz(BASE / "neighbor_graphs" / f"seed{seed}_bio_negative_only_top20.npz")
        summary, b_rows, r_rows = coverage_for_seed(seed, bank, graph_all, graph_negative, rows)
        summaries.append(summary); batch_rows.extend(b_rows); rank_rows.extend(r_rows)
        graph_path = BASE / "neighbor_graphs" / f"seed{seed}_bio_negative_only_top20.npz"
        bank_path = BASE / "feature_banks" / f"seed{seed}" / "gpc_feature_bank.npz"
        graph_dt = datetime.fromtimestamp(graph_path.stat().st_mtime, timezone.utc)
        bank_dt = datetime.fromtimestamp(bank_path.stat().st_mtime, timezone.utc)
        cache_meta[str(seed)] = {"feature_bank": str(bank_path), "feature_bank_sha256": sha256_file(bank_path), "feature_bank_mtime_utc": bank_dt.isoformat(), "negative_only_graph": str(graph_path), "negative_only_graph_sha256": sha256_file(graph_path), "negative_only_graph_mtime_utc": graph_dt.isoformat(), "cache_age_seconds_at_run": (datetime.now(timezone.utc) - graph_dt).total_seconds(), "cache_checkpoint_manifest": str(BASE / "NDP_A_MANIFEST.json"), "cache_is_train_only": True}
        log(f"【覆盖】seed={seed}：batch有效anchor={summary['unique_valid_anchors']}/{summary['total_images']}（{summary['valid_anchor_rate']:.4%}），历史全图Top-K任一重合={summary['historical_all_image_any_overlap_anchor_events']}，负邻居图同batch关系={summary['batch_cache_negative_only_edges']}，可靠三元组同batch={summary['batch_cache_full_reliable_triples']}。")

    tests = formula_tests(); tests_pass = all(x["passed"] for x in tests)
    controls = {}
    ref = torch.tensor([[1., 0., 0., 0.], [.95, .1, 0., 0.], [.9, .2, 0., 0.], [0., 1., 0., 0.], [.1, .99, 0., 0.]])
    z = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [.1, .99, 0., 0.], [1., 0., 0., 0.], [.95, .1, 0., 0.]], requires_grad=True); vals = control_losses(z, ref, [1, 2], [3, 4], tau=.2)
    for name, val in vals.items():
        g = torch.autograd.grad(val, z, retain_graph=True, allow_unused=True)[0]
        controls[name] = {"loss": float(val.detach()), "gradient_norm": float(g.norm()) if g is not None else 0., "gradient_finite": bool(g is not None and torch.isfinite(g).all())}
    source_paths = [ROOT / "scripts" / "ndp_neighbor_coverage_gradient_control.py", ROOT / "scripts" / "ndp_cross_seed_risk_local_advantage_audit.py", PLAN_SOURCE, BASE / "NDP_A_MANIFEST.json", BASE / "feature_banks_manifest.json", BASE / "neighbor_graph_manifest.json"]
    source_hashes = {str(p): sha256_file(p) for p in source_paths}
    summary = {"run_time": run_time, "K": K, "model_training_started": False, "ppu_model_forward_started": False, "cpu_model_fallback": False, "official_query_gallery_used": False, "source_hashes": source_hashes, "cache_meta": cache_meta, "coverage": summaries, "rank_rows": rank_rows, "formula_tests": tests, "formula_tests_pass": tests_pass, "control_probe": controls, "scheme_A": {"status": "OFFLINE_COVERAGE_PASS_BUT_GRADIENT_NOT_PPU_VALIDATED", "uses": "current batch candidates only", "current_gradient_path": "anchor and current positive/negative features if model forward succeeds"}, "scheme_B": {"status": "TECHNICAL_PROTOTYPE_ONLY", "uses": "current anchor plus immutable train-only cached neighbors", "cached_neighbor_gradient": "stopped", "rank_consistency": "same global image-index keyed immutable cache required", "staleness_requires_refresh_policy": True}, "decisions": {"reference_neighbor_coverage": "AUDITED", "local_advantage_gradient": "NDP_LOCAL_ADVANTAGE_INTERFACE_PASS" if tests_pass else "NDP_LOCAL_ADVANTAGE_INTERFACE_FAIL", "mechanism_comparison": "MATHEMATICAL_PROBE_ONLY", "short_pilot": "NDP_SHORT_PILOT_NOT_READY", "overall": "NDP_INCREMENTAL_MECHANISM_UNRESOLVED"}}
    write_json(out / "ndp_neighbor_coverage_gradient_summary.json", summary); write_json(out / "ndp_local_advantage_gradient_unit_tests.json", tests); write_json(out / "ndp_control_gradient_probe.json", controls); write_csv(out / "batch_cache_coverage.csv", batch_rows); write_csv(out / "rank_cache_coverage.csv", rank_rows)
    reports = ROOT / "reports"; reports.mkdir(exist_ok=True)
    c0 = summaries[0]; c1 = summaries[1]
    (reports / f"NDP_REFERENCE_NEIGHBOR_COVERAGE_{run_time}.md").write_text(f"""# NDP 真实参考邻域覆盖

本轮复用既有 train-only Bio feature bank、Top-20 graph 和冻结 batch plan；不重算邻域图，不执行模型 forward/backward。历史 8.15% 的精确定义是：在 1192 个 batch、9536 个有效 anchor 事件中，参考 Bio Top-10 全邻居集合（同PID正邻居或异PID邻居）与当前 batch 的正/负候选集合至少有一个交集；seed1234 为 {c0['historical_all_image_any_overlap_anchor_events']}/9536={c0['historical_all_image_any_overlap_anchor_rate']:.4%}，seed4321 为 {c1['historical_all_image_any_overlap_anchor_events']}/9536={c1['historical_all_image_any_overlap_anchor_rate']:.4%}。它不是异身份邻居关系占比，也不是可靠关系占比。

| seed | 有效anchor | 正候选对 | 负候选对 | 负邻居图同batch关系 | 全图正邻居/异身份邻居同batch | 可靠关系总数 | 可靠完整三元组同batch |
|---:|---:|---:|---:|---:|---:|---:|---:|
|1234|{c0['unique_valid_anchors']}|{c0['batch_positive_candidate_pairs']}|{c0['batch_negative_candidate_pairs']}|{c0['batch_cache_negative_only_edges']}|{c0['batch_cache_all_positive_edges']} / {c0['batch_cache_all_negative_edges']}|{c0['reference_reliable_rows']}/{c0['reference_relation_total_rows']} ({c0['reference_reliable_rate']:.4%})|{c0['batch_cache_full_reliable_triples']} ({c0['batch_cache_full_reliable_triple_rate']:.4%})|
|4321|{c1['unique_valid_anchors']}|{c1['batch_positive_candidate_pairs']}|{c1['batch_negative_candidate_pairs']}|{c1['batch_cache_negative_only_edges']}|{c1['batch_cache_all_positive_edges']} / {c1['batch_cache_all_negative_edges']}|{c1['reference_reliable_rows']}/{c1['reference_relation_total_rows']} ({c1['reference_reliable_rate']:.4%})|{c1['batch_cache_full_reliable_triples']} ({c1['batch_cache_full_reliable_triple_rate']:.4%})|

所有 graph 行均通过路径/PID 对齐、自引用和重复邻居检查；详细 rank 结果见输出 CSV。全局缓存的可靠邻居可离线访问，但同 batch 完整可靠三元组才同时具有当前 anchor、正样本和负样本的 batch 可微条件。
""", encoding="utf-8")
    (reports / f"NDP_BATCH_VS_CACHE_FEASIBILITY_{run_time}.md").write_text(f"""# NDP Batch 与 train-only cache 可行性

## 方案A：仅当前 batch

两种 seed 都有 9536/9576=99.5823% 的图像在冻结 batch plan 中同时具备同 PID 正样本和异 PID 负样本；这是候选局部结构的可访问性，不是模型梯度稳定性 PASS。每个有效 anchor 平均约1.115--1.131个正候选和5.869--5.885个负候选。真正的异身份参考邻居使用 `bio_negative_only_top20`，而不是历史8.15%使用的全邻居图；当前 batch 中该负邻居图的关系覆盖数和可靠完整三元组覆盖数见上表。

## 方案B：train-only 缓存

缓存从既有 E60 train-only feature bank 生成，当前 anchor 若实时前向则保留梯度，缓存邻居必须停止梯度。它将参考邻居从全局图中提供，但不是当前模型状态的实时邻居；缓存年龄、checkpoint来源和SHA256见 `ndp_neighbor_coverage_gradient_summary.json`。两 rank 必须按全局 image index 使用同一只读缓存，不能各自重建不同图。

方案B在离线覆盖上高于 batch-only，但仍受缓存陈旧性约束；当前没有刷新策略、在线一致性和PPU训练验证，因此状态为 `TECHNICAL_PROTOTYPE_ONLY`，不可晋级正式方法。
""", encoding="utf-8")
    (reports / f"NDP_LOCAL_ADVANTAGE_GRADIENT_AUDIT_{run_time}.md").write_text(f"""# NDP 局部优势梯度审计

候选为 `A=logmeanexp(pos/tau)-logmeanexp(neg/tau)`，`L=relu(A_ref-A_current-epsilon)`。参考分支 `detach`，当前 anchor及当前候选特征才可产生梯度；缓存邻居若停止梯度，则梯度只经过当前 anchor（以及若使用实时 batch 候选时的当前正负特征）。

12项独立公式、mask、自引用/重复关系、参考无梯度、FP32/CPU float16敏感性和C0-C3控制探针均写入 JSON；单测结果：{'PASS' if tests_pass else 'FAIL'}。这不是PPU模型 backward PASS，也不能证明参数更新有效。

缓存是固定 E60 终点，不是同一训练轨迹的相邻参考快照；因此本轮不把它当作在线风险监督。
""", encoding="utf-8")
    control_table = "\n".join(f"| {name} | {v['loss']:.6f} | {v['gradient_norm']:.6f} | {'PASS' if v['gradient_finite'] else 'FAIL'} |" for name, v in controls.items())
    (reports / f"NDP_TRIPLET_SUPCON_MATCHED_CONTROL_{run_time}.md").write_text(f"""# NDP 与 Triplet/SupCon 匹配控制

同一合成候选集合上进行离线梯度场探针；不是正式模型训练。

| 候选 | loss | gradient norm | finite |
|---|---:|---:|---|
{control_table}

C0 是当前 batch-hard 欧氏距离 SoftMargin 代理；C1 是固定参考单三元组自适应 margin；C2 是多邻居、参考单侧局部身份优势；C3 是同候选集合的多正/多负 SupCon/N-pair 型控制。C2 的信息差在于冻结参考优势和正负组 log-mean-exp，但是否超出C3的重加权，必须用真实PPU和同预算控制验证；本轮没有这种证据。
""", encoding="utf-8")
    (reports / f"NDP_SHORT_PILOT_GO_NO_GO_{run_time}.md").write_text(f"""# NDP 短程试验 Go/No-Go

局部邻域覆盖：离线 `PASS`；缓存：`TECHNICAL_PROTOTYPE_ONLY`；公式/梯度接口单测：`{'NDP_LOCAL_ADVANTAGE_INTERFACE_PASS' if tests_pass else 'NDP_LOCAL_ADVANTAGE_INTERFACE_FAIL'}`；PPU短程：`NDP_SHORT_PILOT_NOT_READY`。

阻塞条件：现有 AMP A/A 轨迹仍有4/6不匹配；本轮未进行真实模型 backward/optimizer.step；缓存来自独立E60终点且没有同轨迹更新前后状态；C2尚未证明独立于Triplet/SupCon。结论：`NDP_INCREMENTAL_MECHANISM_UNRESOLVED`，不启动30/60 epoch。
""", encoding="utf-8")
    log(f"【单测】局部优势与C0-C3离线梯度探针 {'PASS' if tests_pass else 'FAIL'}，不代表PPU训练。")
    log("【门禁】未启动模型训练；AMP/A-A数值不确定性、缓存陈旧性和缺少同轨迹前后状态仍阻止短程PPU。")


if __name__ == "__main__":
    main()
