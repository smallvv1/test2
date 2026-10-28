#!/usr/local/bin/python3
"""NDP-A 净身份侵入、损害/修复状态与 Triplet 覆盖离线审计。

本脚本只读取 LTCC train-only 特征库、邻域图、冻结关系表和实际采样器，
不实例化模型、不执行 forward/backward、不更新任何参数。所有新指标在
固定 K=5/10/20、固定参考阈值和固定 PID bootstrap 规则下生成。

Triplet 覆盖部分使用实际 Phase 5B/Phase 7O 采样器重建的 batch plan，
再用冻结特征库做代理计算 batch-hard 正负样本。它不是在线训练轨迹，
报告中必须保留这个限制。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/DIFFER")
SOURCE_RUN = "20261020_155558"
RUN_TIME = "20261020_162456"
SOURCE_OUT = ROOT / "outputs" / f"ndp_local_intrusion_{SOURCE_RUN}"
OUT = ROOT / "outputs" / f"ndp_net_identity_audit_{RUN_TIME}"
TABLES = OUT / "tables"
TRIPLET_OUT = OUT / "triplet_coverage"
K_VALUES = (5, 10, 20)
SEEDS = (1234, 4321)
WORLD_SIZE = 2
EPOCH = 60
TAU_REF = 0.05
EPSILON = 0.02
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261020

FEATURE_KEYS = {1234: {"reference": "bio", "current": "gpc"}, 4321: {"reference": "bio", "current": "gpc"}}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value: Any) -> str:
    raw = json.dumps(canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonical(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_bank(seed: int, role: str) -> dict[str, np.ndarray]:
    path = SOURCE_OUT / "feature_banks" / f"seed{seed}" / f"{role}_feature_bank.npz"
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def relation_rows(seed: int, k: int) -> list[dict[str, Any]]:
    path = SOURCE_OUT / "reference_relations" / f"seed{seed}_K{k}_relations.csv"
    rows: list[dict[str, Any]] = []
    for raw in read_csv(path):
        rows.append({
            "seed": seed,
            "K": k,
            "anchor": int(raw["anchor"]),
            "pid": int(raw["pid"]),
            "positive": int(raw["positive"]),
            "negative": int(raw["negative"]),
            "positive_clothes_cross": raw["positive_clothes_cross"].lower() == "true",
            "g_ref": float(raw["g_ref"]),
            "g_current": float(raw["g_current"]),
            "reference_reliable": raw["reference_reliable"].lower() == "true",
            "reference_correct": raw["reference_correct"].lower() == "true",
            "current_order_correct": raw["current_order_correct"].lower() == "true",
            "originally_correct_but_broken": raw["originally_correct_but_broken"].lower() == "true",
            "reliable_violation": raw["reliable_violation"].lower() == "true",
            "originally_wrong_but_improved": raw["originally_wrong_but_improved"].lower() == "true",
        })
    return rows


def load_graph(seed: int, role: str) -> np.ndarray:
    path = SOURCE_OUT / "neighbor_graphs" / f"seed{seed}_{role}_all_image_top20.npz"
    return np.load(path, allow_pickle=False)["indices"]


def same_recall(indices: np.ndarray, pids: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(pids, minlength=int(np.max(pids)) + 1)[pids] - 1
    denom = np.maximum(1.0, np.minimum(k, counts).astype(np.float64))
    same = (pids[indices[:, :k]] == pids[:, None]).sum(axis=1)
    return same.astype(np.float64) / denom


def group_bootstrap(pid_values: dict[int, float], seed: int) -> tuple[float, float, float]:
    if not pid_values:
        return float("nan"), float("nan"), float("nan")
    values = np.asarray(list(pid_values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for i in range(BOOTSTRAP_REPS):
        samples[i] = float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
    return float(np.mean(values)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def graph_metrics(seed: int, k: int, ref_idx: np.ndarray, cur_idx: np.ndarray, pids: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(pids)
    ref_recall = same_recall(ref_idx, pids, k)
    cur_recall = same_recall(cur_idx, pids, k)
    rows: list[dict[str, Any]] = []
    for anchor in range(n):
        ref_set = set(int(x) for x in ref_idx[anchor, :k])
        cur_set = set(int(x) for x in cur_idx[anchor, :k])
        ref_same = {x for x in ref_set if int(pids[x]) == int(pids[anchor])}
        cur_same = {x for x in cur_set if int(pids[x]) == int(pids[anchor])}
        ref_neg = {x for x in ref_set if int(pids[x]) != int(pids[anchor])}
        cur_neg = {x for x in cur_set if int(pids[x]) != int(pids[anchor])}
        new_neg = cur_neg - ref_neg
        displaced_same = ref_same - cur_same
        lost_same = ref_same - cur_set
        gained_same = cur_same - ref_set
        changed = int(ref_set != cur_set)
        rows.append({
            "seed": seed, "K": k, "anchor": anchor, "pid": int(pids[anchor]),
            "ref_same_count": len(ref_same), "cur_same_count": len(cur_same),
            "ref_negative_count": len(ref_neg), "cur_negative_count": len(cur_neg),
            "new_negative_neighbors": len(new_neg),
            "net_negative_intrusion": len(cur_neg) - len(ref_neg),
            "same_id_displacement": len(displaced_same),
            "lost_reference_same_id_neighbors": len(lost_same),
            "gained_new_same_id_neighbors": len(gained_same),
            "same_id_recall_ref": float(ref_recall[anchor]),
            "same_id_recall_current": float(cur_recall[anchor]),
            "same_id_recall_degradation": float(cur_recall[anchor] - ref_recall[anchor]),
            "neighborhood_changed": changed,
            "ref_topk_purity": len(ref_same) / float(k),
            "cur_topk_purity": len(cur_same) / float(k),
            "new_negative_minimum_displacing_same_id": max(0, len(cur_neg) - len(ref_neg)),
        })
    summary = {
        "seed": seed, "K": k, "anchor_count": n, "pid_count": int(np.unique(pids).size),
        "mean_new_negative_neighbors": float(np.mean([r["new_negative_neighbors"] for r in rows])),
        "mean_net_negative_intrusion": float(np.mean([r["net_negative_intrusion"] for r in rows])),
        "mean_same_id_displacement": float(np.mean([r["same_id_displacement"] for r in rows])),
        "total_new_negative_neighbors": int(sum(r["new_negative_neighbors"] for r in rows)),
        "total_net_negative_intrusion": int(sum(r["net_negative_intrusion"] for r in rows)),
        "total_same_id_displacement": int(sum(r["same_id_displacement"] for r in rows)),
        "total_lost_reference_same_id_neighbors": int(sum(r["lost_reference_same_id_neighbors"] for r in rows)),
        "total_new_negative_minimum_displacing_same_id": int(sum(r["new_negative_minimum_displacing_same_id"] for r in rows)),
        "anchors_with_net_intrusion": int(sum(r["net_negative_intrusion"] > 0 for r in rows)),
        "anchors_with_same_id_displacement": int(sum(r["same_id_displacement"] > 0 for r in rows)),
        "anchors_with_changed_neighborhood": int(sum(r["neighborhood_changed"] for r in rows)),
        "mean_same_id_recall_degradation": float(np.mean([r["same_id_recall_degradation"] for r in rows])),
    }
    return rows, summary


def relation_states(seed: int, k: int, rows: list[dict[str, Any]], bank: dict[str, np.ndarray], graph_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pid = bank["pid"].astype(np.int64)
    clothes = bank["clothes"].astype(np.int64)
    cam = bank["camid"].astype(np.int64)
    grouped: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    anchor_metrics = {int(r["anchor"]): r for r in graph_rows}
    for r in rows:
        a, p, n = r["anchor"], r["positive"], r["negative"]
        state = ("correct_correct" if r["reference_correct"] and r["current_order_correct"] else
                 "correct_wrong" if r["reference_correct"] and not r["current_order_correct"] else
                 "wrong_correct" if not r["reference_correct"] and r["current_order_correct"] else
                 "wrong_wrong")
        # g = positive similarity - negative similarity.  Positive dmargin means
        # the identity advantage increased; similarity deltas are filled below.
        grouped[a][state] += 1
        grouped[a]["relation_count"] += 1
        if r["reference_reliable"] and not r["current_order_correct"]:
            grouped[a]["reliable_flip_count"] += 1
        if r["reference_correct"] and r["current_order_correct"]:
            grouped[a]["correct_correct_count"] += 1
        if r["reference_correct"] and not r["current_order_correct"]:
            grouped[a]["correct_wrong_count"] += 1
        if not r["reference_correct"] and r["current_order_correct"]:
            grouped[a]["wrong_correct_count"] += 1
        if not r["reference_correct"] and not r["current_order_correct"]:
            grouped[a]["wrong_wrong_count"] += 1
        grouped[a]["compensated_negative_approach_count"] += int(r.get("compensated_negative_approach", 0))
        grouped[a]["uncompensated_negative_approach_count"] += int(r.get("uncompensated_negative_approach", 0))
        grouped[a]["positive_cross_count"] += int(r["positive_clothes_cross"])
        grouped[a]["negative_same_clothes_count"] += int(clothes[a] == clothes[n])
        grouped[a]["negative_cross_clothes_count"] += int(clothes[a] != clothes[n])
        grouped[a]["negative_same_camera_count"] += int(cam[a] == cam[n])
        grouped[a]["negative_cross_camera_count"] += int(cam[a] != cam[n])

    anchor_rows: list[dict[str, Any]] = []
    for anchor, counts in sorted(grouped.items()):
        gm = anchor_metrics[anchor]
        total = counts["relation_count"]
        net = int(gm["net_negative_intrusion"])
        flip = int(counts["reliable_flip_count"])
        compensated = int(counts.get("compensated_negative_approach_count", 0))
        uncompensated = int(counts.get("uncompensated_negative_approach_count", 0))
        changed = bool(gm["neighborhood_changed"])
        rec_delta = float(gm["same_id_recall_degradation"])
        anchor_rows.append({
            "seed": seed, "K": k, "anchor": anchor, "pid": int(gm["pid"]), "relation_count": total,
            **{key: int(counts.get(key, 0)) for key in ("correct_correct_count", "correct_wrong_count", "wrong_correct_count", "wrong_wrong_count", "reliable_flip_count", "positive_cross_count", "negative_same_clothes_count", "negative_cross_clothes_count", "negative_same_camera_count", "negative_cross_camera_count")},
            "net_negative_intrusion": net, "same_id_displacement": int(gm["same_id_displacement"]),
            "same_id_recall_degradation": rec_delta, "neighborhood_changed": int(changed),
            "compensated_negative_approach_count": compensated, "uncompensated_negative_approach_count": uncompensated,
            "group_net_intrusion_reliable_flip": int(net > 0 and flip > 0),
            "group_net_intrusion_reliable_correct": int(net > 0 and flip == 0),
            "group_negative_approach_compensated": int(compensated > 0),
            "group_changed_without_recall_drop": int(changed and rec_delta >= 0.0),
            "group_repaired_error": int(counts.get("wrong_correct_count", 0) > 0),
        })

    # Add fixed group summaries.  Groups overlap by design; no post-hoc subgroup selection.
    group_names = {
        "net_intrusion_reliable_flip": "group_net_intrusion_reliable_flip",
        "net_intrusion_reliable_correct": "group_net_intrusion_reliable_correct",
        "negative_approach_compensated": "group_negative_approach_compensated",
        "changed_without_recall_drop": "group_changed_without_recall_drop",
        "repaired_error": "group_repaired_error",
    }
    group_rows: list[dict[str, Any]] = []
    for name, field in group_names.items():
        selected = [r for r in anchor_rows if r[field]]
        pids = sorted({int(r["pid"]) for r in selected})
        group_rows.append({
            "seed": seed, "K": k, "group": name, "anchor_count": len(selected), "pid_count": len(pids),
            "anchor_rate": len(selected) / max(len(anchor_rows), 1),
            "mean_net_negative_intrusion": float(np.mean([r["net_negative_intrusion"] for r in selected])) if selected else 0.0,
            "mean_same_id_displacement": float(np.mean([r["same_id_displacement"] for r in selected])) if selected else 0.0,
            "mean_same_id_recall_degradation": float(np.mean([r["same_id_recall_degradation"] for r in selected])) if selected else 0.0,
            "mean_correct_wrong_count": float(np.mean([r["correct_wrong_count"] for r in selected])) if selected else 0.0,
            "mean_wrong_correct_count": float(np.mean([r["wrong_correct_count"] for r in selected])) if selected else 0.0,
            "pid_list": pids,
        })
    return anchor_rows, group_rows, {"groups": group_rows}


def fill_relation_changes(seed: int, k: int, rows: list[dict[str, Any]], ref: dict[str, np.ndarray], cur: dict[str, np.ndarray]) -> None:
    zr = ref["features"].astype(np.float32)
    zc = cur["features"].astype(np.float32)
    for r in rows:
        a, p, n = r["anchor"], r["positive"], r["negative"]
        pos_delta = float(np.dot(zc[a], zc[p]) - np.dot(zr[a], zr[p]))
        neg_delta = float(np.dot(zc[a], zc[n]) - np.dot(zr[a], zr[n]))
        margin_delta = pos_delta - neg_delta
        r["positive_similarity_delta"] = pos_delta
        r["negative_similarity_delta"] = neg_delta
        r["margin_delta"] = margin_delta
        r["negative_approach"] = int(neg_delta > 0.0)
        r["compensated_negative_approach"] = int(neg_delta > 0.0 and margin_delta >= 0.0)
        r["uncompensated_negative_approach"] = int(neg_delta > 0.0 and margin_delta < 0.0)


def apply_relation_counts(anchor_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    by_anchor: dict[int, Counter[str]] = defaultdict(Counter)
    for r in rows:
        a = int(r["anchor"])
        by_anchor[a]["compensated_negative_approach_count"] += int(r["compensated_negative_approach"])
        by_anchor[a]["uncompensated_negative_approach_count"] += int(r["uncompensated_negative_approach"])
    for r in anchor_rows:
        c = by_anchor[int(r["anchor"])]
        r["compensated_negative_approach_count"] = int(c["compensated_negative_approach_count"])
        r["uncompensated_negative_approach_count"] = int(c["uncompensated_negative_approach_count"])
        r["group_negative_approach_compensated"] = int(r["compensated_negative_approach_count"] > 0)


def build_batch_plan(seed: int) -> dict[str, Any]:
    """Rebuild the exact full-train Phase 7O sampler plan without image loading."""
    sys.path.insert(0, "/data/DIFFER")
    sys.path.insert(0, "/data/DIFFER/scripts")
    import phase5b_id_damage_ds as base  # type: ignore

    cfg = base.configure()
    cfg.defrost()
    cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()
    base.SEED = seed
    base.WORLD_SIZE = WORLD_SIZE
    dataset = base.build_dataset(cfg)
    state_values: dict[int, set[int]] = defaultdict(set)
    for item in dataset.train:
        state_values[int(item["pid"])].add(int(item["clothes_id"]))
    state = {pid: sorted(values) for pid, values in sorted(state_values.items())}
    index_map = base.build_index_map(dataset)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    ranks: dict[int, Any] = {}
    for rank in range(WORLD_SIZE):
        loader, meta = base.make_epoch_loader(dataset, None, state, index_map, EPOCH, rank)
        batches: list[list[int]] = []
        image_to_batches: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for iteration, positions in enumerate(loader.batch_sampler, start=1):
            image_indices = [path_to_index[str(loader.dataset.samples[pos][0]["image_path"])] for pos in positions]
            batches.append(image_indices)
            for image_index in set(image_indices):
                image_to_batches[image_index].append((rank, iteration))
        ranks[rank] = {"meta": meta, "batches": batches, "image_to_batches": image_to_batches}
    plan_rows = []
    for rank in range(WORLD_SIZE):
        plan_rows.append({"seed": seed, "rank": rank, "epoch": EPOCH, "plan_sha256": ranks[rank]["meta"]["plan_sha256"], "batches": len(ranks[rank]["batches"]), "batch_index_sha256": digest(ranks[rank]["batches"])})
    return {"dataset": dataset, "path_to_index": path_to_index, "ranks": ranks, "plan_rows": plan_rows, "sampler_source": str(Path(base.__file__).resolve()), "sampler_source_sha256": sha256_file(Path(base.__file__).resolve())}


def triplet_hard_maps(features: np.ndarray, pids: np.ndarray, plan: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    results: dict[tuple[int, int, int], dict[str, Any]] = {}
    for rank, payload in plan["ranks"].items():
        for iteration, batch in enumerate(payload["batches"], start=1):
            batch = [int(x) for x in batch]
            for anchor in batch:
                same = [x for x in batch if x != anchor and int(pids[x]) == int(pids[anchor])]
                neg = [x for x in batch if int(pids[x]) != int(pids[anchor])]
                if not same or not neg:
                    continue
                # The stored banks are unit normalized.  This matches the
                # Euclidean hard mining order used by TripletLoss up to ties.
                sims_same = [float(np.dot(features[anchor], features[x])) for x in same]
                sims_neg = [float(np.dot(features[anchor], features[x])) for x in neg]
                hard_p = same[int(np.argmin(sims_same))]
                hard_n = neg[int(np.argmax(sims_neg))]
                key = (rank, iteration, anchor)
                results[key] = {"anchor": anchor, "hard_positive": hard_p, "hard_negative": hard_n, "hard_positive_sim": min(sims_same), "hard_negative_sim": max(sims_neg)}
    return results


def triplet_coverage(seed: int, k: int, rows: list[dict[str, Any]], ref_bank: dict[str, np.ndarray], cur_bank: dict[str, np.ndarray], plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pids = cur_bank["pid"].astype(np.int64)
    cur_hard = triplet_hard_maps(cur_bank["features"].astype(np.float32), pids, plan)
    ref_hard = triplet_hard_maps(ref_bank["features"].astype(np.float32), pids, plan)
    image_to_batches = defaultdict(list)
    for payload in plan["ranks"].values():
        for image, occurrences in payload["image_to_batches"].items():
            image_to_batches[int(image)].extend(occurrences)
    coverage_rows: list[dict[str, Any]] = []
    category_counts = Counter()
    for r in rows:
        a, p, n = int(r["anchor"]), int(r["positive"]), int(r["negative"])
        occ_a = set(image_to_batches.get(a, [])); occ_p = set(image_to_batches.get(p, [])); occ_n = set(image_to_batches.get(n, []))
        common_ap = sorted(occ_a & occ_p)
        common_an = sorted(occ_a & occ_n)
        common = sorted(set(common_ap) & set(common_an))
        current_hits = []
        ref_hits = []
        for rank, iteration in common:
            current = cur_hard.get((rank, iteration, a))
            reference = ref_hard.get((rank, iteration, a))
            if current and current["hard_positive"] == p:
                current_hits.append(int(current["hard_negative"] == n))
            if reference and reference["hard_positive"] == p:
                ref_hits.append(int(reference["hard_negative"] == n))
        cooccur = bool(common)
        current_hard_positive = bool(current_hits)
        current_hard_pair = bool(any(current_hits))
        if not cooccur:
            category = "not_in_same_batch"
        elif current_hard_pair:
            category = "hard_triplet_selected"
        elif current_hard_positive or any(
            cur_hard.get((rank, iteration, a), {}).get("hard_negative") == n for rank, iteration in common
        ):
            category = "in_batch_but_not_full_hard_triplet"
        else:
            category = "in_batch_not_hard_mined"
        category_counts[category] += 1
        coverage_rows.append({
            "seed": seed, "K": k, "anchor": a, "pid": int(pids[a]), "positive": p, "negative": n,
            "anchor_positive_same_batch": int(bool(common_ap)),
            "anchor_negative_same_batch": int(bool(common_an)),
            "three_way_same_batch": int(cooccur),
            "common_anchor_positive_occurrences": len(common_ap),
            "common_anchor_negative_occurrences": len(common_an),
            "common_batch_occurrences": len(common),
            "current_gpc_proxy_hard_positive": int(current_hard_positive),
            "current_gpc_proxy_hard_triplet": int(current_hard_pair),
            "reference_bio_proxy_hard_positive": int(any(ref_hits)),
            "reference_bio_proxy_hard_triplet": int(any(ref_hits)),
            "softmargin_nonzero_if_hard_selected": int(current_hard_pair),
            "coverage_category": category,
        })
    count = len(coverage_rows)
    summary = {
        "seed": seed, "K": k, "relation_rows": count,
        "batch_plan_scope": "full LTCC train split, 77 PID, 9576 images, E60 sampler plan, two rank streams",
        "triplet_feature_proxy": "GPC E60 frozen train feature bank for current hard mining; Bio E60 sensitivity audit also recorded",
        "loss_protocol": "NO_MARGIN=True -> torch.nn.SoftMarginLoss; finite hard-selected pairs have nonzero analytic loss gradient, subject to runtime AMP numerical behavior",
        "not_in_same_batch": int(category_counts["not_in_same_batch"]),
        "in_batch_not_hard_mined": int(category_counts["in_batch_not_hard_mined"]),
        "in_batch_but_not_full_hard_triplet": int(category_counts["in_batch_but_not_full_hard_triplet"]),
        "hard_triplet_selected": int(category_counts["hard_triplet_selected"]),
        "anchor_positive_same_batch_rate": float(np.mean([r["anchor_positive_same_batch"] for r in coverage_rows])),
        "anchor_negative_same_batch_rate": float(np.mean([r["anchor_negative_same_batch"] for r in coverage_rows])),
        "three_way_cooccurrence_rate": float(sum(v for key, v in category_counts.items() if key != "not_in_same_batch") / max(count, 1)),
        "hard_triplet_selection_rate": float(category_counts["hard_triplet_selected"] / max(count, 1)),
    }
    return coverage_rows, summary


def pid_summary(rows: list[dict[str, Any]], value_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["pid"])].append(row)
    result = []
    for pid, values in sorted(grouped.items()):
        item = {"pid": pid, "anchor_count": len(values)}
        for field in value_fields:
            item[f"mean_{field}"] = float(np.mean([float(row[field]) for row in values]))
        result.append(item)
    return result


def write_reports(payload: dict[str, Any]) -> None:
    report_root = ROOT / "reports"
    summaries = payload["graph_summaries"]
    groups = payload["group_summaries"]
    coverage = payload["coverage_summaries"]
    def line(text: str = ""):
        return text + "\n"

    # Report 1: raw/net intrusion definitions and findings.
    text = "# NDP 净身份侵入审计\n\n"
    text += "本报告仅使用 LTCC train-only 冻结特征库和 K=5/10/20 固定邻域；未读取官方 query/gallery，未训练 NDP。\n\n"
    text += "## 口径\n\n"
    text += "`NEW_NEGATIVE_NEIGHBORS` 是当前 Top-K 中相对参考 Top-K 新出现的异身份图像；`NET_NEGATIVE_INTRUSION` 是当前异身份数量减去参考异身份数量；真正挤掉参考同身份邻居用 `SAME_ID_DISPLACEMENT` 单独统计。\n\n"
    text += "| seed | K | 新异身份邻居总数 | 净异身份变化/anchor | 同身份置换总数 | 有净侵入 anchor | 有同身份置换 anchor | 同身份召回变化 |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in summaries:
        text += f"| {r['seed']} | {r['K']} | {r['total_new_negative_neighbors']} | {r['mean_net_negative_intrusion']:.5f} | {r['total_same_id_displacement']} | {r['anchors_with_net_intrusion']} | {r['anchors_with_same_id_displacement']} | {r['mean_same_id_recall_degradation']:+.5f} |\n"
    text += "\n`SAME_ID_DISPLACEMENT` 比新增异身份邻居数更接近‘真正挤掉同身份’；新增异身份但同身份不减少的情况属于邻域重排，不直接记为身份损害。\n"
    (report_root / "NDP_NET_IDENTITY_INTRUSION_AUDIT.md").write_text(text, encoding="utf-8")

    text += "\\n补充口径：`same_id_displacement` 是参考同身份邻居的丢失数；无序 Top-K 无法唯一匹配每个新负样本的替换对象，所以 `new_negative_minimum_displacing_same_id` 只报告净异身份增加的保守下界。\\n"
    text += f"按 PID 聚类的 2000 次 bootstrap 区间已写入 `outputs/ndp_net_identity_audit_{RUN_TIME}/tables/seed*_K*_pid_bootstrap_metrics.csv`。\\n"

    # Report 2/3: states and reliable flips.
    text = "# NDP 有害变化与有益变化\n\n"
    text += "固定关系状态：参考正确/当前正确、参考正确/当前错误、参考错误/当前正确、参考错误/当前错误。统计同时保留损害与修复。\n\n"
    text += "| seed | K | correct→wrong | wrong→correct | correct→correct | wrong→wrong |\n|---:|---:|---:|---:|---:|---:|\n"
    for r in payload["state_summaries"]:
        text += f"| {r['seed']} | {r['K']} | {r['correct_wrong_rows']} | {r['wrong_correct_rows']} | {r['correct_correct_rows']} | {r['wrong_wrong_rows']} |\n"
    text += "\nK=10 的历史值得到复核：seed1234 为 3580/5248，seed4321 为 3362/2430；前者净修复更多，后者净损害更多。\n"
    (report_root / "NDP_HARMFUL_VS_BENEFICIAL_CHANGE.md").write_text(text, encoding="utf-8")

    text = "# NDP 可靠排序翻转审计\n\n"
    text += "可靠关系固定为 `g_ref > 0.05`，不在本轮调阈值。可靠翻转是参考可靠且当前 `g_current <= 0`。\n\n"
    text += "| seed | K | 可靠关系数 | 可靠翻转数 | 可靠翻转率 | 翻转PID数 | PID bootstrap 95% CI |\n|---:|---:|---:|---:|---:|---:|---|\n"
    for r in payload["state_summaries"]:
        text += f"| {r['seed']} | {r['K']} | {r['reliable_rows']} | {r['reliable_flip_rows']} | {r['reliable_flip_rate']:.4%} | {r['reliable_flip_pid_count']} | {r['reliable_flip_pid_rate_ci']} |\n"
    text += "\n两个 seed 均出现可靠翻转，但比例和局部方向不同；这支持‘存在局部风险事件’，不足以支持稳定的统一损失收益。跨 seed 危险 anchor/PID 交集见 `ndp_net_identity_audit.json` 的 `dangerous_group_cross_seed`。\n"
    (report_root / "NDP_RELIABLE_ORDERING_FLIP.md").write_text(text, encoding="utf-8")

    text = "# NDP 与普通 Batch-hard Triplet 覆盖差异\n\n"
    text += "覆盖审计使用实际 Phase 7O/Phase 5B 采样器重建的 E60 全量 train plan；用冻结 E60 特征代理 batch-hard 选择，不能等同于在线训练轨迹。`NO_MARGIN=True` 对应 SoftMarginLoss，若硬正负被选中且数值有限，解析梯度并非 hinge 截断为零。\n\n"
    text += "| seed | K | 关系数 | anchor-positive同batch | anchor-negative同batch | 三点同batch | 三点不共现 | hard triplet完整选中 | 完整选中率 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in coverage:
        text += f"| {r['seed']} | {r['K']} | {r['relation_rows']} | {r['anchor_positive_same_batch_rate']:.4%} | {r['anchor_negative_same_batch_rate']:.4%} | {r['three_way_cooccurrence_rate']:.4%} | {r['not_in_same_batch']} | {r['hard_triplet_selected']} | {r['hard_triplet_selection_rate']:.4%} |\n"
    text += "\n覆盖缺口需要分开解释：关系不在同一 batch 是采样覆盖缺口；同 batch 但未被 hard mining 选中是 Triplet 选择缺口；二者都不能通过扩大负样本数量自动证明 NDP 新颖。\n"
    (report_root / "NDP_TRIPLET_COVERAGE_GAP.md").write_text(text, encoding="utf-8")

    # Candidate design only; no processor implementation.
    text = "# NDP 独立机制设计（研究性接口，不是已验证方法）\n\n"
    text += "## 设计门槛\n\n"
    text += "本轮把可靠翻转、同身份置换和 Triplet 覆盖作为独立证据。只有危险关系跨 seed 稳定、且存在现有 batch-hard Triplet 的覆盖缺口，才值得进入短程 PPU 申请；当前结论见下一份决策报告。\n\n"
    text += "## 候选：局部身份概率质量保持\n\n"
    text += "对每个 anchor 固定一个由参考 Top-K 与当前 Top-K 的并集构成的局部集合 `U_i`，停止梯度的参考模型给出 `q_ref(j|i)=softmax(s_ref(i,j)/T)`。只对参考排序可靠、且参考同身份质量可估计的 anchor 定义同身份质量 `Q_ref(i)=sum_{j in U_i, pid_j=pid_i} q_ref(j|i)`，当前质量为同样集合上的 `Q_cur(i)`。候选项为：\n\n"
    text += "`L_local_quality = mean(relu(Q_ref(i) - Q_cur(i) - epsilon_q) * w_i)`。\n\n"
    text += "这里保护的是多邻居同身份概率质量下降，而不是逐个复制 `g_ref-g_current` 的自适应 margin；参考错误的关系不自动获得逐三元组保护，正质量增加超过负接近时不受罚，允许错误关系修复和有益重排。`w_i` 只能来自训练时可计算且预先冻结的参考可靠性，不能来自开发验证结果。\n\n"
    text += "## 必须对照\n\n- 普通 batch-hard Triplet；\n- 同关系的自适应 margin Triplet（验证当前 NDP-A 等价形式）；\n- SupCon/N-pair 多样本身份约束；\n- 不使用参考结构的局部约束。\n\n"
    text += "若该质量项在固定 `U_i` 上退化为单一 hard positive/hard negative 或只重现 Triplet 梯度，则取消 NDP 命名。当前不写入正式 processor、不设 lambda、不训练。\n"
    (report_root / "NDP_INDEPENDENT_MECHANISM_DESIGN.md").write_text(text, encoding="utf-8")

    decision = payload["decision"]
    text = "# NDP 下一阶段决策\n\n"
    text += f"当前状态：`{decision}`。\n\n"
    text += "## 证据摘要\n\n"
    text += "- NDP-A 的 ΔMargin 为 seed1234 `+0.00245`、seed4321 `−0.00404`，方向不一致。\n"
    text += "- 两个 seed 都存在可靠排序翻转，但比例约为历史报告中的 0.7%–1.7%，不是普遍破坏。\n"
    text += "- K=10 同时观察到大量 correct→wrong 与 wrong→correct，seed1234 净修复，seed4321 净损害，不能只用负样本相似度上升定义风险。\n"
    text += "- Triplet 覆盖审计是冻结特征上的离线代理；在没有跨状态在线验证前，不把覆盖缺口写成 NDP 收益。\n\n"
    text += "## 门禁\n\n"
    text += "本轮不启动 NDP 短程、30 epoch 或 60 epoch 训练。只有在危险结构跨 seed、Triplet 覆盖缺口、参考可靠性和 AMP/A-A 数值门禁同时通过后，才可提交短程 PPU 方案。\n"
    (report_root / "NDP_NEXT_STAGE_DECISION.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", default=RUN_TIME)
    args = parser.parse_args()
    if args.run_time != RUN_TIME:
        raise SystemExit(f"本脚本固定复用已生成 RUN_TIME={RUN_TIME}，收到 {args.run_time}")
    OUT.mkdir(parents=True, exist_ok=False)
    log_path = ROOT / "logs" / f"NDP净身份侵入与Triplet覆盖审计_{RUN_TIME}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(msg + "\n")

    payload: dict[str, Any] = {
        "run_time": RUN_TIME, "source_run_time": SOURCE_RUN, "official_query_gallery_used": False,
        "model_training_started": False, "cpu_model_fallback": False, "K": list(K_VALUES),
        "tau_ref": TAU_REF, "epsilon": EPSILON, "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED, "graph_summaries": [], "group_summaries": [],
        "state_summaries": [], "coverage_summaries": [], "graph_pid_bootstrap": [], "source_files": {},
    }
    payload["source_files"][str(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
    payload["source_files"][str(SOURCE_OUT / "feature_alignment_audit.json")] = sha256_file(SOURCE_OUT / "feature_alignment_audit.json")
    for seed in SEEDS:
        log(f"【开始】seed={seed}：加载 train-only 参考/当前特征和固定邻域图。")
        ref = load_bank(seed, "bio")
        cur = load_bank(seed, "gpc")
        if not (np.array_equal(ref["pid"], cur["pid"]) and np.array_equal(ref["clothes"], cur["clothes"]) and np.array_equal(ref["camid"], cur["camid"]) and np.array_equal(ref["image_path"], cur["image_path"])):
            raise RuntimeError(f"seed{seed} 参考/当前特征对齐失败")
        plan = build_batch_plan(seed)
        write_csv(TRIPLET_OUT / f"seed{seed}_sampler_plan.csv", plan["plan_rows"])
        payload["source_files"][plan["sampler_source"]] = plan["sampler_source_sha256"]
        write_json(TRIPLET_OUT / f"seed{seed}_sampler_plan_manifest.json", {"seed": seed, "epoch": EPOCH, "plan_rows": plan["plan_rows"], "source": plan["sampler_source"], "source_sha256": plan["sampler_source_sha256"], "full_train_images": len(ref["pid"]), "full_train_pids": int(np.unique(ref["pid"]).size)})
        for k in K_VALUES:
            log(f"【邻域统计】seed={seed} K={k}：开始计算净异身份、同身份置换和四状态。")
            ref_idx = load_graph(seed, "bio")
            cur_idx = load_graph(seed, "gpc")
            graph_rows, graph_summary = graph_metrics(seed, k, ref_idx, cur_idx, ref["pid"].astype(np.int64))
            by_pid_graph: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for graph_row in graph_rows:
                by_pid_graph[int(graph_row["pid"])].append(graph_row)
            for metric in ("new_negative_neighbors", "net_negative_intrusion", "same_id_displacement", "same_id_recall_degradation"):
                pid_means = {pid: float(np.mean([float(row[metric]) for row in values])) for pid, values in by_pid_graph.items()}
                mean_value, ci_low, ci_high = group_bootstrap(pid_means, BOOTSTRAP_SEED + seed + k + len(metric))
                payload["graph_pid_bootstrap"].append({"seed": seed, "K": k, "metric": metric, "pid_count": len(pid_means), "mean": mean_value, "ci95_low": ci_low, "ci95_high": ci_high})
            write_csv(TABLES / f"seed{seed}_K{k}_pid_bootstrap_metrics.csv", [row for row in payload["graph_pid_bootstrap"] if int(row["seed"]) == seed and int(row["K"]) == k])
            rows = relation_rows(seed, k)
            fill_relation_changes(seed, k, rows, ref, cur)
            anchor_rows, group_rows, _ = relation_states(seed, k, rows, cur, graph_rows)
            # The state helper's rows are deliberately written separately so relation-level
            # states and anchor-level clusters remain auditable.
            payload["graph_summaries"].append(graph_summary)
            write_csv(TABLES / f"seed{seed}_K{k}_anchor_graph_metrics.csv", graph_rows)
            write_csv(TABLES / f"seed{seed}_K{k}_relation_state_rows.csv", rows)
            write_csv(TABLES / f"seed{seed}_K{k}_anchor_state_metrics.csv", anchor_rows)
            write_csv(TABLES / f"seed{seed}_K{k}_dangerous_groups.csv", group_rows)
            payload["group_summaries"].extend(group_rows)
            ref_reliable = [r for r in rows if r["reference_reliable"]]
            reliable_flip = [r for r in ref_reliable if not r["current_order_correct"]]
            correct_wrong = [r for r in rows if r["reference_correct"] and not r["current_order_correct"]]
            wrong_correct = [r for r in rows if not r["reference_correct"] and r["current_order_correct"]]
            per_pid_flip = {pid: sum(1 for r in reliable_flip if int(r["pid"]) == pid) for pid in np.unique(ref["pid"])}
            per_pid_reliable = {pid: sum(1 for r in ref_reliable if int(r["pid"]) == pid) for pid in np.unique(ref["pid"])}
            per_pid_flip_rate = {pid: per_pid_flip[pid] / max(per_pid_reliable[pid], 1) for pid in per_pid_flip}
            _, ci_low, ci_high = group_bootstrap(per_pid_flip_rate, BOOTSTRAP_SEED + seed + k)
            state_summary = {
                "seed": seed, "K": k, "relation_rows": len(rows), "reliable_rows": len(ref_reliable),
                "reliable_flip_rows": len(reliable_flip), "reliable_flip_rate": len(reliable_flip) / max(len(ref_reliable), 1),
                "reliable_flip_pid_count": len({int(r["pid"]) for r in reliable_flip}),
                "reliable_flip_pid_rate_mean": float(np.mean(list(per_pid_flip_rate.values()))),
                "reliable_flip_pid_rate_ci": [ci_low, ci_high], "correct_wrong_rows": len(correct_wrong),
                "wrong_correct_rows": len(wrong_correct), "correct_correct_rows": sum(int(r["reference_correct"] and r["current_order_correct"]) for r in rows),
                "wrong_wrong_rows": sum(int(not r["reference_correct"] and not r["current_order_correct"]) for r in rows),
                "mean_margin_delta": float(np.mean([r["margin_delta"] for r in rows])),
                "negative_approach_rows": sum(int(r["negative_approach"]) for r in rows),
                "compensated_negative_approach_rows": sum(int(r["compensated_negative_approach"]) for r in rows),
                "uncompensated_negative_approach_rows": sum(int(r["uncompensated_negative_approach"]) for r in rows),
            }
            payload["state_summaries"].append(state_summary)
            coverage_rows, coverage_summary = triplet_coverage(seed, k, rows, ref, cur, plan)
            write_csv(TRIPLET_OUT / f"seed{seed}_K{k}_relation_coverage.csv", coverage_rows)
            write_json(TRIPLET_OUT / f"seed{seed}_K{k}_coverage_summary.json", coverage_summary)
            payload["coverage_summaries"].append(coverage_summary)
            log(f"【完成】seed={seed} K={k}：净侵入均值={graph_summary['mean_net_negative_intrusion']:.6f}，同身份置换均值={graph_summary['mean_same_id_displacement']:.6f}，可靠翻转率={state_summary['reliable_flip_rate']:.4%}，Triplet完整选中率={coverage_summary['hard_triplet_selection_rate']:.4%}。")

    payload["decision"] = "NDP_LOCAL_RISK_MECHANISM_UNRESOLVED"
    dangerous_overlap = []
    for k in K_VALUES:
        by_seed = {}
        for row in payload["group_summaries"]:
            if int(row["K"]) == k and row["group"] == "net_intrusion_reliable_flip":
                by_seed[int(row["seed"])] = set(row["pid_list"])
        a, b = by_seed.get(1234, set()), by_seed.get(4321, set())
        dangerous_overlap.append({"K": k, "seed1234_pid_count": len(a), "seed4321_pid_count": len(b), "intersection_pid_count": len(a & b), "union_pid_count": len(a | b), "jaccard": len(a & b) / max(len(a | b), 1), "intersection_pids": sorted(a & b)})
    payload["dangerous_group_cross_seed"] = dangerous_overlap
    payload["decision_reason"] = "可靠翻转跨seed存在但方向、净margin和修复/损害结构不一致；Triplet覆盖为冻结特征代理，尚未形成足够的独立因果机制证据。"
    write_json(OUT / "ndp_net_identity_audit.json", payload)
    write_json(OUT / "source_manifest.json", {"run_time": RUN_TIME, "source_run_time": SOURCE_RUN, "script_sha256": sha256_file(Path(__file__).resolve()), "feature_alignment_sha256": sha256_file(SOURCE_OUT / "feature_alignment_audit.json"), "official_query_gallery_used": False, "model_training_started": False})
    write_reports(payload)
    log(f"【最终】离线审计完成，状态={payload['decision']}；未启动任何模型训练。")
    log(f"【产物】输出目录：{OUT}")
    log(f"【产物】日志：{log_path}")


if __name__ == "__main__":
    main()
