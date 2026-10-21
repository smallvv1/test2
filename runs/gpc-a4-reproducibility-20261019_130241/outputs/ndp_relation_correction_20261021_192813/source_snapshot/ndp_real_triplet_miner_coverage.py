#!/usr/bin/env python3
"""只读重建 Batch-Hard Triplet miner 覆盖；无模型 forward、无 optimizer.step。

若没有历史逐步 batch 日志，本脚本使用真实 DIFFER sampler 重建 epoch/rank 的
batch，并明确标记 reconstructed_sampler；特征为固定 update0 PPU 导出，只做
关系覆盖近似，不能冒充历史实际梯度覆盖。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
sys.path.insert(0, str(Path("/data/DIFFER/scripts")))
import gpc_v2_train as v2  # noqa: E402


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def euclidean(x):
    sq = np.sum(x * x, axis=1, keepdims=True)
    return np.sqrt(np.maximum(sq + sq.T - 2 * (x @ x.T), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, nargs="+", default=[60, 61])
    ap.add_argument("--max-batches", type=int, default=0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bank = np.load(args.features, allow_pickle=False)
    features = bank["features"].astype(np.float32)
    image_index = bank["image_index"].astype(np.int64)
    pid = bank["pid"].astype(np.int64)
    clothes = bank["clothes"].astype(np.int64)
    if list(image_index) != list(range(len(image_index))):
        raise RuntimeError("要求完整、按 train index 排列的 update0 特征")
    graph = read_json(args.graph)
    relations = [r for r in graph["relations"] if r["graph_type"] == "difficult" and int(r["k_pid"]) == 5]
    relations.sort(key=lambda r: (int(r["anchor_order"]), int(r["negative_pid"])))
    relation_state = {}
    for r in relations:
        key = (int(r["anchor_local_index"]), int(r["negative_pid"]))
        if key in relation_state:
            raise RuntimeError("NDP graph 出现重复 anchor/PID 关系")
        relation_state[key] = {
            "anchor": key[0], "negative_pid": key[1], "relation": r,
            "batch_opportunities": 0, "hard_positive": 0, "hard_negative": 0,
            "full_miner_selected": 0, "active_selected": 0,
            "negative_pid_exposure": 0,
        }

    class Args:
        output = OLD_OUTPUT
        run_time = "20261015_155822"
        arm = "B0"
        stage = "two_seed"
        epochs = 60
        full_data = False
    Args.seed = args.seed
    split = read_json(OLD_OUTPUT / "development_split_manifest.json")
    protocol = v2.install_protocol(Args, split, args.output_dir, None)
    cfg = v2.p7.configure()
    dataset = v2.base.build_dataset(cfg)
    active_state = v2.b6.full_active_state(dataset)
    index_map = v2.base.build_index_map(dataset)
    original_loader = protocol["restore"][2]
    path_to_index = {str(item["image_path"]): i for i, item in enumerate(dataset.train)}
    miner_rows = []
    relation_batches = defaultdict(list)
    anchor_any = defaultdict(bool)
    anchor_active = defaultdict(bool)
    batch_count = 0
    for epoch in args.epochs:
        for rank in range(2):
            loader, _meta = original_loader(dataset, None, active_state, index_map, epoch, rank)
            for batch_position, batch_indices in enumerate(loader.batch_sampler):
                entries = [loader.dataset.samples[pos][0] for pos in batch_indices]
                batch_ids = sorted(path_to_index[str(item["image_path"])] for item in entries)
                batch_hash = digest(batch_ids)
                x = features[batch_ids]
                labels = pid[batch_ids]
                dist = euclidean(x)
                for local_i, anchor in enumerate(batch_ids):
                    pos_local = [j for j, label in enumerate(labels) if label == labels[local_i] and j != local_i]
                    neg_local = [j for j, label in enumerate(labels) if label != labels[local_i]]
                    if not pos_local or not neg_local:
                        continue
                    hp_local = pos_local[int(np.argmax(dist[local_i, pos_local]))]
                    hn_local = neg_local[int(np.argmin(dist[local_i, neg_local]))]
                    d_ap = float(dist[local_i, hp_local]); d_an = float(dist[local_i, hn_local])
                    difference = d_an - d_ap
                    soft_loss = float(np.logaddexp(0.0, difference))
                    soft_grad = float(1.0 / (1.0 + math.exp(min(max(-difference, -80.0), 80.0))))
                    active = soft_grad > 1e-8
                    hp, hn = batch_ids[hp_local], batch_ids[hn_local]
                    anchor_any[anchor] = True; anchor_active[anchor] |= active
                    miner_rows.append({
                        "seed": args.seed, "epoch": epoch, "rank": rank,
                        "batch_position": batch_position, "batch_hash": batch_hash,
                        "anchor": anchor, "anchor_pid": int(pid[anchor]),
                        "anchor_clothes": int(clothes[anchor]), "hard_positive": hp,
                        "hard_positive_pid": int(pid[hp]), "hard_negative": hn,
                        "hard_negative_pid": int(pid[hn]), "dist_ap": d_ap,
                        "dist_an": d_an, "soft_margin_loss": soft_loss,
                        "soft_margin_gradient": soft_grad, "active": active,
                    })
                    batch_set = set(batch_ids)
                    for key, state in relation_state.items():
                        if state["anchor"] != anchor:
                            continue
                        relation = state["relation"]
                        pos_set = set(int(v) for v in relation["positive_local_indices"])
                        neg_set = set(int(v) for v in relation["negative_local_indices"])
                        if not (neg_set & batch_set):
                            continue
                        state["negative_pid_exposure"] += 1
                        if not (pos_set & batch_set):
                            continue
                        state["batch_opportunities"] += 1
                        relation_batches[key].append((epoch, rank, batch_position, batch_hash))
                        state["hard_positive"] += int(hp in pos_set)
                        state["hard_negative"] += int(hn in neg_set)
                        full = hp in pos_set and hn in neg_set
                        state["full_miner_selected"] += int(full)
                        state["active_selected"] += int(full and active)
                batch_count += 1
                if args.max_batches and batch_count >= args.max_batches:
                    break
            if args.max_batches and batch_count >= args.max_batches:
                break
        if args.max_batches and batch_count >= args.max_batches:
            break

    relation_rows = []
    for key, state in sorted(relation_state.items()):
        r = state["relation"]
        relation_rows.append({
            "anchor": state["anchor"], "anchor_pid": int(pid[state["anchor"]]),
            "negative_pid": state["negative_pid"], "relation_sha256": r["relation_sha256"],
            "positive_count": len(r["positive_local_indices"]),
            "negative_image_count": len(r["negative_local_indices"]),
            "negative_pid_exposure": state["negative_pid_exposure"],
            "batch_opportunities": state["batch_opportunities"],
            "hard_positive_selected": state["hard_positive"],
            "hard_negative_selected": state["hard_negative"],
            "full_miner_selected": state["full_miner_selected"],
            "active_triplet_selected": state["active_selected"],
            "anchor_any_triplet": int(anchor_any[state["anchor"]]),
            "anchor_any_active_triplet": int(anchor_active[state["anchor"]]),
            "opportunity_batch_count": len(relation_batches[key]),
        })
    with (args.output_dir / "miner_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(miner_rows[0])); writer.writeheader(); writer.writerows(miner_rows)
    with (args.output_dir / "ndp_triplet_coverage_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(relation_rows[0])); writer.writeheader(); writer.writerows(relation_rows)
    relation_count = len(relation_rows)
    opportunity = sum(row["batch_opportunities"] > 0 for row in relation_rows)
    full = sum(row["full_miner_selected"] > 0 for row in relation_rows)
    active = sum(row["active_triplet_selected"] > 0 for row in relation_rows)
    summary = {
        "status": "PASS_READONLY_RECONSTRUCTED_BATCH_HARD_AUDIT",
        "seed": args.seed, "features": str(args.features),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "graph": str(args.graph), "graph_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
        "sampler_epochs": args.epochs, "rank_count": 2, "batch_count": batch_count,
        "miner_row_count": len(miner_rows), "ndp_relation_count": relation_count,
        "ndp_relations_with_batch_opportunity": opportunity,
        "ndp_relations_with_full_miner_selection": full,
        "ndp_relations_with_active_triplet": active,
        "coverage_opportunity": opportunity / relation_count,
        "coverage_full_miner": full / relation_count,
        "coverage_active_triplet": active / relation_count,
        "ndp_anchors_with_any_triplet": sum(int(v) > 0 for v in {state["anchor"]: int(anchor_any[state["anchor"]]) for state in relation_state.values()}.values()),
        "ndp_anchors_with_any_active_triplet": sum(int(v) > 0 for v in {state["anchor"]: int(anchor_active[state["anchor"]]) for state in relation_state.values()}.values()),
        "triplet_implementation": {
            "distance": "euclidean_dist", "feature_normalization_in_loss": False,
            "hard_positive": "maximum Euclidean distance among same PID",
            "hard_negative": "minimum Euclidean distance among different PID",
            "config_no_margin": True,
            "loss": "SoftMarginLoss(dist_an - dist_ap), not cosine ReLU proxy",
        },
        "historical_batch_log_available": False,
        "reconstructed_sampler": True,
        "feature_time": "update0 eval/no_grad normalized export; approximate coverage",
        "optimizer_step_executed": False, "official_query_gallery_used": False,
    }
    (args.output_dir / "triplet_miner_coverage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
