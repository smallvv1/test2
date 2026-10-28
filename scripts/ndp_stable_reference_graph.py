#!/usr/bin/env python3
"""构建 train-only 的稳定身份邻域图。

可接受多个冻结特征快照。默认只计算固定抽样 anchor，避免重复构造全库
N×N 矩阵。若快照不是同一训练轨迹的相邻状态，结果明确标为非时间有效。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ndp_risk_common import (
    assert_aligned,
    batched_scores,
    digest_arrays,
    fixed_anchor_indices,
    load_feature_bank,
    normalize,
    positive_indices,
    top_negative_indices,
)


def parse_snapshot(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("快照格式必须是 label=/path/bank.npz")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("快照 label/path 不能为空")
    return label, Path(path)


def build_snapshot(bank: dict, anchors: np.ndarray, k: int, pos_count: int) -> dict:
    features = normalize(bank["features"])
    neg_indices = np.full((len(anchors), k), -1, dtype=np.int64)
    pos_indices = []
    hubness = np.zeros(len(features), dtype=np.int64)
    for start, scores in batched_scores(features[anchors], features):
        for local, anchor in enumerate(anchors[start : start + len(scores)]):
            row = scores[local]
            neg = top_negative_indices(row, int(anchor), bank["pid"], k)
            neg_indices[start + local, : len(neg)] = neg
            pos_indices.append(positive_indices(row, int(anchor), bank["pid"], pos_count))
            hubness[neg] += 1
    pos_max = max((len(x) for x in pos_indices), default=0)
    pos_arr = np.full((len(anchors), pos_max), -1, dtype=np.int64)
    for row, values in enumerate(pos_indices):
        pos_arr[row, : len(values)] = values
    return {
        "negative_indices": neg_indices,
        "positive_indices": pos_arr,
        "hubness": hubness,
        "graph_sha256": digest_arrays(neg_indices, pos_arr),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--snapshot", action="append", type=parse_snapshot, required=True)
    ap.add_argument("--sample-anchors", type=int, default=1024)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--positive-count", type=int, default=4)
    ap.add_argument("--independent-endpoints", action="store_true")
    args = ap.parse_args()
    if len(args.snapshot) < 2:
        raise SystemExit("至少需要两个快照")
    first_label, first_path = args.snapshot[0]
    first = load_feature_bank(first_path)
    n = len(first["pid"])
    anchors = fixed_anchor_indices(n, args.seed, args.sample_anchors)
    snapshots = []
    for label, path in args.snapshot:
        bank = load_feature_bank(path)
        assert_aligned(first, bank)
        graph = build_snapshot(bank, anchors, args.k, args.positive_count)
        snapshots.append((label, bank, graph))

    pair_rows = []
    for row, anchor in enumerate(anchors):
        sets = {label: set(graph["negative_indices"][row].tolist()) - {-1} for label, _, graph in snapshots}
        psets = {label: set(graph["positive_indices"][row].tolist()) - {-1} for label, _, graph in snapshots}
        labels = list(sets)
        a, b = labels[0], labels[1]
        def jaccard(x, y):
            union = len(x | y)
            return float(len(x & y) / union) if union else 1.0
        pair_rows.append({
            "anchor": int(anchor),
            "pid": int(first["pid"][anchor]),
            "image_path": str(first["image_path"][anchor]),
            "negative_jaccard": jaccard(sets[a], sets[b]),
            "positive_jaccard": jaccard(psets[a], psets[b]),
            "negative_intersection": int(len(sets[a] & sets[b])),
            "negative_union": int(len(sets[a] | sets[b])),
            "reference_stability": jaccard(sets[a], sets[b]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"seed{args.seed}_reference_graph.npz",
        anchors=anchors,
        **{f"{label}_negative": graph["negative_indices"] for label, _, graph in snapshots},
        **{f"{label}_positive": graph["positive_indices"] for label, _, graph in snapshots},
    )
    with (args.output_dir / f"seed{args.seed}_anchor_graph.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(pair_rows[0]) if pair_rows else ["anchor"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(pair_rows)
    summary = {
        "status": "PASS",
        "seed": args.seed,
        "n_train_images": n,
        "n_train_pids": int(np.unique(first["pid"]).size),
        "sample_anchor_count": int(len(anchors)),
        "k": args.k,
        "positive_count": args.positive_count,
        "official_query_gallery_used": False,
        "snapshot_temporal_valid": not bool(args.independent_endpoints),
        "temporal_warning": "快照来自独立 Bio/GPC E60 终点，不是同一训练轨迹相邻状态" if args.independent_endpoints else None,
        "snapshots": [
            {"label": label, "path": bank["path"], "sha256": bank["sha256"], "graph_sha256": graph["graph_sha256"]}
            for label, bank, graph in snapshots
        ],
        "stability": {
            "negative_jaccard_mean": float(np.mean([r["negative_jaccard"] for r in pair_rows])) if pair_rows else None,
            "negative_jaccard_median": float(np.median([r["negative_jaccard"] for r in pair_rows])) if pair_rows else None,
            "positive_jaccard_mean": float(np.mean([r["positive_jaccard"] for r in pair_rows])) if pair_rows else None,
        },
        "hubness": {
            label: {
                "max": int(graph["hubness"].max()),
                "mean_nonzero": float(graph["hubness"][graph["hubness"] > 0].mean()) if np.any(graph["hubness"] > 0) else 0.0,
                "top10_sum": int(np.sort(graph["hubness"])[-10:].sum()),
            }
            for label, _, graph in snapshots
        },
        "pair_rows": str(args.output_dir / f"seed{args.seed}_anchor_graph.csv"),
        "graph_file": str(args.output_dir / f"seed{args.seed}_reference_graph.npz"),
    }
    (args.output_dir / f"seed{args.seed}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "seed": args.seed, "anchors": len(anchors), "stability": summary["stability"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
