#!/usr/local/bin/python3.10
"""Build the pre-registered NDP-V2 update0-only K=5 reference graph.

This is an offline train-only artifact.  It never reads a future checkpoint,
query/gallery data, or model parameters.  The graph deliberately uses a
larger deterministic anchor set than N1 and keeps one representative image
for each of the five hardest negative PIDs at update0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--feature", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--anchor-count", type=int, default=2048)
    ap.add_argument("--k-pid", type=int, default=5)
    args = ap.parse_args()
    start = time.time()
    if args.k_pid != 5:
        raise ValueError("NDP-V2 is pre-registered with k_pid=5")

    data = np.load(args.feature, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    image_index = np.asarray(data["image_index"], dtype=np.int64)
    pid = np.asarray(data["pid"], dtype=np.int64)
    clothes = np.asarray(data["clothes"], dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != 1024:
        raise ValueError(f"expected [N,1024] feature bank, got {features.shape}")
    if len(set(image_index.tolist())) != len(image_index):
        raise ValueError("image_index is not unique")

    by_pid: dict[int, np.ndarray] = {}
    for p in sorted(set(pid.tolist())):
        by_pid[int(p)] = np.flatnonzero(pid == p)

    legal: list[int] = []
    positive_candidates: dict[int, np.ndarray] = {}
    for i in range(len(image_index)):
        same = by_pid[int(pid[i])]
        cross = same[clothes[same] != clothes[i]]
        candidates = cross if len(cross) >= 4 else same[same != i]
        if len(candidates) >= 4:
            legal.append(i)
            positive_candidates[i] = candidates
    legal.sort(key=lambda i: int(image_index[i]))
    if len(legal) < args.anchor_count:
        raise ValueError(f"only {len(legal)} legal anchors, requested {args.anchor_count}")
    anchors = legal[: args.anchor_count]

    # This matrix is about 80 MB for 2048 x 9576.  Keeping it in memory makes
    # the selection deterministic and avoids a hidden second pass over data.
    scores = features[anchors] @ features.T
    rows: list[dict[str, object]] = []
    for row_index, local_anchor in enumerate(anchors):
        anchor_pid = int(pid[local_anchor])
        candidates = positive_candidates[local_anchor]
        positive_order = candidates[np.argsort(-scores[row_index, candidates], kind="stable")[:4]]

        neg_best: list[tuple[float, int, int]] = []
        row_scores = scores[row_index]
        for neg_pid, neg_indices in by_pid.items():
            if neg_pid == anchor_pid:
                continue
            best_local = int(neg_indices[np.argmax(row_scores[neg_indices])])
            neg_best.append((float(row_scores[best_local]), int(neg_pid), best_local))
        neg_best.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen_neg = neg_best[: args.k_pid]
        if len(chosen_neg) != args.k_pid:
            raise ValueError("not enough negative PIDs")

        pos_values = [float(row_scores[i]) for i in positive_order]
        neg_values = [item[0] for item in chosen_neg]
        rows.append({
            "graph_type": "difficult",
            "k_pid": args.k_pid,
            "anchor_order": row_index,
            "anchor_image_index": int(image_index[local_anchor]),
            "anchor_local_index": int(local_anchor),
            "anchor_pid": anchor_pid,
            "positive_indices": [int(image_index[i]) for i in positive_order],
            "positive_local_indices": [int(i) for i in positive_order],
            "negative_pid": [item[1] for item in chosen_neg],
            "negative_indices": [int(image_index[item[2]]) for item in chosen_neg],
            "negative_local_indices": [int(item[2]) for item in chosen_neg],
            "reference_positive_mean": float(np.mean(pos_values)),
            "reference_negative_max": float(max(neg_values)),
            "reference_margin": float(np.mean(pos_values) - max(neg_values)),
            "positive_cross_clothes": bool(np.all(clothes[positive_order] != clothes[local_anchor])),
            "source_update": 0,
            "source_checkpoint_feature_file": str(args.feature),
            "future_nodes_used_for_selection": False,
        })

    summary = {
        "status": "PASS_NDP_V2_UPDATE0_ONLY_K5",
        "seed": args.seed,
        "feature_source": str(args.feature),
        "feature_sha256": sha256_file(args.feature),
        "source_update": 0,
        "anchor_count": len(rows),
        "legal_anchor_count": len(legal),
        "image_count": len(image_index),
        "pid_count": len(by_pid),
        "k_pid": args.k_pid,
        "selection_rule": "first anchors by image_index with four positives; top-similarity four same-PID positives and one max-similarity image for each of five hardest negative PIDs at update0",
        "future_nodes_used_for_selection": False,
        "official_query_gallery_used": False,
        "optimizer_step_executed": False,
        "elapsed_seconds": time.time() - start,
    }
    output = {"summary": summary, "relations": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["graph_sha256"] = sha256_file(args.output)
    args.output.write_text(json.dumps({"summary": summary, "relations": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
