#!/usr/local/bin/python3.10
"""Build a fixed PRCC train-only NDYN-TOP5 graph from t0/t1 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TOP5_FRACTION = 0.05
TARGET_DELTA_CAP = 0.005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_features(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    data = np.load(path, allow_pickle=False)
    features = data["features"].astype(np.float32)
    image_index = data["image_index"].astype(np.int32)
    update = int(np.asarray(data["update"]).item())
    if features.ndim != 2 or features.shape[1] != 1024:
        raise RuntimeError(f"unexpected feature shape at {path}: {features.shape}")
    if len(features) != len(image_index):
        raise RuntimeError(f"feature/index length mismatch at {path}")
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return features, image_index, update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--graph-t0", type=Path, required=True)
    parser.add_argument("--output-graph", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    f0, image0, update0 = read_features(args.t0)
    f1, image1, update1 = read_features(args.t1)
    if not np.array_equal(image0, image1):
        raise RuntimeError("t0/t1 image_index arrays are not identical")
    graph0 = json.loads(args.graph_t0.read_text(encoding="utf-8"))
    if graph0.get("query_gallery_used") or not graph0.get("train_only"):
        raise RuntimeError("t0 graph is not train-only")
    image_to_local = {int(value): index for index, value in enumerate(image0.tolist())}

    rows = []
    candidates = []
    for relation_index, original in enumerate(graph0["relations"]):
        anchor_id = int(original["anchor_image_index"])
        positive_ids = [int(v) for v in original["positive_indices"]]
        negative_ids = [int(v) for v in original["negative_indices"]]
        endpoint_ids = [anchor_id, *positive_ids, *negative_ids]
        if any(value not in image_to_local for value in endpoint_ids):
            raise RuntimeError(f"missing endpoint at relation {relation_index}")
        anchor = image_to_local[anchor_id]
        positives = [image_to_local[value] for value in positive_ids]
        negatives = [image_to_local[value] for value in negative_ids]
        positive0 = float((f0[positives] @ f0[anchor]).mean())
        positive1 = float((f1[positives] @ f1[anchor]).mean())
        margin0 = [float(positive0 - value) for value in (f0[negatives] @ f0[anchor])]
        margin1 = [float(positive1 - value) for value in (f1[negatives] @ f1[anchor])]
        drop = [float(m0 - m1) for m0, m1 in zip(margin0, margin1)]
        valid = [bool(m0 > 0.0 and m1 > 0.0 and d > 0.0) for m0, m1, d in zip(margin0, margin1, drop)]
        scores = [float(d / (m1 + 0.02)) if ok else float("-inf") for d, m1, ok in zip(drop, margin1, valid)]
        row = dict(original)
        row["dynamic_valid_pid"] = valid
        row["dynamic_m0_pid_margin"] = margin0
        row["dynamic_m1_pid_margin"] = margin1
        row["dynamic_drop_pid"] = drop
        row["dynamic_risk_score_pid"] = scores
        row["dynamic_top5_valid_pid"] = [False] * len(scores)
        rows.append(row)
        for pid_index, score in enumerate(scores):
            if math.isfinite(score):
                candidates.append((score, relation_index, pid_index))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    top5_count = max(1, math.ceil(TOP5_FRACTION * len(candidates))) if candidates else 0
    selected = candidates[:top5_count]
    selected_keys = {(relation_index, pid_index) for _score, relation_index, pid_index in selected}
    for relation_index, row in enumerate(rows):
        row["dynamic_top5_valid_pid"] = [
            (relation_index, pid_index) in selected_keys
            for pid_index in range(len(row["negative_indices"]))
        ]

    selected_anchors = {
        int(rows[relation_index]["anchor_image_index"])
        for _score, relation_index, _pid_index in selected
    }
    output_graph = {
        "version": "ndp-dynamic-top5-prcc-train-only-k5",
        "dynamic": True,
        "dynamic_top5": True,
        "top5_fraction": TOP5_FRACTION,
        "target_delta_cap": TARGET_DELTA_CAP,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "feature_path": str(args.t1),
        "feature_t0_path": str(args.t0),
        "graph_t0_path": str(args.graph_t0),
        "train_only": True,
        "query_gallery_used": False,
        "source_update_t0": update0,
        "source_update_t1": update1,
        "relation_count": len(rows),
        "candidate_pid_count": len(candidates),
        "top5_pid_count": len(selected),
        "top5_anchor_count": len(selected_anchors),
        "selection": "M0>0, M1>0, M0-M1>0; risk=(M0-M1)/(M1+0.02); deterministic top 5%",
        "relations": rows,
    }
    args.output_graph.parent.mkdir(parents=True, exist_ok=True)
    args.output_graph.write_text(json.dumps(output_graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "status": "PASS",
        "t0": {"path": str(args.t0), "sha256": sha256_file(args.t0), "update": update0},
        "t1": {"path": str(args.t1), "sha256": sha256_file(args.t1), "update": update1},
        "graph_t0": {"path": str(args.graph_t0), "sha256": sha256_file(args.graph_t0)},
        "output_graph": {"path": str(args.output_graph), "sha256": sha256_file(args.output_graph)},
        "image_index_equal": True,
        "top5_fraction": TOP5_FRACTION,
        "target_delta_cap": TARGET_DELTA_CAP,
        "relation_count": len(rows),
        "candidate_pid_count": len(candidates),
        "top5_pid_count": len(selected),
        "top5_anchor_count": len(selected_anchors),
        "query_gallery_used": False,
    }
    args.output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
