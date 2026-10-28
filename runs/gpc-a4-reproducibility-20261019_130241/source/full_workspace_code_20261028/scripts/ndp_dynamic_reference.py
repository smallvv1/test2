#!/usr/local/bin/python3.10
"""Build an NDYN train-only graph from two B0 feature snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DELTA = 0.005
MAX_CURRENT_MARGIN = 0.05


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

    f0, i0, u0 = read_features(args.t0)
    f1, i1, u1 = read_features(args.t1)
    if not np.array_equal(i0, i1):
        raise RuntimeError("t0/t1 image_index arrays are not identical")
    graph0 = json.loads(args.graph_t0.read_text(encoding="utf-8"))
    if graph0.get("query_gallery_used") or not graph0.get("train_only"):
        raise RuntimeError("t0 graph is not train-only")
    image_to_local = {int(value): index for index, value in enumerate(i0.tolist())}

    relations = []
    selected_pid_count = 0
    selected_relation_count = 0
    selected_anchor_ids = set()
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
        positive0 = (f0[positives] @ f0[anchor]).mean()
        positive1 = (f1[positives] @ f1[anchor]).mean()
        margin0 = (positive0 - (f0[negatives] @ f0[anchor])).astype(np.float64).tolist()
        margin1 = (positive1 - (f1[negatives] @ f1[anchor])).astype(np.float64).tolist()
        dynamic_valid = [
            bool(m0 > 0.0 and m1 > 0.0 and m1 < m0 - DELTA and m1 < MAX_CURRENT_MARGIN)
            for m0, m1 in zip(margin0, margin1)
        ]
        selected = int(sum(dynamic_valid))
        selected_pid_count += selected
        if selected:
            selected_relation_count += 1
            selected_anchor_ids.add(anchor_id)
        row = dict(original)
        row["dynamic_valid_pid"] = dynamic_valid
        row["dynamic_m0_pid_margin"] = [float(value) for value in margin0]
        row["dynamic_m1_pid_margin"] = [float(value) for value in margin1]
        row["dynamic_delta"] = DELTA
        relations.append(row)

    output_graph = {
        "version": "ndp-dynamic-prcc-train-only-k5",
        "dynamic": True,
        "dynamic_delta": DELTA,
        "dynamic_current_margin_cap": MAX_CURRENT_MARGIN,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "feature_path": str(args.t1),
        "feature_t0_path": str(args.t0),
        "graph_t0_path": str(args.graph_t0),
        "train_only": True,
        "query_gallery_used": False,
        "source_update_t0": u0,
        "source_update_t1": u1,
        "anchor_count": len(selected_anchor_ids),
        "relation_count": len(relations),
        "dynamic_relation_count": selected_relation_count,
        "dynamic_pid_count": selected_pid_count,
        "selection": "M0>0, M1>0, M1<M0-0.005, M1<0.05 using identical train-only endpoints",
        "relations": relations,
    }
    args.output_graph.parent.mkdir(parents=True, exist_ok=True)
    args.output_graph.write_text(json.dumps(output_graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "status": "PASS",
        "t0": {"path": str(args.t0), "sha256": sha256_file(args.t0), "update": u0},
        "t1": {"path": str(args.t1), "sha256": sha256_file(args.t1), "update": u1},
        "graph_t0": {"path": str(args.graph_t0), "sha256": sha256_file(args.graph_t0)},
        "output_graph": {"path": str(args.output_graph), "sha256": sha256_file(args.output_graph)},
        "image_index_equal": True,
        "delta": DELTA,
        "current_margin_cap": MAX_CURRENT_MARGIN,
        "relation_count": len(relations),
        "dynamic_relation_count": selected_relation_count,
        "dynamic_pid_count": selected_pid_count,
        "dynamic_anchor_count": len(selected_anchor_ids),
        "query_gallery_used": False,
    }
    args.output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
