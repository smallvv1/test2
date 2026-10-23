#!/usr/local/bin/python3.10
"""Train-only local metrics for an exported N5 update0/current feature pair."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = np.load(args.features, allow_pickle=False)
    updates = [int(x) for x in data["updates"]]
    if updates[0] != 0 or len(updates) != 2:
        raise RuntimeError(f"expected [0, current] updates, got {updates}")
    graph = [r for r in json.loads(args.graph.read_text(encoding="utf-8"))["relations"]
             if r.get("graph_type") == "difficult" and int(r.get("k_pid", -1)) == 5 and int(r.get("source_update", 0)) == 0]
    image_to_local = {int(x): i for i, x in enumerate(data["image_index"])}
    features = np.asarray(data["features"], dtype=np.float32)
    pids = np.asarray(data["pid"], dtype=np.int64)
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite exported features")
    rows = []
    reference_error_pids = None
    for node_i, update in enumerate(updates):
        z = features[node_i]
        margins = []
        correct = []
        error_pids = set()
        for row in graph:
            ai = image_to_local[int(row["anchor_image_index"])]
            pos = z[[image_to_local[int(x)] for x in row["positive_indices"]]]
            neg = z[[image_to_local[int(x)] for x in row["negative_indices"]]]
            pos_mean = float(np.mean(pos @ z[ai]))
            neg_sim = neg @ z[ai]
            margins.append(pos_mean - float(np.max(neg_sim)))
            correct.append(pos_mean - float(np.max(neg_sim)) > 0.0)
            for npid, sim in zip(row["negative_pid"], neg_sim):
                if pos_mean - float(sim) <= 0.0:
                    error_pids.add((int(row["anchor_image_index"]), int(npid)))
        if reference_error_pids is None:
            reference_error_pids = error_pids
        purity = []
        for row in graph:
            ai = image_to_local[int(row["anchor_image_index"])]
            sims = z @ z[ai]
            order = np.argsort(-sims, kind="stable")
            order = [j for j in order if j != ai][:10]
            purity.append(float(np.mean(pids[order] == pids[ai])))
        rows.append({
            "seed": args.seed, "update": update, "relation_count": len(graph),
            "correct_relation_count": int(np.sum(correct)), "correct_relation_rate": float(np.mean(correct)),
            "mean_margin": float(np.mean(margins)), "median_margin": float(np.median(margins)),
            "q25_margin": float(np.quantile(margins, 0.25)), "min_margin": float(np.min(margins)),
            "current_error_pid_count": len(error_pids),
            "new_error_pid_count_vs_update0": len(error_pids - reference_error_pids),
            "recovered_pid_count_vs_update0": len(reference_error_pids - error_pids),
            "purity_at_10_train_bank": float(np.mean(purity)),
            "purity_at_10_q25": float(np.quantile(purity, 0.25)),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = {"status": "PASS_NDP_V5_TRAIN_ONLY_LOCAL_METRICS", "seed": args.seed,
                "features": str(args.features), "features_sha256": sha256_file(args.features),
                "graph": str(args.graph), "graph_sha256": sha256_file(args.graph),
                "output_csv": str(args.output.with_suffix(".csv")), "optimizer_step_executed": False,
                "official_query_gallery_used": False, "rows": rows}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
