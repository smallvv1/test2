#!/usr/local/bin/python3.10
"""Train-only local metrics for the fixed K=5 graph.

This is an offline evaluator: it consumes detached feat[0] exports from the
same fixed graph for B0 and N1.  It never performs a forward/backward or an
optimizer step and it never reads LTCC query/gallery data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


UPDATES = (0, 20, 50, 100, 200, 400)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_relations(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload["relations"]
            if r.get("graph_type") == "difficult" and int(r.get("k_pid", -1)) == 5]
    rows.sort(key=lambda r: (int(r["anchor_order"]), int(r["negative_pid"])))
    if len(rows) != 2560:
        raise RuntimeError(f"expected fixed K=5 graph with 2560 relations, got {len(rows)}")
    return rows


def evaluate_arm(name: str, feature_path: Path, relations: list[dict], out_dir: Path) -> tuple[list[dict], list[dict]]:
    data = np.load(feature_path, allow_pickle=False)
    updates = [int(x) for x in data["updates"]]
    if tuple(updates) != UPDATES:
        raise RuntimeError(f"{name}: unexpected updates {updates}")
    image_to_local = {int(x): i for i, x in enumerate(data["image_index"])}
    features = np.asarray(data["features"], dtype=np.float32)
    pids = np.asarray(data["pid"], dtype=np.int64)
    relation_rows: list[dict] = []
    node_rows: list[dict] = []

    # Purity is measured in the exported train-only K=5 candidate bank, not
    # on the official query/gallery split.
    anchors = sorted({int(r["anchor_image_index"]) for r in relations})
    anchor_local = [image_to_local[x] for x in anchors]
    anchor_pid = {x: int(pids[image_to_local[x]]) for x in anchors}
    for node_index, update in enumerate(updates):
        z = features[node_index]
        if not np.isfinite(z).all():
            raise RuntimeError(f"{name} update{update}: non-finite feature")
        margins = []
        correct = []
        for relation_index, row in enumerate(relations):
            a_id = int(row["anchor_image_index"])
            a = z[image_to_local[a_id]]
            pos = z[[image_to_local[int(x)] for x in row["positive_indices"]]]
            neg = z[[image_to_local[int(x)] for x in row["negative_indices"]]]
            pos_mean = float(np.mean(pos @ a))
            neg_max = float(np.max(neg @ a))
            margin = pos_mean - neg_max
            margins.append(margin)
            is_correct = margin > 0.0
            correct.append(is_correct)
            if update == 0:
                ref_correct = is_correct
            else:
                ref_correct = bool(relation_rows[relation_index]["reference_correct"])
            relation_rows.append({
                "arm": name,
                "update": update,
                "relation_index": relation_index,
                "anchor_image_index": a_id,
                "anchor_pid": int(row["anchor_pid"]),
                "negative_pid": int(row["negative_pid"]),
                "positive_indices": json.dumps([int(x) for x in row["positive_indices"]]),
                "negative_indices": json.dumps([int(x) for x in row["negative_indices"]]),
                "margin": margin,
                "positive_mean_similarity": pos_mean,
                "negative_max_similarity": neg_max,
                "correct": int(is_correct),
                "reference_correct": int(ref_correct),
                "new_error_from_reference": int(ref_correct and not is_correct),
                "recovered_from_reference": int((not ref_correct) and is_correct),
            })
        margins_np = np.asarray(margins)
        correct_np = np.asarray(correct, dtype=bool)
        if update == 0:
            ref_correct_by_relation = correct_np.copy()
        new_error = ref_correct_by_relation & ~correct_np
        recovered = ~ref_correct_by_relation & correct_np
        # PID-level counts collapse the five negative-PID relations per anchor.
        error_pairs = {(int(r["anchor_image_index"]), int(r["negative_pid"]))
                       for r, bad in zip(relations, new_error) if bad}
        recovered_pairs = {(int(r["anchor_image_index"]), int(r["negative_pid"]))
                           for r, good in zip(relations, recovered) if good}
        purity_values = []
        for a_id in anchors:
            ai = image_to_local[a_id]
            sims = z @ z[ai]
            order = np.argsort(-sims, kind="stable")
            order = [j for j in order if j != ai][:10]
            purity_values.append(float(np.mean(pids[order] == anchor_pid[a_id])))
        node_rows.append({
            "arm": name,
            "update": update,
            "relation_count": len(relations),
            "correct_relation_count": int(correct_np.sum()),
            "correct_relation_rate": float(correct_np.mean()),
            "mean_margin": float(margins_np.mean()),
            "median_margin": float(np.median(margins_np)),
            "q25_margin": float(np.quantile(margins_np, .25)),
            "q10_margin": float(np.quantile(margins_np, .10)),
            "min_margin": float(margins_np.min()),
            "new_error_relation_count_vs_update0": int(new_error.sum()),
            "recovered_relation_count_vs_update0": int(recovered.sum()),
            "new_error_pid_count_vs_update0": len(error_pairs),
            "recovered_pid_count_vs_update0": len(recovered_pairs),
            "current_error_pid_count": int((~correct_np).sum()),
            "anchor_count": len(anchors),
            "purity_at_10_exported_bank": float(np.mean(purity_values)),
            "purity_at_10_q25": float(np.quantile(purity_values, .25)),
            "purity_at_10_q10": float(np.quantile(purity_values, .10)),
        })
    return relation_rows, node_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--b0-features", type=Path, required=True)
    ap.add_argument("--n1-features", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relations = load_relations(args.graph)
    b0_rel, b0_node = evaluate_arm("B0", args.b0_features, relations, args.output_dir)
    n1_rel, n1_node = evaluate_arm("N1", args.n1_features, relations, args.output_dir)
    relation_path = args.output_dir / f"local_relation_metrics_seed{args.seed}.csv"
    node_path = args.output_dir / f"local_node_summary_seed{args.seed}.csv"
    all_rel = b0_rel + n1_rel
    with relation_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rel[0]))
        writer.writeheader(); writer.writerows(all_rel)
    all_nodes = b0_node + n1_node
    with node_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_nodes[0]))
        writer.writeheader(); writer.writerows(all_nodes)
    manifest = {
        "status": "PASS_TRAIN_ONLY_FIXED_K5_METRICS",
        "seed": args.seed,
        "graph": str(args.graph), "graph_sha256": sha256_file(args.graph),
        "b0_features": str(args.b0_features), "b0_features_sha256": sha256_file(args.b0_features),
        "n1_features": str(args.n1_features), "n1_features_sha256": sha256_file(args.n1_features),
        "relation_csv": str(relation_path), "node_csv": str(node_path),
        "optimizer_step_executed": False, "official_query_gallery_used": False,
        "purity_definition": "mean top-10 same-PID fraction in the exported train-only fixed K=5 endpoint bank",
        "margin_definition": "mean positive cosine minus max negative-PID cosine",
    }
    (args.output_dir / "local_metrics_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "seed": args.seed, "node_csv": str(node_path), "relation_csv": str(relation_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
