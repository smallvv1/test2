#!/usr/local/bin/python3.10
"""Train-only local margin/error/Purity@10 metrics for the V2 graph."""
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


def load_graph(path: Path) -> list[dict]:
    rows = [r for r in json.loads(path.read_text(encoding="utf-8"))["relations"]
            if r.get("graph_type") == "difficult" and int(r.get("k_pid", -1)) == 5 and int(r.get("source_update", 0)) == 0]
    rows.sort(key=lambda r: int(r["anchor_order"]))
    if len(rows) != 2048:
        raise RuntimeError(f"expected V2 2048 relations, got {len(rows)}")
    if any(len(r["negative_indices"]) != 5 or len(set(r["negative_pid"])) != 5 for r in rows):
        raise RuntimeError("V2 graph is not one anchor/five distinct negative PIDs")
    return rows


def evaluate(name: str, feature_path: Path, graph: list[dict], out_dir: Path) -> tuple[list[dict], list[dict]]:
    data = np.load(feature_path, allow_pickle=False)
    updates = [int(x) for x in data["updates"]]
    if updates != [0, 596]:
        raise RuntimeError(f"{name}: unexpected updates {updates}")
    image_to_local = {int(x): i for i, x in enumerate(data["image_index"])}
    z_all = np.asarray(data["features"], dtype=np.float32)
    pids = np.asarray(data["pid"], dtype=np.int64)
    if not np.isfinite(z_all).all():
        raise RuntimeError(f"{name}: non-finite features")
    relation_rows: list[dict] = []
    node_rows: list[dict] = []
    ref_correct = None
    ref_error_pids: set[tuple[int, int]] | None = None
    anchors = [int(r["anchor_image_index"]) for r in graph]
    for node_i, update in enumerate(updates):
        z = z_all[node_i]
        margins = []; correct = []; error_pids = set()
        current_relation: list[dict] = []
        for ri, row in enumerate(graph):
            aid = int(row["anchor_image_index"]); ai = image_to_local[aid]
            pos = z[[image_to_local[int(x)] for x in row["positive_indices"]]]
            neg = z[[image_to_local[int(x)] for x in row["negative_indices"]]]
            pos_mean = float(np.mean(pos @ z[ai])); neg_sim = neg @ z[ai]
            margin = pos_mean - float(np.max(neg_sim)); is_correct = margin > 0
            margins.append(margin); correct.append(is_correct)
            for npid, sim in zip(row["negative_pid"], neg_sim):
                if float(pos_mean - float(sim)) <= 0:
                    error_pids.add((aid, int(npid)))
            current_relation.append({
                "arm": name, "update": update, "relation_index": ri, "anchor_image_index": aid,
                "anchor_pid": int(row["anchor_pid"]), "negative_pid": json.dumps([int(x) for x in row["negative_pid"]]),
                "positive_indices": json.dumps([int(x) for x in row["positive_indices"]]), "negative_indices": json.dumps([int(x) for x in row["negative_indices"]]),
                "margin": margin, "positive_mean_similarity": pos_mean, "negative_max_similarity": float(np.max(neg_sim)),
                "correct": int(is_correct),
            })
        margins_np = np.asarray(margins); correct_np = np.asarray(correct, dtype=bool)
        if update == 0:
            ref_correct = correct_np.copy(); ref_error_pids = set(error_pids)
        assert ref_correct is not None and ref_error_pids is not None
        new_errors = ref_correct & ~correct_np
        recovered = ~ref_correct & correct_np
        for r, bad in zip(current_relation, new_errors):
            r["reference_correct"] = int(bool(ref_correct[int(r["relation_index"])])); r["new_error_from_update0"] = int(bad); r["recovered_from_update0"] = int(recovered[int(r["relation_index"])])
        relation_rows.extend(current_relation)
        purity = []
        for aid in anchors:
            ai = image_to_local[aid]; sims = z @ z[ai]; order = np.argsort(-sims, kind="stable"); order = [j for j in order if j != ai][:10]
            purity.append(float(np.mean(pids[order] == pids[ai])))
        node_rows.append({
            "arm": name, "update": update, "relation_count": len(graph), "correct_relation_count": int(correct_np.sum()), "correct_relation_rate": float(correct_np.mean()),
            "mean_margin": float(margins_np.mean()), "median_margin": float(np.median(margins_np)), "q25_margin": float(np.quantile(margins_np, .25)), "q10_margin": float(np.quantile(margins_np, .10)), "min_margin": float(margins_np.min()),
            "new_error_relation_count_vs_update0": int(new_errors.sum()), "recovered_relation_count_vs_update0": int(recovered.sum()),
            "current_error_pid_count": len(error_pids), "new_error_pid_count_vs_update0": len(error_pids - ref_error_pids), "recovered_pid_count_vs_update0": len(ref_error_pids - error_pids),
            "anchor_count": len(anchors), "purity_at_10_train_bank": float(np.mean(purity)), "purity_at_10_q25": float(np.quantile(purity, .25)), "purity_at_10_q10": float(np.quantile(purity, .10)),
        })
    return relation_rows, node_rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--graph", type=Path, required=True); ap.add_argument("--b0-features", type=Path, required=True); ap.add_argument("--n2-features", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True); args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True); graph = load_graph(args.graph)
    b0_rel, b0_node = evaluate("B0", args.b0_features, graph, args.output_dir); n2_rel, n2_node = evaluate("N2", args.n2_features, graph, args.output_dir)
    rel_path = args.output_dir / f"local_relation_metrics_seed{args.seed}.csv"; node_path = args.output_dir / f"local_node_summary_seed{args.seed}.csv"
    with rel_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list((b0_rel + n2_rel)[0])); w.writeheader(); w.writerows(b0_rel + n2_rel)
    with node_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list((b0_node + n2_node)[0])); w.writeheader(); w.writerows(b0_node + n2_node)
    manifest = {"status": "PASS_NDP_V2_TRAIN_ONLY_LOCAL_METRICS", "seed": args.seed, "graph": str(args.graph), "graph_sha256": sha256_file(args.graph), "b0_features": str(args.b0_features), "b0_features_sha256": sha256_file(args.b0_features), "n2_features": str(args.n2_features), "n2_features_sha256": sha256_file(args.n2_features), "relation_csv": str(rel_path), "node_csv": str(node_path), "optimizer_step_executed": False, "official_query_gallery_used": False, "margin_definition": "mean positive cosine minus max of five negative-PID cosine similarities", "purity_definition": "mean top-10 same-PID fraction over the 9576-image train-only bank"}
    (args.output_dir / "local_metrics_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
