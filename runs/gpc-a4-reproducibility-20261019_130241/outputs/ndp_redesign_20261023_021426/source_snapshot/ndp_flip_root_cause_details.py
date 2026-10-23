#!/usr/bin/env python3.10
"""补充全部 200->400 翻转的正参考/负冒名者分解。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def margin_parts(features: np.ndarray, row: dict, image_to_pos: dict[int, int]) -> tuple[float, float]:
    anchor = features[image_to_pos[int(row["anchor_image_index"])]].astype(np.float32)
    anchor = anchor / max(float(np.linalg.norm(anchor)), 1e-12)
    pos = features[[image_to_pos[int(v)] for v in row["positive_indices"]]].astype(np.float32)
    neg = features[[image_to_pos[int(v)] for v in row["negative_indices"]]].astype(np.float32)
    pos = pos / np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
    neg = neg / np.maximum(np.linalg.norm(neg, axis=1, keepdims=True), 1e-12)
    return float(np.mean(pos @ anchor)), float(np.max(neg @ anchor))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-output", type=Path, required=True)
    ap.add_argument("--graph-root", type=Path, required=True)
    ap.add_argument("--feature-root", type=Path, required=True)
    args = ap.parse_args()
    for seed in (1234, 4321):
        flip_path = args.run_output / f"seed{seed}/missed_flip_rows_seed{seed}.csv"
        relation_path = args.run_output / f"seed{seed}/relation_evidence_seed{seed}.csv"
        graph = json.loads((args.graph_root / f"seed{seed}/multi_pid_reference_graph.json").read_text(encoding="utf-8"))
        relations = [r for r in graph["relations"] if r["graph_type"] == "difficult" and int(r["k_pid"]) == 5]
        relations.sort(key=lambda r: (int(r["anchor_order"]), int(r["negative_pid"])))
        bank = np.load(args.feature_root / f"seed{seed}/k5_trajectory_features_seed{seed}.npz", allow_pickle=False)
        updates = [int(x) for x in bank["updates"]]
        image_to_pos = {int(x): i for i, x in enumerate(bank["image_index"])}
        features = {u: bank["features"][updates.index(u)] for u in updates}
        out = []
        with flip_path.open(encoding="utf-8") as fh:
            for base in csv.DictReader(fh):
                r = relations[int(base["relation_index"])]
                p0, n0 = margin_parts(features[0], r, image_to_pos)
                p200, n200 = margin_parts(features[200], r, image_to_pos)
                out.append({
                    **base,
                    "positive_mean_similarity_0": p0,
                    "positive_mean_similarity_200": p200,
                    "positive_reference_delta_0_200": p200 - p0,
                    "negative_max_similarity_0": n0,
                    "negative_max_similarity_200": n200,
                    "negative_impostor_rise_0_200": n200 - n0,
                    "category_history_unreliable": int(not bool(int(base["reliable_history"]))),
                    "category_reliable_current_degraded": int(bool(int(base["reliable_history"])) and bool(int(base["observed_degradation"]))),
                    "category_reliable_current_not_degraded_then_flip": int(bool(int(base["reliable_history"])) and not bool(int(base["observed_degradation"]))),
                    "category_initial_near_zero_or_unstable": int(float(base["history_min_margin"]) < 0.05 or float(base["history_std"]) > 0.05),
                    "category_positive_reference_changed_gt_0.02": int(abs(p200 - p0) > 0.02),
                    "category_negative_impostor_rise_gt_0.02": int(n200 - n0 > 0.02),
                })
        out_path = args.run_output / f"seed{seed}/missed_flip_root_cause_details_seed{seed}.csv"
        fields = list(out[0]) if out else []
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(out)
        summary = {
            "seed": seed,
            "flip_count": len(out),
            "category_counts": {key: int(sum(int(row[key]) for row in out)) for key in (
                "category_history_unreliable", "category_reliable_current_degraded", "category_reliable_current_not_degraded_then_flip",
                "category_initial_near_zero_or_unstable", "category_positive_reference_changed_gt_0.02", "category_negative_impostor_rise_gt_0.02")},
            "thresholds_pre_registered": {"positive_reference_delta_abs": 0.02, "negative_impostor_rise": 0.02},
            "future_update400_used_only_for_event_label": True,
            "selection_uses_update400": False,
            "rows_csv": str(out_path),
        }
        current_error_path = args.run_output / f"seed{seed}/current_error_rows_seed{seed}.csv"
        with relation_path.open(encoding="utf-8") as fh:
            relation_rows = [row for row in csv.DictReader(fh) if int(row["current_correct"]) == 0]
        with current_error_path.open("w", newline="", encoding="utf-8") as fh:
            fields = list(relation_rows[0]) if relation_rows else []
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(relation_rows)
        summary["current_error_count"] = len(relation_rows)
        summary["current_error_rows_csv"] = str(current_error_path)
        (args.run_output / f"seed{seed}/missed_flip_root_cause_summary_seed{seed}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
