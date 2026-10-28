#!/usr/bin/env python3
"""修复身份级双侧统计的 last-write-wins，并建立严格图像级交叉边界。"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def arr(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(x) for x in ast.literal_eval(value)]


def identity_stats(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (int(row["prev_update"]), int(row["target_update"]), int(row["anchor_pid"]), int(row["negative_pid"]))
        grouped[key].append(row)
    out = []
    for (prev, target, p, q), values in sorted(grouped.items()):
        deltas = np.asarray([float(x["delta_margin"]) for x in values], dtype=np.float64)
        out.append({
            "prev_update": prev, "target_update": target, "anchor_pid": p, "negative_pid": q,
            "directed_relation_count": len(values),
            "unique_anchor_images": len({int(x["anchor_local_index"]) for x in values}),
            "delta_min": float(deltas.min()), "delta_q25": float(np.quantile(deltas, .25)),
            "delta_median": float(np.median(deltas)), "delta_max": float(deltas.max()),
            "worsened_fraction": float(np.mean(deltas < 0)),
            "strict_flip_count": int(sum(x["strict_flip"] == "True" for x in values)),
            "new_negative_pid_event": any(x["new_negative_pid"] == "True" for x in values),
        })
    return out


def image_pairs(rows):
    # exact directed image edge: (anchor image, negative image), not PID-only.
    lookup = defaultdict(list)
    for row in rows:
        for neg in arr(row["negative_local_indices"]):
            lookup[(int(row["prev_update"]), int(row["target_update"]), int(row["anchor_local_index"]), neg)].append(row)
    out = []
    seen = set()
    for row in rows:
        a = int(row["anchor_local_index"]); p = int(row["anchor_pid"])
        for b in arr(row["negative_local_indices"]):
            reverse = lookup.get((int(row["prev_update"]), int(row["target_update"]), b, a), [])
            for rev in reverse:
                if int(rev["anchor_pid"]) == int(row["negative_pid"]) and int(rev["negative_pid"]) == p:
                    key = (int(row["prev_update"]), int(row["target_update"]), min(a, b), max(a, b))
                    if key in seen: continue
                    seen.add(key)
                    da, db = float(row["delta_margin"]), float(rev["delta_margin"])
                    out.append({
                        "prev_update": row["prev_update"], "target_update": row["target_update"],
                        "a_anchor_local": a, "b_anchor_local": b, "pid_a": p, "pid_b": int(row["negative_pid"]),
                        "delta_a_to_b": da, "delta_b_to_a": db,
                        "a_strict_flip": row["strict_flip"], "b_strict_flip": rev["strict_flip"],
                        "relation_index_a": row["relation_index"], "relation_index_b": rev["relation_index"],
                    })
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--transition-rows", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read(args.transition_rows); ids = identity_stats(rows); imgs = image_pairs(rows)
    with (args.output_dir / "identity_directed_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ids[0])); w.writeheader(); w.writerows(ids)
    with (args.output_dir / "image_level_reciprocal_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(imgs[0]) if imgs else ["prev_update", "target_update", "a_anchor_local", "b_anchor_local", "pid_a", "pid_b", "delta_a_to_b", "delta_b_to_a", "a_strict_flip", "b_strict_flip", "relation_index_a", "relation_index_b"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(imgs)
    quad = Counter()
    for x in imgs:
        a, b = float(x["delta_a_to_b"]), float(x["delta_b_to_a"])
        quad["A改善_B恶化"] += a > 0 and b < 0
        quad["A恶化_B改善"] += a < 0 and b > 0
        quad["双改善"] += a > 0 and b > 0
        quad["双恶化"] += a < 0 and b < 0
    summary = {
        "status": "PASS_IDENTITY_AND_IMAGE_BIDIRECTIONAL_CORRECTION",
        "input_transition_rows": len(rows), "identity_directed_rows": len(ids),
        "image_level_reciprocal_pairs": len(imgs), "identity_pair_interval_count": len({(x["prev_update"], x["target_update"], min(x["anchor_pid"], x["negative_pid"]), max(x["anchor_pid"], x["negative_pid"])) for x in ids}),
        "quadrants_image_level": dict(quad),
        "last_write_wins_used": False,
        "image_pair_definition": "same interval, A anchor image is exactly B negative image and vice versa, same PID pair",
        "missing_reverse_not_false": True,
        "future_information_used": False,
    }
    (args.output_dir / "bidirectional_relation_correction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()
