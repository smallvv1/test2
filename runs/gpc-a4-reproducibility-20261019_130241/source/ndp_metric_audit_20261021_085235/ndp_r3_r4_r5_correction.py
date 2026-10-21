#!/usr/bin/env python3
"""R3/R4/R5 结构与评估口径修正。

R3 明确标记为独立终点间历史代理，不能称为时间历史；R4 按 seed 隔离
hubness；R5 只在可找到反向图像方向时建立显式双向记录，并按身份对去重。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path, seed: int) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["seed"] = seed
            r["anchor"] = int(r["anchor"]); r["pid"] = int(r["pid"])
            r["delta_i"] = float(r["delta_i"])
            r["edge_flip"] = r["edge_flip"].lower() == "true"
            r["negative_representatives"] = [int(x) for x in r["negative_representatives"].split(";") if x]
            r["negative_pids"] = [int(x) for x in r["negative_pids"].split(";") if x]
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs=2, required=True)
    ap.add_argument("--history", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    pair_rows = []
    for seed, row_path, hist_path in zip((1234, 4321), args.rows, args.history):
        rows = read_rows(row_path, seed)
        by_anchor = {r["anchor"]: r for r in rows}
        counts: dict[int, int] = {}
        for r in rows:
            for rep in r["negative_representatives"]:
                counts[rep] = counts.get(rep, 0) + 1
        for r in rows:
            hub = float(np.mean([counts[x] for x in r["negative_representatives"]])) if r["negative_representatives"] else 0.0
            r["sampled_hubness_mean_seed_local"] = hub
            r["sampled_hubness_max_seed_local"] = max(counts.values()) if counts else 0
            all_rows.append(r)
            for rep in r["negative_representatives"]:
                other = by_anchor.get(rep)
                if other is None or r["pid"] not in other["negative_pids"]:
                    continue
                pair_key = tuple(sorted((int(r["pid"]), int(other["pid"]))))
                pair_rows.append({
                    "seed": seed,
                    "identity_pair": f"{pair_key[0]}-{pair_key[1]}",
                    "anchor_a": r["anchor"], "anchor_b": other["anchor"],
                    "pid_a": r["pid"], "pid_b": other["pid"],
                    "delta_a_to_b": r["delta_i"], "delta_b_to_a": other["delta_i"],
                    "a_edge_flip": r["edge_flip"], "b_edge_flip": other["edge_flip"],
                    "one_side_improved_other_worsened": bool((r["delta_i"] < 0) != (other["delta_i"] < 0)),
                })
        history_rows = list(csv.DictReader(hist_path.open(encoding="utf-8")))
        summaries.append({
            "seed": seed,
            "rows": len(rows),
            "sampled_anchor_hubness_max": max(counts.values()) if counts else 0,
            "sampled_anchor_hubness_nonzero": len(counts),
            "sampled_anchor_hubness_top10_sum": int(sum(sorted(counts.values(), reverse=True)[:10])),
            "edge_flip_count": sum(r["edge_flip"] for r in rows),
            "explicit_bidirectional_image_pairs": sum(1 for r in pair_rows if r["seed"] == seed),
            "one_side_improve_other_worsen_count": sum(r["one_side_improved_other_worsened"] for r in pair_rows if r["seed"] == seed),
            "history_source_rows": len(history_rows),
            "history_temporal_valid": False,
            "history_warning": "Bio E60/GPC E60 independent endpoints; not same-trajectory history",
        })
    row_fields = [k for k in all_rows[0] if k != "negative_representatives" and k != "negative_pids"]
    with (args.output_dir / "r4_seed_local_hubness_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=row_fields); writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in row_fields})
    pair_fields = list(pair_rows[0]) if pair_rows else ["seed", "identity_pair"]
    with (args.output_dir / "r5_bidirectional_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=pair_fields); writer.writeheader(); writer.writerows(pair_rows)
    (args.output_dir / "r3_r4_r5_summary.json").write_text(json.dumps({
        "status": "PASS_WITH_LIMITATIONS",
        "R3": {"status": "NOT_ONLINE_VALID", "reason": "history Jaccard is between independent Bio/GPC endpoints"},
        "R4": {"status": "SEED_ISOLATED_SAMPLE_HUBNESS", "warning": "sampled-anchor hubness, not full-train hubness"},
        "R5": {"status": "EXPLICIT_REVERSE_IMAGE_DIRECTION_WHEN_AVAILABLE", "deduplication": "identity-pair key is reported; multiple image directions retained for audit"},
        "rows": summaries,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_WITH_LIMITATIONS", "summaries": summaries, "bidirectional_rows": len(pair_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
