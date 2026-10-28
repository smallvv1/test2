#!/usr/bin/env python3
"""固定关系上新增错误 PID 的可辨识性审计。

只有一个正样本和一个异 PID 负样本时，阶段3的 log-mean-exp advantage
才可作为该单一图像关系的符号证据。多图像或多 PID 聚合关系不拆分。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("seed", "anchor", "pid", "feature_update", "target_update", "negative_pid_count"):
                r[k] = int(r[k])
            r["negative_pid_decomposable"] = r["negative_pid_decomposable"].lower() == "true"
            r["prev_margin"] = float(r["prev_margin"]); r["target_margin"] = float(r["target_margin"])
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []; summaries = []
    for path in args.rows:
        rows = read_rows(path); seed = rows[0]["seed"]
        exact = [r for r in rows if r["negative_pid_decomposable"]]
        status_counts = {}
        for r in exact:
            status = ("new_error_pid" if r["prev_margin"] > 0 and r["target_margin"] <= 0 else
                      "preexisting_error_pid" if r["prev_margin"] <= 0 and r["target_margin"] <= 0 else
                      "recovered_pid" if r["prev_margin"] <= 0 and r["target_margin"] > 0 else
                      "correct_pid")
            status_counts[status] = status_counts.get(status, 0) + 1
            all_rows.append({**r, "pid_event_state": status})
        summaries.append({
            "seed": seed,
            "groups": len({r["anchor"] for r in rows}),
            "transition_rows": len(rows),
            "exact_single_image_pair_groups": len({r["anchor"] for r in exact}),
            "nondecomposable_groups": len({r["anchor"] for r in rows if not r["negative_pid_decomposable"]}),
            "exact_transition_rows": len(exact),
            "status_counts": status_counts,
            "multi_pid_aggregate_rows": sum(r["negative_pid_count"] > 1 for r in rows),
            "interpretation": "仅 exact_single_image_pair 可报告单一负 PID 的状态；其它行不可由 aggregate advantage 拆分",
        })
    fields = list(all_rows[0]) if all_rows else []
    with (args.output_dir / "new_negative_pid_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    out = {"status": "PASS_WITH_EXPLICIT_NOT_AVAILABLE", "rows": summaries,
           "official_query_gallery_used": False, "optimizer_step_executed": False,
           "multi_pid_decomposition": "NOT_AVAILABLE_WITH_CURRENT_STAGE3_CSV"}
    (args.output_dir / "new_negative_pid_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
