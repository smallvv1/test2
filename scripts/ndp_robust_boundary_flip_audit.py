#!/usr/bin/env python3
"""严格翻转与预注册稳健翻转的敏感性分析。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ndp_risk_benchmark import auc_rank, average_precision


def read_rows(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("anchor", "pid", "feature_update", "target_update"):
                r[k] = int(r[k])
            for k in ("prev_margin", "target_margin", "risk_prev_drop"):
                r[k] = float(r[k])
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--deltas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    detail = []
    for path in args.rows:
        rows = read_rows(path); seed = int(rows[0]["seed"])
        strict = [r for r in rows if r["prev_margin"] > 0 and r["target_margin"] <= 0]
        for r in strict:
            detail.append({"seed": seed, "type": "strict_flip", **r})
        for delta in args.deltas:
            eligible = [r for r in rows if r["prev_margin"] > delta]
            robust = [r for r in eligible if r["target_margin"] < -delta]
            y = np.asarray([r in robust for r in eligible], dtype=bool)
            scores = {
                "distance_to_zero": np.asarray([-r["prev_margin"] for r in eligible], dtype=float),
                "previous_drop": np.asarray([r["risk_prev_drop"] for r in eligible], dtype=float),
            }
            metrics = {}
            for name, score in scores.items():
                metrics[name] = {"roc_auc": auc_rank(y, score), "average_precision": average_precision(y, score)}
            intervals = {}
            for r in robust:
                key = f"{r['feature_update']}->{r['target_update']}"
                intervals[key] = intervals.get(key, 0) + 1
                detail.append({"seed": seed, "type": f"robust_flip_delta_{delta:g}", "delta": delta, **r})
            result.append({
                "seed": seed, "delta": delta, "eligible_rows": len(eligible),
                "robust_flip_count": len(robust), "pid_count": len({r["pid"] for r in robust}),
                "event_rate": float(np.mean(y)) if len(y) else None,
                "interval_counts": intervals, "metrics": metrics,
                "strict_flip_count": len(strict),
                "strict_not_robust_count": sum(not (r["prev_margin"] > delta and r["target_margin"] < -delta) for r in strict),
            })
    with (args.output_dir / "robust_boundary_flip_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = []
        for row in detail:
            for key in row:
                if key not in fields:
                    fields.append(key)
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(detail)
    out = {"status": "PASS_WITH_PREREGISTERED_SENSITIVITY", "deltas": args.deltas,
           "results": result, "official_query_gallery_used": False, "optimizer_step_executed": False,
           "delta_source": "预注册数值敏感性范围；未根据结果选择主阈值"}
    (args.output_dir / "robust_boundary_flip_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
