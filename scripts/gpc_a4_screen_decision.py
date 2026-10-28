#!/usr/local/bin/python3.10
"""Freeze the E30 decision for whether A4 merits paired E60 confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.evaluation.read_text(encoding="utf-8"))
    rows = data["results"]
    deltas = []
    for seed in (1234, 4321):
        baseline = rows[f"screen_B0_seed{seed}"]["metrics"]
        candidate = rows[f"screen_A4_seed{seed}"]["metrics"]
        control = rows[f"screen_R2_seed{seed}"]["metrics"]
        deltas.append({
            "seed": seed,
            "a4_delta_rank1": float(candidate["rank1"] - baseline["rank1"]),
            "a4_delta_mAP": float(candidate["mAP"] - baseline["mAP"]),
            "r2_delta_rank1": float(control["rank1"] - baseline["rank1"]),
            "r2_delta_mAP": float(control["mAP"] - baseline["mAP"]),
        })
    a4_positive_both = all(row["a4_delta_rank1"] > 0 and row["a4_delta_mAP"] >= -0.01 for row in deltas)
    payload = {
        "evaluation": str(args.evaluation),
        "per_seed": deltas,
        "a4_positive_rank1_and_mAP_both_seeds": a4_positive_both,
        "decision": "PROCEED_E60_A4_B0" if a4_positive_both else "STOP_NO_E60_PROMOTION",
        "interpretation": "E30 仅作为筛选门禁；即使通过也不构成稳定提升证据。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# GPC-A4 E30 复核门禁", "", "| seed | A4 ΔR1 | A4 ΔmAP | R2 ΔR1 | R2 ΔmAP |", "|---:|---:|---:|---:|---:|"]
    for row in deltas:
        lines.append(f"| {row['seed']} | {row['a4_delta_rank1']:+.4%} | {row['a4_delta_mAP']:+.4%} | {row['r2_delta_rank1']:+.4%} | {row['r2_delta_mAP']:+.4%} |")
    lines += ["", f"**E60 决策：`{payload['decision']}`**。", "", "E30 只用于决定是否继续配对 E60，不使用官方测试集。"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not a4_positive_both:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
