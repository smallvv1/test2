#!/usr/local/bin/python3.10
"""A4/R2 paired PID bootstrap for the new corrected-selector runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/home/DIFFER/scripts")
from gpc_v2_paired_pid_bootstrap import BOOTSTRAP_REPEATS, BOOTSTRAP_SEED, load_json, paired, result_row  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--pairs", nargs="+", required=True, help="baseline:candidate:label")
    args = parser.parse_args()

    evaluation = load_json(args.evaluation)
    results = evaluation["results"]
    model_rows = [result_row(label, results[label]) for label in sorted({part for spec in args.pairs for part in spec.split(":")[:2]})]
    comparisons = []
    for index, spec in enumerate(args.pairs):
        baseline, candidate, label = spec.split(":", 2)
        comparisons.append(paired(results[baseline], results[candidate], label, index))

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "model_metrics.csv", model_rows)
    write_csv(args.output / "paired_pid_deltas.csv", [row for item in comparisons for row in item["pid_rows"]])
    write_csv(args.output / "paired_query_deltas.csv", [row for item in comparisons for row in item["query_rows"]])
    payload = {
        "evaluation": str(args.evaluation),
        "evaluation_sha256": sha256(args.evaluation),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "query_count": evaluation["query_count"],
        "gallery_count": evaluation["gallery_count"],
        "effective_pid_count": evaluation["effective_pid_count"],
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "model_metrics": model_rows,
        "paired_comparisons": comparisons,
        "official_query_gallery_used": False,
        "cpu_model_fallback": False,
    }
    (args.output / "paired_pid_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# GPC-A4 配对 PID Bootstrap 报告",
        "",
        f"- Evaluation：`{args.evaluation}`；SHA256：`{sha256(args.evaluation)}`。",
        f"- Query：`{evaluation['query_count']}`；Gallery：`{evaluation['gallery_count']}`；有效 PID：`{evaluation['effective_pid_count']}`。",
        f"- Bootstrap：{BOOTSTRAP_REPEATS} 次，PID 为重采样单位；同一比较复用相同抽样索引。",
        "- 结果仅用于冻结开发集探索性复核，不使用官方 LTCC query/gallery。",
        "",
        "## 模型结果",
        "",
        "| 模型 | seed | R1 | R5 | R10 | mAP | PID宏R1 | PID宏mAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(f"| {row['label']} | {row['seed']} | {row['rank1']:.4%} | {row['rank5']:.4%} | {row['rank10']:.4%} | {row['mAP']:.4%} | {row['pid_macro_rank1']:.4%} | {row['pid_macro_mAP']:.4%} |")
    lines += ["", "## 配对结果", "", "| 比较 | PID | Query | ΔR1 | R1 CI95 | ΔmAP | mAP CI95 | Query R1 胜/平/负 |", "|---|---:|---:|---:|---|---:|---|---|"]
    for item in comparisons:
        boot = item["pid_bootstrap"]
        wins = item["query_pair_counts"]
        lines.append(f"| {item['pair']} | {item['common_pid_count']} | {item['common_query_count']} | {boot['delta_rank1_mean']:+.4%} | [{boot['delta_rank1_ci95'][0]:+.4%}, {boot['delta_rank1_ci95'][1]:+.4%}] | {boot['delta_mAP_mean']:+.4%} | [{boot['delta_mAP_ci95'][0]:+.4%}, {boot['delta_mAP_ci95'][1]:+.4%}] | {wins['rank1_wins']}/{wins['rank1_ties']}/{wins['rank1_losses']} |")
    lines += [
        "",
        "逐 PID：`paired_pid_deltas.csv`；逐 Query：`paired_query_deltas.csv`。",
        "当前只有 11 个有效验证 PID，CI 跨 0 时只能报告探索性正向信号，不能宣称稳定提升。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "report": str(args.report), "comparisons": [item["pair"] for item in comparisons]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
