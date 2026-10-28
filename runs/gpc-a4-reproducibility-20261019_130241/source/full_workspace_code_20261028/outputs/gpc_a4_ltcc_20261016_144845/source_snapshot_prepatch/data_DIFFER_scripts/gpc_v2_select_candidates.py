#!/usr/local/bin/python3
"""Freeze deterministic screen/two-seed decisions for GPC-v2."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def screen(output: Path, report: Path):
    rows = list(csv.DictReader((output / "dev_evaluation_screen.csv").open(encoding="utf-8")))
    baseline = next(row for row in rows if row["arm"] == "B0")
    base_r1, base_map = float(baseline["rank1"]), float(baseline["mAP"])
    candidates = [row for row in rows if row["arm"] != "B0" and float(row["rank1"]) > base_r1 and float(row["mAP"]) >= base_map - 0.01]
    candidates.sort(key=lambda row: (float(row["rank1"]), float(row["mAP"])), reverse=True)
    selected = [row["arm"] for row in candidates[:2]]
    write_json(output / "candidate_selection.json", {"baseline": baseline, "selected_arms": selected, "screen_rows": rows, "rule": "E30固定开发验证：Rank-1严格高于B0，mAP不低于B0超过1pp，最多两个"})
    lines = ["# GPC-v2 LTCC 候选筛选报告", "", "使用固定 PID 不重叠开发验证集；官方 query/gallery 未参与筛选。", "", "| 组别 | Rank-1 | mAP |", "|---|---:|---:|"]
    lines += [f"| {row['arm']} | {float(row['rank1'])*100:.4f}% | {float(row['mAP'])*100:.4f}% |" for row in rows]
    lines += ["", f"B0：Rank-1={base_r1*100:.4f}%，mAP={base_map*100:.4f}%。", f"双 seed 复核候选：{selected or '无'}。", "筛选规则在训练前冻结，不使用单个 epoch 最优值。"]
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected

def two_seed(output: Path, selected: list[str], report: Path):
    rows = list(csv.DictReader((output / "dev_evaluation_two_seed.csv").open(encoding="utf-8")))
    baseline = {int(row["seed"]): row for row in rows if row["arm"] == "B0"}
    passed = {}
    lines = ["# GPC-v2 LTCC 双 seed 复核报告", "", "| 组别 | seed | Rank-1 | mAP | 相对对应 B0 Rank-1 |", "|---|---:|---:|---:|---:|"]
    for arm in ["B0"] + selected:
        for row in [item for item in rows if item["arm"] == arm]:
            seed = int(row["seed"])
            delta = 0.0 if arm == "B0" else float(row["rank1"]) - float(baseline[seed]["rank1"])
            if arm != "B0":
                passed[arm] = passed.get(arm, True) and delta > 0 and float(row["mAP"]) >= float(baseline[seed]["mAP"]) - 0.01
            lines.append(f"| {arm} | {seed} | {float(row['rank1'])*100:.4f}% | {float(row['mAP'])*100:.4f}% | {'—' if arm == 'B0' else f'{delta*100:+.4f}pp'} |")
    good = [arm for arm in selected if passed.get(arm, False)]
    write_json(output / "two_seed_selection.json", {"passed_arms": good, "per_arm": passed, "rows": rows})
    lines += ["", f"两个 seed 均满足预注册条件的方案：{good or '无'}。", "若 seed 结论不一致，最终状态必须为 MIXED。"]
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return good

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("screen", "two_seed"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "screen":
        selected = screen(args.output, args.report)
    else:
        selected = two_seed(args.output, json.loads((args.output / "candidate_selection.json").read_text())["selected_arms"], args.report)
    print(json.dumps({"mode": args.mode, "selected": selected}, ensure_ascii=False))

if __name__ == "__main__":
    main()
