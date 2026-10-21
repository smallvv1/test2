#!/usr/bin/env python3
"""同一 Bio+GPC continuation 的无更新风险时序审计。

输入是阶段3在 PPU eval/no_grad 下产生的固定64组逐节点优势值；不重新训练、
不使用未来节点选择关系。用前一节点的可用状态预测下一节点的激活标签，
并明确这是小覆盖的 continuation 诊断，不是全训练集风险模型。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ndp_risk_benchmark import auc_rank, average_precision


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("seed", "update", "anchor", "pid", "positive_count", "negative_count"):
                r[k] = int(r[k])
            for k in ("reference_advantage", "current_advantage", "advantage_drop", "epsilon"):
                r[k] = float(r[k])
            r["active"] = r["active"].lower() == "true"
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summary = []
    for path in args.rows:
        rows = load_rows(path)
        seed = rows[0]["seed"]
        by_anchor: dict[int, list[dict]] = {}
        for r in rows:
            by_anchor.setdefault(r["anchor"], []).append(r)
        for values in by_anchor.values():
            values.sort(key=lambda x: x["update"])
            for target_i in range(1, len(values)):
                target = values[target_i]
                prev = values[target_i - 1]
                prevprev = values[target_i - 2] if target_i >= 2 else None
                prev_drop = prev["advantage_drop"]
                prior_drops = [x["advantage_drop"] for x in values[:target_i]]
                slope = (prev_drop - prevprev["advantage_drop"]) if prevprev else 0.0
                row = {
                    "seed": seed,
                    "anchor": target["anchor"],
                    "pid": target["pid"],
                    "feature_update": prev["update"],
                    "target_update": target["update"],
                    "prev_current_advantage": prev["current_advantage"],
                    "prev_drop": prev_drop,
                    "prev_drop_slope": slope,
                    "max_prior_drop": max(prior_drops),
                    "target_drop": target["advantage_drop"],
                    "target_active": target["active"],
                    "relation_sha256": target["relation_sha256"],
                }
                all_rows.append(row)
        labels = np.asarray([r["target_active"] for r in all_rows if r["seed"] == seed])
        # 每个 target transition 计算三种只依赖过去的简单分数。
        seed_rows = [r for r in all_rows if r["seed"] == seed]
        metrics = []
        for name in ("prev_drop", "max_prior_drop", "prev_drop_slope"):
            score = np.asarray([float(r[name]) for r in seed_rows])
            metrics.append({"method": name, "rows": len(seed_rows), "positive_count": int(labels.sum()),
                            "roc_auc": auc_rank(labels, score), "average_precision": average_precision(labels, score)})
        summary.append({
            "seed": seed,
            "rows": len(seed_rows),
            "anchor_count": len(by_anchor),
            "feature_updates": sorted({r["feature_update"] for r in seed_rows}),
            "target_updates": sorted({r["target_update"] for r in seed_rows}),
            "target_active_count": int(labels.sum()),
            "target_active_rate": float(labels.mean()) if len(labels) else None,
            "metrics": metrics,
            "state_source": str(path),
        })
    fields = list(all_rows[0]) if all_rows else []
    with (args.output_dir / "same_trajectory_prediction_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    (args.output_dir / "same_trajectory_summary.json").write_text(json.dumps({
        "status": "PASS",
        "scope": "固定64组；同一Bio+GPC continuation；PPU eval/no_grad阶段3产物",
        "optimizer_step_executed_by_this_script": False,
        "official_query_gallery_used": False,
        "future_leakage_control": "特征只取 target_update 之前节点；target 节点仅生成评价标签",
        "coverage_warning": "不是9576张训练图像的全库预测评估，不能外推为全局在线风险模型",
        "rows": summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(all_rows), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
