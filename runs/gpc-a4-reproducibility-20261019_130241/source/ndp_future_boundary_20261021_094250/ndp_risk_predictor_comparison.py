#!/usr/bin/env python3
"""同轨迹未来真正 boundary flip 的无参数风险基线比较。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ndp_risk_benchmark import auc_rank, average_precision, top_fraction_metrics


def read_rows(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("seed", "anchor", "pid", "feature_update", "target_update"):
                r[k] = int(r[k])
            for k in ("risk_prev_drop", "risk_low_prev_margin", "risk_recent_slope"):
                r[k] = float(r[k])
            r["y_flip"] = r["y_flip"].lower() == "true"
            r["prev_correct"] = r["prev_correct"].lower() == "true"
            r["negative_indices"] = [int(x) for x in r["negative_indices"].split(";") if x]
            out.append(r)
    return out


def metric(rows: list[dict], score_key: str, label_key: str = "y_flip") -> dict:
    y = np.asarray([r[label_key] for r in rows], dtype=bool)
    s = np.asarray([r[score_key] for r in rows], dtype=np.float64)
    return {"rows": len(rows), "positive_count": int(y.sum()), "event_rate": float(y.mean()),
            "roc_auc": auc_rank(y, s), "average_precision": average_precision(y, s),
            "top10": top_fraction_metrics(y, s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--future-rows", type=Path, nargs=2, required=True)
    ap.add_argument("--bidirectional-rows", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    all_out = []; summary = []
    for path in args.future_rows:
        rows = read_rows(path); seed = rows[0]["seed"]
        hub_counts = {}
        for r in rows:
            for n in r["negative_indices"]:
                hub_counts[n] = hub_counts.get(n, 0) + 1
        for r in rows:
            r["risk_static_hubness"] = float(np.mean([hub_counts[n] for n in r["negative_indices"]])) if r["negative_indices"] else 0.0
            all_out.append(r)
        sr = [r for r in all_out if r["seed"] == seed]
        # 严格翻转标签只在前一节点 M_t>0 的关系上定义；此前错误的关系
        # 不能进入真正的“正确->错误”预测分母。
        flip_rows = [r for r in sr if r["prev_correct"]]
        summary.append({
            "seed": seed, "rows": len(flip_rows), "all_transition_rows": len(sr), "pid_count": len({r["pid"] for r in flip_rows}),
            "event": "next-transition true boundary flip; previous margin > 0",
            "available_methods": {
                "B0_current_hard_margin": metric(flip_rows, "risk_low_prev_margin"),
                "B2_previous_margin_drop": metric(flip_rows, "risk_prev_drop"),
                "B3_history_drop_slope": metric(flip_rows, "risk_recent_slope"),
                "B7_static_sampled_hubness": metric(flip_rows, "risk_static_hubness"),
            },
            "not_run": {
                "B1_nearest_negative_similarity": "缺少每个 continuation 节点的完整特征/负样本相似度",
                "B4_temporal_history_stability": "已有历史 Jaccard 来自独立终点，不具在线时间有效性",
                "B5_high_impostor_tail": "缺少每个 continuation 节点的全局负 PID 尾部特征",
                "B6_bidirectional": "见独立 identity-pair 结果，不与单向 anchor 行直接混合",
            },
        })
    fields = list(all_out[0]) if all_out else []
    with (args.output_dir / "predictor_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(all_out)
    bi_summary = []
    for path in args.bidirectional_rows:
        rows=[]
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["seed"]=int(r["seed"]); r["pid_a"]=int(r["pid_a"]); r["pid_b"]=int(r["pid_b"])
                r["score_b6"] = max(-float(r["m_a_prev"]), -float(r["m_b_prev"]))
                r["y_pair_flip"] = r["a_flip"].lower()=="true" or r["b_flip"].lower()=="true"
                rows.append(r)
        y=np.asarray([r["y_pair_flip"] for r in rows],dtype=bool); s=np.asarray([r["score_b6"] for r in rows],dtype=float)
        bi_summary.append({"seed":rows[0]["seed"] if rows else None,"rows":len(rows),"identity_pair_count":len({r["identity_pair"] for r in rows}),"B6_bidirectional_boundary_margin": {"positive_count":int(y.sum()),"event_rate":float(y.mean()) if len(y) else None,"roc_auc":auc_rank(y,s),"average_precision":average_precision(y,s),"top10":top_fraction_metrics(y,s)}})
    (args.output_dir / "risk_predictor_summary.json").write_text(json.dumps({
        "status":"PASS_WITH_LIMITED_FEATURES", "rows":summary, "B6":bi_summary,
        "future_leakage_control":"all scores use previous node/fixed relation; target only labels",
        "official_query_gallery_used":False, "optimizer_step_executed":False,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS_WITH_LIMITED_FEATURES","rows":summary,"B6":bi_summary},ensure_ascii=False))


if __name__ == "__main__":
    main()
