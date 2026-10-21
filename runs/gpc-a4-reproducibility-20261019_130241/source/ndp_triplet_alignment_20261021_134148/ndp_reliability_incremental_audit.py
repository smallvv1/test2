#!/usr/bin/env python3
"""只在 update200 仍正确的关系中审计首次翻转风险及历史信息增量。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def auc(scores, labels):
    p = [float(s) for s, y in zip(scores, labels) if y]
    n = [float(s) for s, y in zip(scores, labels) if not y]
    if not p or not n: return None
    return (sum(x > y for x in p for y in n) + 0.5 * sum(x == y for x in p for y in n)) / (len(p) * len(n))


def ap(scores, labels):
    order = sorted(range(len(labels)), key=lambda i: (-float(scores[i]), i))
    positives = sum(labels)
    if not positives: return None
    hits = 0; total = 0.0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            hits += 1; total += hits / rank
    return total / positives


def bootstrap_rate(rows, repeats=1000, seed=20261021):
    by_pid = defaultdict(list)
    for r in rows: by_pid[int(r["anchor_pid"])].append(int(r["label"]))
    pids = sorted(by_pid); rng = np.random.default_rng(seed); vals=[]
    for _ in range(repeats):
        sample = rng.choice(pids, size=len(pids), replace=True)
        den = sum(len(by_pid[int(p)]) for p in sample); num = sum(sum(by_pid[int(p)]) for p in sample)
        vals.append(num / den if den else np.nan)
    return {"pid_count": len(pids), "mean": float(np.nanmean(vals)), "ci95_low": float(np.nanpercentile(vals, 2.5)), "ci95_high": float(np.nanpercentile(vals, 97.5))}


def matched_current_margin_audit(rows, repeats=200, seed=20261021):
    """对当前 Margin 做近邻匹配，描述历史变量是否还有额外排序信息。

    只使用 update200 的 prev_margin 进行匹配；update400 仅提供标签。
    每次为每个事件随机抽取其当前 Margin 最近的非事件控制，避免单次
    随机 mask 造成结论依赖。该结果是描述性控制，不是拟合模型。
    """
    events = [i for i, r in enumerate(rows) if int(r["label"])]
    controls = [i for i, r in enumerate(rows) if not int(r["label"])]
    if not events or len(controls) < len(events):
        return {"status": "NOT_AVAILABLE", "reason": "insufficient_controls"}
    rng = np.random.default_rng(seed)
    event_margin = np.asarray([float(rows[i]["prev_margin"]) for i in events])
    control_margin = np.asarray([float(rows[i]["prev_margin"]) for i in controls])
    nearest = []
    for value in event_margin:
        order = np.argsort(np.abs(control_margin - value), kind="stable")
        nearest.append(order[:min(20, len(order))])

    def local_auc(values, labels):
        pos = [float(v) for v, y in zip(values, labels) if y]
        neg = [float(v) for v, y in zip(values, labels) if not y]
        if not pos or not neg:
            return np.nan
        return (sum(x > y for x in pos for y in neg) + 0.5 * sum(x == y for x in pos for y in neg)) / (len(pos) * len(neg))

    keys = ("history_instability", "history_min_margin_risk", "history_decline_slope", "unreliability")
    values = {key: [] for key in keys}
    current_values = []
    for _ in range(repeats):
        chosen = []
        used = set()
        for choices in nearest:
            available = [int(j) for j in choices if int(j) not in used]
            if not available:
                available = [j for j in range(len(controls)) if j not in used]
            pick = int(rng.choice(available))
            used.add(pick)
            chosen.append(controls[pick])
        subset = events + chosen
        labels = [1] * len(events) + [0] * len(chosen)
        for key in keys:
            if key == "history_instability": score = [float(rows[i]["history_std"]) for i in subset]
            elif key == "history_min_margin_risk": score = [-float(rows[i]["history_min_margin"]) for i in subset]
            elif key == "history_decline_slope": score = [-float(rows[i]["history_slope"]) for i in subset]
            else: score = [1.0 - float(rows[i]["history_correct_rate"]) for i in subset]
            values[key].append(local_auc(score, labels))
        current_values.append(local_auc([-float(rows[i]["prev_margin"]) for i in subset], labels))
    return {
        "status": "PASS_DESCRIPTIVE_MATCHED_CONTROL",
        "repeats": repeats,
        "event_count": len(events),
        "control_count_per_repeat": len(chosen),
        "current_margin_auc_mean": float(np.nanmean(current_values)),
        "history_auc": {key: {"mean": float(np.nanmean(vals)), "ci95_low": float(np.nanpercentile(vals, 2.5)), "ci95_high": float(np.nanpercentile(vals, 97.5))} for key, vals in values.items()},
        "uses_future_for_score": False,
    }


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--transition-rows", type=Path, required=True); parser.add_argument("--reliability-rows", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--seed", type=int, required=True); args=parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    trans=read(args.transition_rows); rel={int(r["relation_index"]):r for r in read(args.reliability_rows)}
    hist=defaultdict(list)
    for r in trans:
        if int(r["target_update"]) <= 200: hist[int(r["relation_index"])].append((int(r["target_update"]), float(r["target_margin"])))
    hist_pre200=defaultdict(list)
    for r in trans:
        if int(r["target_update"]) < 200: hist_pre200[int(r["relation_index"])].append((int(r["target_update"]), float(r["target_margin"])))
    rows=[]
    for r in trans:
        if int(r["prev_update"]) != 200 or int(r["target_update"]) != 400: continue
        # 关键分母：update200 当前正确；已错误关系不参与首次翻转预测。
        prev=float(r["prev_margin"]); target=float(r["target_margin"])
        if not prev > 0: continue
        rr=rel[int(r["relation_index"])]
        series=sorted(hist[int(r["relation_index"])])
        pre_series=sorted(hist_pre200[int(r["relation_index"])])
        if len(series)>=2:
            x=np.asarray([u for u,_ in series],dtype=float); y=np.asarray([m for _,m in series],dtype=float); slope=float(np.polyfit(x,y,1)[0])
        else: slope=0.0
        if len(pre_series)>=2:
            px=np.asarray([u for u,_ in pre_series],dtype=float); py=np.asarray([m for _,m in pre_series],dtype=float); pre_slope=float(np.polyfit(px,py,1)[0])
        else: pre_slope=0.0
        rows.append({
            "relation_index":r["relation_index"],"anchor_pid":r["anchor_pid"],"negative_pid":r["negative_pid"],"prev_margin":prev,"target_margin":target,
            "label":int(target<=0),"robust_1e-3":int(prev>1e-3 and target<-1e-3),"robust_1e-2":int(prev>1e-2 and target<-1e-2),
            "current_margin_score":-prev,"history_correct_rate":float(rr["history_correct_rate"]),"history_std":float(rr["history_std"]),"history_min_margin":float(rr["history_min_margin"]),"history_slope":slope,
            "pre200_history_correct_rate":sum(m>0 for _,m in pre_series)/len(pre_series) if pre_series else float("nan"),
            "pre200_history_std":float(np.std([m for _,m in pre_series])) if pre_series else float("nan"),
            "pre200_history_min_margin":min((m for _,m in pre_series), default=float("nan")),
            "pre200_history_slope":pre_slope,
            "reliability":rr["reliability"],"vulnerability":rr["vulnerability"],"target_correct":target>0,
        })
    if not rows: raise RuntimeError("no update200-correct rows")
    with (args.output_dir/"reliability_incremental_rows.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    labels=[r["label"] for r in rows]
    scores={
        "current_margin": [r["current_margin_score"] for r in rows],
        "history_instability": [r["history_std"] for r in rows],
        "history_min_margin_risk": [-r["history_min_margin"] for r in rows],
        "history_decline_slope": [-r["history_slope"] for r in rows],
        "unreliability": [1.0-r["history_correct_rate"] for r in rows],
        "vulnerability_label": [float(r["vulnerability"]=="HIGH_VULNERABILITY") for r in rows],
        "pre200_history_instability": [r["pre200_history_std"] for r in rows],
        "pre200_history_min_margin_risk": [-r["pre200_history_min_margin"] for r in rows],
        "pre200_history_decline_slope": [-r["pre200_history_slope"] for r in rows],
        "pre200_unreliability": [1.0-r["pre200_history_correct_rate"] for r in rows],
    }
    metrics={k:{"auc":auc(v,labels),"average_precision":ap(v,labels)} for k,v in scores.items()}
    bins=defaultdict(list)
    for r in rows:
        b="<=0.05" if r["prev_margin"]<=.05 else "0.05-0.10" if r["prev_margin"]<=.10 else "0.10-0.20" if r["prev_margin"]<=.20 else ">0.20"; bins[b].append(r)
    strata={k:{"n":len(v),"events":sum(x["label"] for x in v),"event_rate":sum(x["label"] for x in v)/len(v),"bootstrap":bootstrap_rate(v,seed=args.seed)} for k,v in bins.items()}
    summary={"status":"PASS_FIRST_FLIP_RISK_DENOMINATOR_CORRECTION","seed":args.seed,"candidate_definition":"prev_margin > 0 only","eligible_count":len(rows),"strict_flip_count":sum(labels),"strict_flip_rate":sum(labels)/len(labels),"robust_counts":{"delta_1e-3":sum(r["robust_1e-3"] for r in rows),"delta_1e-2":sum(r["robust_1e-2"] for r in rows)},"metrics":metrics,"margin_strata":strata,"matched_current_margin_control":matched_current_margin_audit(rows,seed=20261021+args.seed),"pid_count":len({r["anchor_pid"] for r in rows}),"future_information_used_for_scores":False,"official_query_gallery_used":False,"optimizer_step_executed":False,"interpretation":"描述性回顾审计；未宣称未来风险预测 PASS"}
    (args.output_dir/"reliability_incremental_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__": main()
