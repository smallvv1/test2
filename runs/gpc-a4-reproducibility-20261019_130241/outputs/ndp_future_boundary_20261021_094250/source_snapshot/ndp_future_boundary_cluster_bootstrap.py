#!/usr/bin/env python3
"""对同轨迹未来风险指标做 PID cluster bootstrap；不执行模型计算。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ndp_risk_benchmark import auc_rank, average_precision


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["pid"] = int(r["pid"])
            for k in ("prev_drop", "risk_prev_drop", "risk_low_prev_margin", "delta_drop"):
                r[k] = float(r[k])
            for k in ("y_onset", "y_increment", "y_flip", "prev_correct"):
                r[k] = r[k].lower() == "true"
            rows.append(r)
    return rows


def cluster_bootstrap(rows: list[dict], label: str, score: str,
                      repeats: int, seed: int) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault(r["pid"], []).append(r)
    pids = sorted(groups)
    y = np.asarray([r[label] for r in rows], dtype=bool)
    s = np.asarray([r[score] for r in rows], dtype=float)
    point = {"rows": len(rows), "pid_count": len(pids), "positive_count": int(y.sum()),
             "event_rate": float(y.mean()) if len(y) else None,
             "roc_auc": auc_rank(y, s), "average_precision": average_precision(y, s)}
    rng = np.random.default_rng(seed)
    aucs, aps, rates = [], [], []
    for _ in range(repeats):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        sample = [r for pid in sampled for r in groups[int(pid)]]
        yy = np.asarray([r[label] for r in sample], dtype=bool)
        ss = np.asarray([r[score] for r in sample], dtype=float)
        rates.append(float(yy.mean()))
        if yy.any() and (~yy).any():
            a = auc_rank(yy, ss); ap = average_precision(yy, ss)
            if a is not None and ap is not None:
                aucs.append(float(a)); aps.append(float(ap))

    def ci(values):
        return {"valid_replicates": len(values), "q025": float(np.quantile(values, .025)),
                "q975": float(np.quantile(values, .975))} if values else {"valid_replicates": 0}

    point["bootstrap"] = {"repeats": repeats, "seed": seed, "event_rate_ci": ci(rates),
                           "roc_auc_ci": ci(aucs), "average_precision_ci": ci(aps)}
    return point


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs=2, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2000)
    args = ap.parse_args()
    out = []
    for path in args.rows:
        rows = read_rows(path); seed = int(rows[0]["seed"])
        specs = [
            ("onset_prev_unactive", [r for r in rows if r["prev_drop"] <= .02], "y_onset", "risk_prev_drop"),
            ("increment_all", rows, "y_increment", "risk_prev_drop"),
            ("increment_excluding_0_to_20", [r for r in rows if int(r["feature_update"]) > 0], "y_increment", "risk_prev_drop"),
            ("flip_prev_correct", [r for r in rows if r["prev_correct"]], "y_flip", "risk_low_prev_margin"),
        ]
        for name, subset, label, score in specs:
            result = cluster_bootstrap(subset, label, score, args.repeats, 20261021 + seed + len(name))
            result.update({"seed": seed, "analysis": name, "label": label, "score": score})
            out.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS_OFFLINE_PID_CLUSTER_BOOTSTRAP", "rows": out}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_OFFLINE_PID_CLUSTER_BOOTSTRAP", "count": len(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
