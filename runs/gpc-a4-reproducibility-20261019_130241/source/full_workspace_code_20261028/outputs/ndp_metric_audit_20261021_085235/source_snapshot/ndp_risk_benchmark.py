#!/usr/bin/env python3
"""P3：冻结终点之间的风险排序对照。

若输入不是同一训练轨迹的未来 checkpoint，本程序会明确把结果标为
endpoint-shift exploratory，禁止解释为在线风险预测性能。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _finite_metric_arrays(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return finite rows and explicitly count discarded non-finite scores.

    Risk metrics must not silently turn NaN/Inf into a favorable rank.  The
    benchmark records this count in its JSON; callers can therefore distinguish
    a valid metric from a metric computed after data loss.
    """
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    if y.shape != score.shape:
        raise ValueError("labels and scores shape mismatch")
    finite = np.isfinite(score)
    return y[finite], score[finite], int((~finite).sum())


def auc_rank(y: np.ndarray, score: np.ndarray) -> float | None:
    """Tie-correct ROC-AUC for larger risk scores meaning more positive.

    Ranks are assigned in ascending score order, with average ranks for ties.
    This is the Mann--Whitney definition and is equivalent to
    ``sklearn.metrics.roc_auc_score``.  The previous implementation sorted in
    descending order but used the ascending-rank formula, reversing the result.
    """
    y, score, _ = _finite_metric_arrays(y, score)
    pos = int(y.sum()); neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks_sorted = np.empty(len(order), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(len(order), dtype=np.float64); ranks[order] = ranks_sorted
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y, score, _ = _finite_metric_arrays(y, score)
    pos = int(y.sum())
    if pos == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ys = y[order].astype(np.float64)
    ss = score[order]
    ap = 0.0
    tp_before = 0.0
    start = 0
    while start < len(ss):
        end = start + 1
        while end < len(ss) and ss[end] == ss[start]:
            end += 1
        group_tp = float(ys[start:end].sum())
        if group_tp:
            tp_after = tp_before + group_tp
            ap += (group_tp / pos) * (tp_after / end)
            tp_before = tp_after
        start = end
    return float(ap)


def top_fraction_metrics(y: np.ndarray, score: np.ndarray, fraction: float = 0.10) -> dict:
    y, score, invalid = _finite_metric_arrays(y, score)
    if len(y) == 0:
        return {"top_fraction": fraction, "top_count": 0, "precision": None,
                "recall": None, "positive_count": 0, "invalid_score_count": invalid}
    count = max(1, int(np.ceil(len(y) * fraction)))
    chosen = np.argsort(-score, kind="stable")[:count]
    tp = int(y[chosen].sum())
    return {
        "top_fraction": fraction,
        "top_count": count,
        "precision": tp / count,
        "recall": tp / max(int(y.sum()), 1),
        "positive_count": int(y.sum()),
        "invalid_score_count": invalid,
    }


def bootstrap_pid(rows: list[dict], score_key: str, label_key: str, seed: int, rounds: int = 500) -> dict:
    pids = np.asarray(sorted({int(r["pid"]) for r in rows}), dtype=np.int64)
    by_pid = {pid: [r for r in rows if int(r["pid"]) == int(pid)] for pid in pids}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        subset = [r for pid in sampled for r in by_pid[int(pid)]]
        y = np.asarray([bool(r[label_key]) for r in subset])
        s = np.asarray([float(r[score_key]) for r in subset])
        ap = average_precision(y, s)
        if ap is not None:
            values.append(ap)
    if not values:
        return {"pid_count": len(pids), "rounds": rounds, "ci95": None}
    return {"pid_count": len(pids), "rounds": rounds, "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))], "mean": float(np.mean(values))}


def sklearn_cross_check(y: np.ndarray, score: np.ndarray) -> dict:
    """独立核对；不替换本地实现。"""
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception as exc:  # 环境没有 sklearn 时仍可运行本地指标
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    y2, s2, invalid = _finite_metric_arrays(y, score)
    if int(y2.sum()) == 0 or int((~y2).sum()) == 0:
        return {"available": True, "invalid_score_count": invalid,
                "roc_auc": None, "average_precision": None}
    return {
        "available": True,
        "invalid_score_count": invalid,
        "roc_auc": float(roc_auc_score(y2.astype(np.int64), s2)),
        "average_precision": float(average_precision_score(y2.astype(np.int64), s2)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrusion-summary", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--history-csv", type=Path, nargs=2, required=False)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()
    summary_paths = args.intrusion_summary
    row_paths = [Path(json.loads(p.read_text())["rows_csv"]) for p in summary_paths]
    rows = []
    for seed_index, path in enumerate(row_paths):
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for key in ("pid", "reverse_available_count"):
                    row[key] = int(row[key])
                for key in ("delta_i", "edge_drop", "reference_edge", "current_edge", "i_ref", "i_cur", "reverse_delta_i_mean"):
                    row[key] = float(row[key]) if row[key] not in ("", "None") else float("nan")
                row["edge_flip"] = row["edge_flip"].lower() == "true"
                row["seed"] = 1234 if seed_index == 0 else 4321
                rows.append(row)
    history_by_seed = {1234: {}, 4321: {}}
    if args.history_csv:
        for seed, p in zip((1234, 4321), args.history_csv):
            with p.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    history_by_seed[seed][int(row["anchor"])] = float(row.get("negative_jaccard", 0.0))
    for row in rows:
        stability = history_by_seed.get(int(row["seed"]), {}).get(int(row["anchor"]), 0.0)
        row["history_stability"] = stability
        row["hubness_penalty"] = 1.0  # replaced below where hubness is available from rows only
        # R0/R1 are retrospective diagnostics on the same fixed endpoint pair;
        # R2+ are soft intrusion variants. None is online-valid without a future trajectory.
        row["R0_hard_margin_drop"] = row["edge_drop"]
        row["R1_DBP"] = max(row["edge_drop"], 0.0)
        row["R2_soft_intrusion"] = row["delta_i"]
        row["R3_soft_plus_history"] = row["delta_i"] * stability
    # Hubness is derived from repeated representative appearances in this sample.
    counts_by_seed = {1234: {}, 4321: {}}
    for row in rows:
        counts = counts_by_seed[int(row["seed"])]
        for token in row["negative_representatives"].split(";"):
            counts[int(token)] = counts.get(int(token), 0) + 1
    for row in rows:
        counts = counts_by_seed[int(row["seed"])]
        hub = np.mean([counts.get(int(x), 0) for x in row["negative_representatives"].split(";")])
        row["hubness_mean"] = float(hub)
        row["R4_soft_hubness_control"] = row["delta_i"] / (1.0 + np.log1p(hub))
        if row["reverse_available_count"] > 0 and np.isfinite(row["reverse_delta_i_mean"]):
            row["R5_bidirectional"] = (row["delta_i"] + row["reverse_delta_i_mean"]) / 2.0
        else:
            row["R5_bidirectional"] = row["delta_i"]
    methods = {
        "R0_hard_margin_drop": "R0_hard_margin_drop",
        "R1_DBP": "R1_DBP",
        "R2_soft_intrusion": "R2_soft_intrusion",
        "R3_soft_plus_history": "R3_soft_plus_history",
        "R4_soft_hubness_control": "R4_soft_hubness_control",
        "R5_bidirectional": "R5_bidirectional",
    }
    result_rows = []
    for seed in sorted({int(r["seed"]) for r in rows}):
        seed_rows = [r for r in rows if int(r["seed"]) == seed]
        y = np.asarray([r["edge_flip"] for r in seed_rows])
        for name, key in methods.items():
            score = np.asarray([float(r[key]) for r in seed_rows])
            cross = sklearn_cross_check(y, score)
            own_auc = auc_rank(y, score)
            own_ap = average_precision(y, score)
            result_rows.append({
                "seed": seed,
                "method": name,
                "rows": len(seed_rows),
                "pid_count": len({int(r["pid"]) for r in seed_rows}),
                "event_count": int(y.sum()),
                "roc_auc": own_auc,
                "roc_auc_metric": "tie_correct_mann_whitney",
                "roc_auc_sklearn": cross.get("roc_auc"),
                "roc_auc_abs_diff": (abs(own_auc - cross["roc_auc"])
                                      if own_auc is not None and cross.get("roc_auc") is not None else None),
                "pr_auc": own_ap,
                "pr_auc_metric": "average_precision",
                "average_precision_sklearn": cross.get("average_precision"),
                "average_precision_abs_diff": (abs(own_ap - cross["average_precision"])
                                                if own_ap is not None and cross.get("average_precision") is not None else None),
                "invalid_score_count": int(cross.get("invalid_score_count", 0)),
                "top10": top_fraction_metrics(y, score),
                "pid_bootstrap_ap": bootstrap_pid(seed_rows, key, "edge_flip", 20261021 + seed + len(name), args.bootstrap),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps({
        "status": "PASS",
        "methods": list(methods),
        "event_definition": "reference selected boundary > 0 and fixed-current selected boundary <= 0",
        "metric_definitions": {
            "roc_auc": "tie-correct Mann-Whitney AUC; larger score means greater event risk",
            "pr_auc": "average precision, not trapezoidal PR integral",
            "top10": "top ceil(10% of finite rows), descending score order",
        },
        "online_predictive_valid": False,
        "validity_warning": "当前输入是独立 Bio/GPC E60 endpoint shift，不是同一训练轨迹的未来 checkpoint；结果只作离线诊断，不能证明在线预测。",
        "rows": result_rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "benchmark_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["seed", "method", "rows", "pid_count", "event_count", "roc_auc",
                  "roc_auc_metric", "roc_auc_sklearn", "roc_auc_abs_diff", "pr_auc",
                  "pr_auc_metric", "average_precision_sklearn", "average_precision_abs_diff",
                  "invalid_score_count", "top10", "pid_bootstrap_ap"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(result_rows)
    # 保留逐 anchor 的轻量风险分数，供人工复核。
    with (args.output_dir / "anchor_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["seed", "anchor", "pid", "edge_flip"] + list(methods)
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({"seed": row["seed"], "anchor": row["anchor"], "pid": row["pid"], "edge_flip": row["edge_flip"], **{k: row[v] for k, v in methods.items()}})
    print(json.dumps({"status": "PASS", "rows": len(result_rows), "online_predictive_valid": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
