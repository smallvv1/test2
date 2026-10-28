#!/usr/bin/env python3
"""固定同轨迹关系的历史可靠性分层。

可靠性只使用当前节点及其之前的节点；后一个节点只作为离线评估标签，绝不参与
当时的类别划分。关系候选来自预先冻结的 groups，不在未来重建邻域。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def classify_reliability(history: np.ndarray, stable_min_margin: float, max_history_std: float,
                         min_history_nodes: int) -> str:
    """只使用历史前缀判断可靠性；节点不足时不伪称稳定。"""
    history = np.asarray(history, dtype=np.float64)
    if len(history) < min_history_nodes:
        return "INSUFFICIENT_HISTORY"
    if bool(np.all(history > 0)) and float(history.min()) >= stable_min_margin and float(history.std()) <= max_history_std:
        return "A_STABLE_CORRECT"
    if bool(np.all(history > 0)):
        return "B_CORRECT_NEAR_OR_VARIABLE"
    return "C_HISTORY_NOT_RELIABLE"


def classify_vulnerability(history: np.ndarray, vulnerability_margin: float,
                           vulnerability_std: float) -> str:
    """把当前边界脆弱性作为独立维度，不把它并入 reliability。"""
    history = np.asarray(history, dtype=np.float64)
    if len(history) < 2:
        return "VULNERABILITY_UNKNOWN"
    recent = history[-min(3, len(history)):]
    sign_flip = bool(np.any((recent[:-1] > 0) != (recent[1:] > 0)))
    high = float(history[-1]) <= vulnerability_margin or float(recent.std()) > vulnerability_std or sign_flip
    return "HIGH_VULNERABILITY" if high else "LOW_VULNERABILITY"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stable-min-margin", type=float, default=0.05)
    ap.add_argument("--max-history-std", type=float, default=0.05)
    ap.add_argument("--min-history-nodes", type=int, default=3)
    ap.add_argument("--vulnerability-margin", type=float, default=0.05)
    ap.add_argument("--vulnerability-std", type=float, default=0.05)
    args = ap.parse_args()
    if args.min_history_nodes < 2:
        raise ValueError("min-history-nodes 至少为2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank = np.load(args.features, allow_pickle=False)
    features = bank["features"].astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
    updates = [int(x) for x in bank["updates"]]
    index_to_pos = {int(x): i for i, x in enumerate(bank["image_index"])}
    pid = {int(i): int(p) for i, p in zip(bank["image_index"], bank["pid"])}
    groups_data = json.loads(args.groups.read_text(encoding="utf-8"))
    groups = groups_data["groups"]
    if groups_data.get("groups_sha256") and groups_data["groups_sha256"] != digest(groups):
        raise RuntimeError("groups_sha256 校验失败")
    relation_defs = []
    for g in groups:
        anchor = int(g["anchor"])
        neg_by_pid = {}
        for n in g["negatives"]:
            n = int(n)
            if n in index_to_pos and pid[n] != pid[anchor]:
                neg_by_pid.setdefault(pid[n], []).append(n)
        for neg_pid, negs in sorted(neg_by_pid.items()):
            relation_defs.append({"anchor": anchor, "anchor_pid": pid[anchor], "negative_pid": int(neg_pid),
                                  "positive_indices": [int(x) for x in g["positives"] if int(x) in index_to_pos],
                                  "negative_indices": negs})
    rows = []
    # 先得到所有节点的固定关系 margin，再逐节点只切历史前缀。
    margin = {}
    for r, rel in enumerate(relation_defs):
        ai = index_to_pos[rel["anchor"]]
        pi = [index_to_pos[x] for x in rel["positive_indices"]]
        ni = [index_to_pos[x] for x in rel["negative_indices"]]
        if not pi or not ni:
            continue
        margin[r] = []
        for node in range(len(updates)):
            z = features[node]
            u = float(np.quantile(z[ai] @ z[pi].T, 0.25))
            v = float(np.max(z[ai] @ z[ni].T))
            margin[r].append(u - v)
    for r, rel in enumerate(relation_defs):
        if r not in margin:
            continue
        vals = margin[r]
        for node, update in enumerate(updates):
            hist = np.asarray(vals[: node + 1], dtype=np.float64)
            cls = classify_reliability(hist, args.stable_min_margin, args.max_history_std, args.min_history_nodes)
            vulnerability = classify_vulnerability(hist, args.vulnerability_margin, args.vulnerability_std)
            next_update = updates[node + 1] if node + 1 < len(updates) else None
            next_margin = vals[node + 1] if node + 1 < len(updates) else None
            rows.append({
                "anchor": rel["anchor"], "anchor_pid": rel["anchor_pid"], "negative_pid": rel["negative_pid"],
                "update": update, "history_end_index": node, "m_current": vals[node],
                "history_correct_rate": float(np.mean(hist > 0)), "history_min_margin": float(hist.min()),
                "history_margin_std": float(hist.std()), "reliability_class": cls,
                "vulnerability_class": vulnerability,
                "reliability_vulnerability_cell": f"{cls}__{vulnerability}",
                "next_update": next_update, "next_margin": next_margin,
                "next_new_error": bool(next_update is not None and vals[node] > 0 and vals[node + 1] <= 0),
                "relation_key": f"{rel['anchor']}:{rel['negative_pid']}",
            })
    out_csv = args.output_dir / "historical_relation_rows.csv"
    fields = list(rows[0]) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = {"status": "PASS_HISTORY_PREFIX_ONLY" if rows else "NO_VALID_RELATIONS",
               "rows": len(rows), "relation_count": len({r["relation_key"] for r in rows}),
               "anchor_count": len({r["anchor"] for r in rows}), "updates": updates,
               "classes": {c: sum(r["reliability_class"] == c for r in rows)
                           for c in ("A_STABLE_CORRECT", "B_CORRECT_NEAR_OR_VARIABLE", "C_HISTORY_NOT_RELIABLE", "INSUFFICIENT_HISTORY")},
               "vulnerability_classes": {c: sum(r["vulnerability_class"] == c for r in rows)
                                         for c in ("HIGH_VULNERABILITY", "LOW_VULNERABILITY", "VULNERABILITY_UNKNOWN")},
               "reliability_vulnerability_cells": {},
               "event_by_update_and_class": {},
               "stable_min_margin": args.stable_min_margin, "max_history_std": args.max_history_std,
               "min_history_nodes": args.min_history_nodes, "vulnerability_margin": args.vulnerability_margin,
               "vulnerability_std": args.vulnerability_std,
               "features": str(args.features), "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
               "groups": str(args.groups), "groups_sha256": digest(groups),
               "future_labels_used_only_for_evaluation": True, "official_query_gallery_used": False}
    for update in updates[:-1]:
        summary["event_by_update_and_class"][str(update)] = {}
        for cls in ("A_STABLE_CORRECT", "B_CORRECT_NEAR_OR_VARIABLE", "C_HISTORY_NOT_RELIABLE", "INSUFFICIENT_HISTORY"):
            subset = [r for r in rows if r["update"] == update and r["reliability_class"] == cls and r["next_update"] is not None]
            summary["event_by_update_and_class"][str(update)][cls] = {
                "n": len(subset), "new_error": sum(r["next_new_error"] for r in subset),
                "rate": (sum(r["next_new_error"] for r in subset) / len(subset)) if subset else None,
            }
    for r in rows:
        cell = r["reliability_vulnerability_cell"]
        summary["reliability_vulnerability_cells"][cell] = summary["reliability_vulnerability_cells"].get(cell, 0) + 1
    (args.output_dir / "historical_relation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
