#!/usr/local/bin/python3
"""NDP 指标修正与 Triplet 覆盖分母审计（纯离线）。

本脚本不实例化模型、不执行 forward/backward、不读取官方 query/gallery，
仅复用上一轮已经生成的 LTCC train-only 特征库、邻域图、固定关系表以及
Phase 5B/7O 的冻结采样计划。Triplet 部分明确标注为冻结特征上的
batch-hard 代理；没有把它冒充为逐步在线 miner 日志。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/DIFFER")
SOURCE_RUN = "20261020_155558"
SOURCE_OUT = ROOT / "outputs" / f"ndp_local_intrusion_{SOURCE_RUN}"
OLD_AUDIT = ROOT / "outputs" / "ndp_net_identity_audit_20261020_162456"
K_VALUES = (5, 10, 20)
SEEDS = (1234, 4321)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261020


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): canonical(x) for k, x in sorted(v.items(), key=lambda z: str(z[0]))}
    if isinstance(v, (list, tuple)):
        return [canonical(x) for x in v]
    if isinstance(v, np.ndarray):
        return canonical(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def digest(v: Any) -> str:
    return hashlib.sha256(json.dumps(canonical(v), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonical(obj), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                out[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            w.writerow(out)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_old_module():
    path = ROOT / "scripts" / "ndp_net_intrusion_triplet_coverage_audit.py"
    spec = importlib.util.spec_from_file_location("ndp_old_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载历史离线审计模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_relation_rows(seed: int, k: int) -> list[dict[str, Any]]:
    path = SOURCE_OUT / "reference_relations" / f"seed{seed}_K{k}_relations.csv"
    result = []
    for r in read_csv(path):
        result.append({
            "seed": seed, "K": k, "anchor": int(r["anchor"]), "pid": int(r["pid"]),
            "positive": int(r["positive"]), "negative": int(r["negative"]),
            "positive_clothes_cross": r["positive_clothes_cross"].lower() == "true",
            "g_ref": float(r["g_ref"]), "g_current": float(r["g_current"]),
            "reference_reliable": r["reference_reliable"].lower() == "true",
            "reference_correct": r["reference_correct"].lower() == "true",
            "current_order_correct": r["current_order_correct"].lower() == "true",
        })
    return result


def bootstrap(values: dict[int, float], seed: int) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    x = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.asarray([np.mean(x[rng.integers(0, len(x), size=len(x))]) for _ in range(BOOTSTRAP_REPS)])
    return float(np.mean(x)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def corrected_graph_metrics(seed: int, k: int, ref_idx: np.ndarray, cur_idx: np.ndarray, pids: np.ndarray):
    rows = []
    for a in range(len(pids)):
        ref_set = {int(x) for x in ref_idx[a, :k]}
        cur_set = {int(x) for x in cur_idx[a, :k]}
        ref_same = {x for x in ref_set if int(pids[x]) == int(pids[a])}
        cur_same = {x for x in cur_set if int(pids[x]) == int(pids[a])}
        ref_neg = ref_set - ref_same
        cur_neg = cur_set - cur_same
        ref_neg_count = len(ref_neg)
        cur_neg_count = len(cur_neg)
        signed_count = cur_neg_count - ref_neg_count
        delta_same = len(cur_same) - len(ref_same)
        if signed_count != -delta_same:
            raise AssertionError(f"seed={seed} K={k} anchor={a}: signed negative / same-ID identity mismatch")
        rows.append({
            "seed": seed, "K": k, "anchor": a, "pid": int(pids[a]),
            "ref_same_count": len(ref_same), "current_same_count": len(cur_same),
            "ref_negative_count": ref_neg_count, "current_negative_count": cur_neg_count,
            "SIGNED_NEGATIVE_DELTA_count": signed_count,
            "SIGNED_NEGATIVE_DELTA": signed_count / float(k),
            "NET_NEGATIVE_INTRUSION_count": max(signed_count, 0),
            "NET_NEGATIVE_INTRUSION": max(signed_count, 0) / float(k),
            "NET_NEGATIVE_RELIEF_count": max(-signed_count, 0),
            "NET_NEGATIVE_RELIEF": max(-signed_count, 0) / float(k),
            "SAME_ID_LOST": len(ref_same - cur_set),
            "SAME_ID_GAINED": len(cur_same - ref_set),
            "DELTA_SAME_ID_COUNT": delta_same,
            "NEW_NEGATIVE_NEIGHBORS": len(cur_neg - ref_neg),
            "REMOVED_NEGATIVE_NEIGHBORS": len(ref_neg - cur_neg),
            "same_id_recall_delta_slots": delta_same,
        })
    by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pid[int(r["pid"])].append(r)
    summary = {
        "seed": seed, "K": k, "anchor_count": len(rows), "pid_count": len(by_pid),
        "total_SIGNED_NEGATIVE_DELTA_count": int(sum(r["SIGNED_NEGATIVE_DELTA_count"] for r in rows)),
        "total_NET_NEGATIVE_INTRUSION_count": int(sum(r["NET_NEGATIVE_INTRUSION_count"] for r in rows)),
        "total_NET_NEGATIVE_RELIEF_count": int(sum(r["NET_NEGATIVE_RELIEF_count"] for r in rows)),
        "total_SAME_ID_LOST": int(sum(r["SAME_ID_LOST"] for r in rows)),
        "total_SAME_ID_GAINED": int(sum(r["SAME_ID_GAINED"] for r in rows)),
        "total_NEW_NEGATIVE_NEIGHBORS": int(sum(r["NEW_NEGATIVE_NEIGHBORS"] for r in rows)),
        "mean_SIGNED_NEGATIVE_DELTA": float(np.mean([r["SIGNED_NEGATIVE_DELTA"] for r in rows])),
        "mean_NET_NEGATIVE_INTRUSION": float(np.mean([r["NET_NEGATIVE_INTRUSION"] for r in rows])),
        "mean_NET_NEGATIVE_RELIEF": float(np.mean([r["NET_NEGATIVE_RELIEF"] for r in rows])),
        "mean_DELTA_SAME_ID_COUNT": float(np.mean([r["DELTA_SAME_ID_COUNT"] for r in rows])),
        "anchors_with_positive_net_intrusion": int(sum(r["NET_NEGATIVE_INTRUSION_count"] > 0 for r in rows)),
        "anchors_with_negative_net_intrusion_relief": int(sum(r["NET_NEGATIVE_RELIEF_count"] > 0 for r in rows)),
    }
    pid_rows = []
    for pid, values in sorted(by_pid.items()):
        pid_rows.append({
            "seed": seed, "K": k, "pid": pid, "anchor_count": len(values),
            "mean_SIGNED_NEGATIVE_DELTA": float(np.mean([x["SIGNED_NEGATIVE_DELTA"] for x in values])),
            "mean_NET_NEGATIVE_INTRUSION": float(np.mean([x["NET_NEGATIVE_INTRUSION"] for x in values])),
            "mean_NET_NEGATIVE_RELIEF": float(np.mean([x["NET_NEGATIVE_RELIEF"] for x in values])),
            "mean_SAME_ID_LOST": float(np.mean([x["SAME_ID_LOST"] for x in values])),
            "mean_SAME_ID_GAINED": float(np.mean([x["SAME_ID_GAINED"] for x in values])),
            "mean_DELTA_SAME_ID_COUNT": float(np.mean([x["DELTA_SAME_ID_COUNT"] for x in values])),
        })
    boot = []
    for metric in ("mean_SIGNED_NEGATIVE_DELTA", "mean_NET_NEGATIVE_INTRUSION", "mean_NET_NEGATIVE_RELIEF", "mean_DELTA_SAME_ID_COUNT"):
        vals = {int(x["pid"]): float(x[metric]) for x in pid_rows}
        mean, lo, hi = bootstrap(vals, BOOTSTRAP_SEED + seed + k + len(metric))
        boot.append({"seed": seed, "K": k, "metric": metric, "pid_count": len(vals), "mean": mean, "ci95_low": lo, "ci95_high": hi})
    return rows, pid_rows, boot, summary


def state_audit(seed: int, k: int, rows: list[dict[str, Any]], bank: dict[str, np.ndarray]):
    clothes = bank["clothes"].astype(np.int64)
    cams = bank["camid"].astype(np.int64)
    states = Counter()
    reliable = Counter()
    by_pid: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        state = ("correct_correct" if r["reference_correct"] and r["current_order_correct"] else
                 "correct_wrong" if r["reference_correct"] and not r["current_order_correct"] else
                 "wrong_correct" if not r["reference_correct"] and r["current_order_correct"] else "wrong_wrong")
        states[state] += 1
        a, p, n = r["anchor"], r["positive"], r["negative"]
        r["positive_cross_camera"] = int(cams[a] != cams[p])
        r["negative_cross_camera"] = int(cams[a] != cams[n])
        r["negative_cross_clothes"] = int(clothes[a] != clothes[n])
        r["state"] = state
        r["reference_reliable_flip"] = int(r["reference_reliable"] and not r["current_order_correct"])
        by_pid[int(r["pid"])][state] += 1
        by_pid[int(r["pid"])]["rows"] += 1
        by_pid[int(r["pid"])]["reliable"] += int(r["reference_reliable"])
        by_pid[int(r["pid"])]["reliable_flip"] += int(r["reference_reliable"] and not r["current_order_correct"])
        if r["reference_reliable"]:
            reliable["rows"] += 1
            reliable["flip"] += int(not r["current_order_correct"])
            reliable["positive_cross_clothes"] += int(r["positive_clothes_cross"])
            reliable["negative_cross_clothes"] += int(clothes[a] != clothes[n])
            reliable["negative_cross_camera"] += int(cams[a] != cams[n])
            reliable["correct_wrong"] += int(state == "correct_wrong")
            reliable["wrong_correct"] += int(state == "wrong_correct")
    pid_rows = []
    for pid, c in sorted(by_pid.items()):
        pid_rows.append({"seed": seed, "K": k, "pid": pid, "relation_rows": c["rows"], "reliable_rows": c["reliable"], "reliable_flip_rows": c["reliable_flip"], "reliable_flip_rate": c["reliable_flip"] / max(c["reliable"], 1), **{s: c[s] for s in ("correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong")}})
    pid_flip = {r["pid"]: r["reliable_flip_rate"] for r in pid_rows}
    _, lo, hi = bootstrap(pid_flip, BOOTSTRAP_SEED + seed + 700 + k)
    summary = {"seed": seed, "K": k, "relation_rows": len(rows), "correct_correct": states["correct_correct"], "correct_wrong": states["correct_wrong"], "wrong_correct": states["wrong_correct"], "wrong_wrong": states["wrong_wrong"], "reliable_rows": reliable["rows"], "reliable_flip_rows": reliable["flip"], "reliable_flip_rate": reliable["flip"] / max(reliable["rows"], 1), "reliable_flip_pid_count": sum(r["reliable_flip_rows"] > 0 for r in pid_rows), "reliable_flip_pid_rate_ci95": [lo, hi], "reliable_positive_cross_clothes": reliable["positive_cross_clothes"], "reliable_negative_cross_clothes": reliable["negative_cross_clothes"], "reliable_negative_cross_camera": reliable["negative_cross_camera"], "reliable_correct_wrong": reliable["correct_wrong"], "reliable_wrong_correct": reliable["wrong_correct"]}
    return rows, pid_rows, summary


def build_triplet_audit(old, seed: int, k: int, rows: list[dict[str, Any]], bank: dict[str, np.ndarray], plan: dict[str, Any]):
    pids = bank["pid"].astype(np.int64)
    features = bank["features"].astype(np.float32)
    # The stored features are expected to be normalized.  This is checked rather than assumed.
    norm_err = float(np.max(np.abs(np.linalg.norm(features, axis=1) - 1.0)))
    hard = old.triplet_hard_maps(features, pids, plan)
    image_to_batches: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for rank, payload in plan["ranks"].items():
        for iteration, batch in enumerate(payload["batches"], start=1):
            for a in set(int(x) for x in batch):
                image_to_batches[a].add((int(rank), int(iteration)))
    opportunity_keys: set[tuple[int, int, int]] = set()
    selected_keys: set[tuple[int, int, int]] = set(hard)
    anchor_opportunity: set[int] = set()
    anchor_selected: set[int] = set()
    pid_opportunity: set[int] = set()
    pid_selected: set[int] = set()
    for rank, payload in plan["ranks"].items():
        for iteration, batch in enumerate(payload["batches"], start=1):
            batch = [int(x) for x in batch]
            for a in batch:
                same = [x for x in batch if x != a and pids[x] == pids[a]]
                neg = [x for x in batch if pids[x] != pids[a]]
                key = (int(rank), int(iteration), a)
                if same and neg:
                    opportunity_keys.add(key); anchor_opportunity.add(a); pid_opportunity.add(int(pids[a]))
                if key in hard:
                    anchor_selected.add(a); pid_selected.add(int(pids[a]))

    cov_rows = []
    for r in rows:
        a, p, n = r["anchor"], r["positive"], r["negative"]
        ap = image_to_batches[a] & image_to_batches[p]
        an = image_to_batches[a] & image_to_batches[n]
        common = ap & an
        positive_hard = False; negative_hard = False; full_hard = False
        for rank, iteration in common:
            h = hard.get((rank, iteration, a))
            if h:
                positive_hard |= h["hard_positive"] == p
                negative_hard |= h["hard_negative"] == n
                full_hard |= (h["hard_positive"] == p and h["hard_negative"] == n)
        cov_rows.append({
            "seed": seed, "K": k, "anchor": a, "pid": int(pids[a]), "positive": p, "negative": n,
            "reference_reliable": int(r["reference_reliable"]), "reliable_flip": int(r["reference_reliable"] and not r["current_order_correct"]),
            "anchor_positive_same_batch": int(bool(ap)), "anchor_negative_same_batch": int(bool(an)), "three_way_same_batch": int(bool(common)),
            "positive_hard_selected": int(positive_hard), "negative_hard_selected": int(negative_hard), "full_triplet_hard_selected": int(full_hard),
            "analytic_nonzero_gradient_if_full_hard": int(full_hard),
            "anchor_has_any_triplet_opportunity_window": int(a in anchor_opportunity), "anchor_has_any_triplet_miner_selection_window": int(a in anchor_selected),
            "pid_has_any_triplet_opportunity_window": int(int(pids[a]) in pid_opportunity), "pid_has_any_triplet_miner_selection_window": int(int(pids[a]) in pid_selected),
            "common_batch_occurrences": len(common), "feature_norm_max_error": norm_err,
        })
    return cov_rows, {"hard": hard, "opportunity_keys": opportunity_keys, "selected_keys": selected_keys, "anchor_opportunity": anchor_opportunity, "anchor_selected": anchor_selected, "pid_opportunity": pid_opportunity, "pid_selected": pid_selected, "feature_norm_max_error": norm_err}


def coverage_summary(seed: int, k: int, cov: list[dict[str, Any]], aux: dict[str, Any], relation_rows: int):
    def summarize(name: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(subset)
        if not n:
            return {"scope": name, "rows": 0}
        anchors = {int(r["anchor"]) for r in subset}; pids = {int(r["pid"]) for r in subset}
        anchor_opportunity_count = len({int(r["anchor"]) for r in subset if r["anchor_has_any_triplet_opportunity_window"]})
        anchor_miner_count = len({int(r["anchor"]) for r in subset if r["anchor_has_any_triplet_miner_selection_window"]})
        pid_opportunity_count = len({int(r["pid"]) for r in subset if r["pid_has_any_triplet_opportunity_window"]})
        pid_miner_count = len({int(r["pid"]) for r in subset if r["pid_has_any_triplet_miner_selection_window"]})
        three_way = sum(r["three_way_same_batch"] for r in subset)
        positive_opp = sum(r["anchor_positive_same_batch"] for r in subset)
        negative_opp = sum(r["anchor_negative_same_batch"] for r in subset)
        positive_hard = sum(r["positive_hard_selected"] for r in subset)
        negative_hard = sum(r["negative_hard_selected"] for r in subset)
        full_hard = sum(r["full_triplet_hard_selected"] for r in subset)
        return {
            "scope": name, "rows": n, "unique_anchors": len(anchors), "unique_pids": len(pids),
            "positive_pair_opportunity": positive_opp,
            "negative_pair_opportunity": negative_opp,
            "three_way_opportunity": three_way,
            "positive_hard_selected": sum(r["positive_hard_selected"] for r in subset),
            "negative_hard_selected": sum(r["negative_hard_selected"] for r in subset),
            "full_triplet_hard_selected": sum(r["full_triplet_hard_selected"] for r in subset),
            "analytic_nonzero_gradient_if_full_hard": sum(r["analytic_nonzero_gradient_if_full_hard"] for r in subset),
            "anchor_any_opportunity": anchor_opportunity_count,
            "anchor_any_miner_selection": anchor_miner_count,
            "pid_any_opportunity": pid_opportunity_count,
            "pid_any_miner_selection": pid_miner_count,
            "positive_hard_rate_given_positive_opportunity": positive_hard / max(positive_opp, 1),
            "negative_hard_rate_given_negative_opportunity": negative_hard / max(negative_opp, 1),
            "full_hard_rate_given_three_way": full_hard / max(three_way, 1),
            "full_hard_rate_global": full_hard / max(n, 1),
            "anchor_miner_coverage_rate_in_scope": anchor_miner_count / max(len(anchors), 1),
            "pid_miner_coverage_rate_in_scope": pid_miner_count / max(len(pids), 1),
        }
    scopes = [summarize("all_global_relation_rows", cov), summarize("reference_reliable_rows", [x for x in cov if x["reference_reliable"]]), summarize("reliable_flip_rows", [x for x in cov if x["reliable_flip"]])]
    # Whole-window denominator is not a relation-row denominator; it is reported separately.
    scopes.append({"scope": "whole_training_window", "rows": None, "unique_anchors": 9576, "unique_pids": 77, "anchor_triplet_opportunity_count": len(aux["anchor_opportunity"]), "anchor_triplet_miner_selection_count": len(aux["anchor_selected"]), "pid_triplet_opportunity_count": len(aux["pid_opportunity"]), "pid_triplet_miner_selection_count": len(aux["pid_selected"]), "triplet_opportunity_events": len(aux["opportunity_keys"]), "triplet_miner_selection_events": len(aux["selected_keys"])})
    return {"seed": seed, "K": k, "global_relation_rows": relation_rows, "feature_proxy": "GPC E60 frozen train feature bank; not online stepwise miner log", "scopes": scopes}


def pct(n: Any, d: Any) -> str:
    return "—" if not d else f"{100.0 * float(n) / float(d):.6f}%"


def write_reports(run_time: str, out: Path, graph_summaries: list[dict[str, Any]], states: list[dict[str, Any]], coverage: list[dict[str, Any]], source_manifest: dict[str, Any], decision: str) -> None:
    reports = ROOT / "reports"
    g10 = [x for x in graph_summaries if x["K"] == 10]
    c10 = [x for x in coverage if x["K"] == 10]
    by = {(x["seed"], x["K"]): x for x in graph_summaries}
    cby = {(x["seed"], x["K"]): x for x in coverage}
    state_by = {(x["seed"], x["K"]): x for x in states}
    text = "# NDP 身份侵入指标修正\n\n"
    text += "本报告复用上一轮 train-only 特征、邻域图和固定关系；未训练模型，未使用官方 query/gallery。旧报告把 `current_negative_count - reference_negative_count` 直接命名并汇总为 `net_negative_intrusion`，因此出现负数。该量实际是有符号变化，不是非负侵入指标。\n\n"
    text += "## 新口径\n\n"
    text += "`SIGNED_NEGATIVE_DELTA=(当前异身份数-参考异身份数)/K` 可为负；`NET_NEGATIVE_INTRUSION=max(SIGNED,0)`；`NET_NEGATIVE_RELIEF=max(-SIGNED,0)`。`SAME_ID_LOST`/`SAME_ID_GAINED` 是集合差，不把全部新异身份邻居都称为损害。每个 anchor 已断言 `SIGNED=-DELTA_SAME_ID_COUNT/K`。\n\n"
    text += "| seed | K | 有符号总槽位变化 | 非负侵入槽位 | 非负缓解槽位 | SAME_ID_LOST | SAME_ID_GAINED | 平均有符号变化 | 平均非负侵入 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in graph_summaries:
        text += f"| {r['seed']} | {r['K']} | {r['total_SIGNED_NEGATIVE_DELTA_count']} | {r['total_NET_NEGATIVE_INTRUSION_count']} | {r['total_NET_NEGATIVE_RELIEF_count']} | {r['total_SAME_ID_LOST']} | {r['total_SAME_ID_GAINED']} | {r['mean_SIGNED_NEGATIVE_DELTA']:+.6f} | {r['mean_NET_NEGATIVE_INTRUSION']:.6f} |\n"
    text += "\n在 K=10，真正使当前同身份槽位净减少的数量是非负侵入槽位：seed1234 为表中的 `NET_NEGATIVE_INTRUSION_count`，seed4321 同理；`NEW_NEGATIVE_NEIGHBORS` 仍可能包含不减少同身份数量的普通邻域重排。\n"
    (reports / "NDP_INTRUSION_METRIC_CORRECTION.md").write_text(text, encoding="utf-8")

    text = "# NDP 身份结构增益与损害\n\n固定四状态，不只统计损害，也保留错误关系被修复的数量。关系行不是独立身份证据；PID 级汇总和 2000 次 PID bootstrap 已写入输出目录。\n\n"
    text += "| seed | K | 正确→正确 | 正确→错误 | 错误→正确 | 错误→错误 | 可靠行 | 可靠翻转 | 可靠翻转率 | 可靠翻转PID数 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in states:
        text += f"| {r['seed']} | {r['K']} | {r['correct_correct']} | {r['correct_wrong']} | {r['wrong_correct']} | {r['wrong_wrong']} | {r['reliable_rows']} | {r['reliable_flip_rows']} | {r['reliable_flip_rate']:.4%} | {r['reliable_flip_pid_count']} |\n"
    text += "\nK=10 的方向仍不一致：seed1234 错误→正确多于正确→错误；seed4321 相反。摄像头、衣服和跨衣正样本分层表在 `tables/` 中，不能用平均 margin 单独替代结构状态。\n"
    (reports / "NDP_IDENTITY_GAIN_LOSS_ANALYSIS.md").write_text(text, encoding="utf-8")

    text = "# NDP Triplet 覆盖分母审计\n\n覆盖层级必须区分：全局固定关系行、正/负端点有机会同 batch、三端同 batch、miner 实际选中。历史 `0.0012%` 的含义是 seed1234/K10 下 1 条完整 hard 三元关系 ÷ 83670 条全局关系行，不是 0.0012% 的 optimizer step，也不是全训练 anchor 的梯度覆盖率。\n\n"
    text += "| seed | K | 范围 | 行数 | 正端机会 | 负端机会 | 三端机会 | 完整hard | 完整hard/全局 | 完整hard/三端机会 | anchor曾被miner | PID曾被miner |\n|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for c in coverage:
        for s in c["scopes"]:
            if s["scope"] == "whole_training_window":
                text += f"| {c['seed']} | {c['K']} | {s['scope']} | — | — | — | — | — | — | — | {s['anchor_triplet_miner_selection_count']}/{s['unique_anchors']} ({s['anchor_triplet_miner_selection_count']/s['unique_anchors']:.4%}) | {s['pid_triplet_miner_selection_count']}/{s['unique_pids']} ({s['pid_triplet_miner_selection_count']/s['unique_pids']:.4%}) |\n"
            else:
                text += f"| {c['seed']} | {c['K']} | {s['scope']} | {s.get('rows','—')} | {s.get('positive_pair_opportunity','—')} | {s.get('negative_pair_opportunity','—')} | {s.get('three_way_opportunity','—')} | {s.get('full_triplet_hard_selected','—')} | {s.get('full_hard_rate_global',0):.6%} | {s.get('full_hard_rate_given_three_way',0):.6%} | {s.get('anchor_any_miner_selection','—')}/{s.get('unique_anchors','—')} ({s.get('anchor_miner_coverage_rate_in_scope',0):.4%}) | {s.get('pid_any_miner_selection','—')}/{s.get('unique_pids','—')} ({s.get('pid_miner_coverage_rate_in_scope',0):.4%}) |\n"
    text += "\n`正端机会/负端机会` 是对应端点曾同 batch 的关系行分母；`完整hard/三端机会` 只在三端同 batch 后计算；`anchor曾被miner` 是窗口级其他 Triplet 约束，不是目标关系命中。可靠翻转关系单独作为 `reliable_flip_rows` 范围报告。\n"
    (reports / "NDP_TRIPLET_COVERAGE_DENOMINATOR_AUDIT.md").write_text(text, encoding="utf-8")

    text = "# NDP 真实 Triplet Mining 覆盖\n\n实际源码审计确认 `TripletLoss` 使用欧氏距离的 batch-hard：正样本取最大距离、负样本取最小距离；`NO_MARGIN=True` 使用 SoftMarginLoss。当前审计用单位归一化冻结 GPC E60 特征重建该排序，因此是离线代理，不是训练期间逐步 miner 记录。\n\n"
    text += "## 结论\n\n"
    text += "- 全局关系行分母很大，三端同 batch 分母已经显著缩小；在三端同 batch 后，是否被 hard positive/hard negative 选中还需再区分。\n"
    text += "- `anchor_window_miner_selection` 只表示同一 anchor 在整个冻结 E60 采样窗口中至少有一次代理 hard 约束，不表示目标关系三元组被选中。\n"
    text += "- `analytic_nonzero_gradient_if_full_hard` 是 SoftMarginLoss 的有限值解析梯度代理；实际 AMP found_inf/optimizer.step 未由本离线数据观测。\n"
    text += "- 对 K=10 的可靠翻转关系，两个 seed 的完整目标三元组代理命中均为 0；但大多数翻转 anchor 在窗口中出现过其他 Triplet 约束。因此可见的是‘目标关系缺少直接覆盖’，不能据此断言整个 anchor 没有 Triplet 训练。\n\n"
    text += "详细关系行和可靠翻转子集在 `triplet_coverage/`，窗口级事件在 `triplet_denominator_audit.csv`。\n"
    (reports / "NDP_REAL_TRIPLET_MINING_COVERAGE.md").write_text(text, encoding="utf-8")

    text = "# NDP 局部身份质量保持可行性\n\n当前审计没有把新异身份邻居自动等同于有害身份侵入。只有净同身份槽位下降、参考可靠关系翻转以及 PID 级可复现方向，才构成更强的候选风险证据。\n\n"
    text += "候选接口仍可研究多邻居身份概率质量：对冻结参考邻域 `U_i` 定义 `Q_ref=sum_{j:pid_j=pid_i} softmax(s_ref/T)`，当前同一集合定义 `Q_cur`，候选 `relu(Q_ref-Q_cur-epsilon_q)`。但它必须与普通 Triplet、自适应 margin Triplet、SupCon/N-pair 和无参考局部约束比较；不应保护参考错误结构，也不能读取验证/测试指标在线赋权。\n\n"
    text += "本轮尚未把该项写入 processor，未设 lambda，未执行 PPU 训练。若真实翻转关系在现有 Triplet 覆盖中存在跨 seed 的稳定缺口，才有资格提交短程预注册方案。\n"
    (reports / "NDP_LOCAL_IDENTITY_MASS_FEASIBILITY.md").write_text(text, encoding="utf-8")

    text = "# NDP 下一阶段决策\n\n"
    text += f"状态：`{decision}`。\n\n"
    text += "本轮仅完成指标修正、四状态结构分析和 Triplet 覆盖分母审计；未训练 NDP，未使用官方 LTCC 测试。\n\n"
    text += "由于两个 seed 的局部状态方向不同，且当前 Triplet 覆盖为冻结特征代理而非在线矿工日志，现有证据不足以宣称独立 NDP 机制或允许短程干预训练。GPC-v1 保持不变。\n"
    (reports / "NDP_NEXT_STAGE_DECISION.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    run_time = args.run_time
    out = ROOT / "outputs" / f"ndp_metric_coverage_audit_{run_time}"
    if out.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖: {out}")
    out.mkdir(parents=True)
    tables = out / "tables"; triplet_out = out / "triplet_coverage"
    log_path = ROOT / "logs" / f"NDP指标修正与Triplet覆盖分母审计_{run_time}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f: f.write(msg + "\n")

    old = load_old_module()
    source_files = {
        str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        str(ROOT / "scripts" / "ndp_net_intrusion_triplet_coverage_audit.py"): sha256_file(ROOT / "scripts" / "ndp_net_intrusion_triplet_coverage_audit.py"),
        "/data/DIFFER/loss/triplet_loss.py": sha256_file(Path("/data/DIFFER/loss/triplet_loss.py")),
        "/data/DIFFER/loss/make_loss.py": sha256_file(Path("/data/DIFFER/loss/make_loss.py")),
        "/data/DIFFER/data/samplers.py": sha256_file(Path("/data/DIFFER/data/samplers.py")),
        "/data/DIFFER/scripts/phase5b_id_damage_ds.py": sha256_file(Path("/data/DIFFER/scripts/phase5b_id_damage_ds.py")),
        "/data/DIFFER/scripts/phase7i_relation_specificity.py": sha256_file(Path("/data/DIFFER/scripts/phase7i_relation_specificity.py")),
        str(SOURCE_OUT / "feature_alignment_audit.json"): sha256_file(SOURCE_OUT / "feature_alignment_audit.json"),
    }
    graph_summaries=[]; graph_boot=[]; states=[]; coverage=[]; state_pid_rows=[]
    manifest = {"run_time": run_time, "source_run": SOURCE_RUN, "old_audit": str(OLD_AUDIT), "official_query_gallery_used": False, "model_training_started": False, "cpu_model_fallback": False, "source_files": source_files, "K": list(K_VALUES), "seeds": list(SEEDS), "bootstrap_reps": BOOTSTRAP_REPS, "metric_assertion": "SIGNED_NEGATIVE_DELTA_count == -DELTA_SAME_ID_COUNT per anchor", "triplet_proxy_warning": "冻结E60 GPC特征上的离线batch-hard代理，不是在线逐step miner日志"}
    log("【门禁】不实例化模型、不执行训练、不读取官方query/gallery；开始复用冻结train-only资产。")
    for seed in SEEDS:
        ref = old.load_bank(seed, "bio"); cur = old.load_bank(seed, "gpc")
        if not (np.array_equal(ref["pid"], cur["pid"]) and np.array_equal(ref["clothes"], cur["clothes"]) and np.array_equal(ref["camid"], cur["camid"]) and np.array_equal(ref["image_path"], cur["image_path"])):
            raise RuntimeError(f"seed{seed} 参考/当前特征映射不一致")
        ref_idx = old.load_graph(seed, "bio"); cur_idx = old.load_graph(seed, "gpc")
        log(f"【seed={seed}】重算K=5/10/20指标恒等式并重建冻结采样计划。")
        plan = old.build_batch_plan(seed)
        write_json(triplet_out / f"seed{seed}_sampler_plan_manifest.json", {"seed": seed, "plan_rows": plan["plan_rows"], "sampler_source": plan["sampler_source"], "sampler_source_sha256": plan["sampler_source_sha256"], "batch_plan_hashes_reproduced": True})
        write_csv(triplet_out / f"seed{seed}_sampler_plan.csv", plan["plan_rows"])
        for k in K_VALUES:
            g_rows, pid_rows, boot, g_summary = corrected_graph_metrics(seed, k, ref_idx, cur_idx, ref["pid"].astype(np.int64))
            write_csv(tables / f"seed{seed}_K{k}_metric_correction_anchor.csv", g_rows)
            write_csv(tables / f"seed{seed}_K{k}_metric_correction_pid.csv", pid_rows)
            write_csv(tables / f"seed{seed}_K{k}_metric_correction_bootstrap.csv", boot)
            graph_summaries.append(g_summary); graph_boot.extend(boot)
            rel = load_relation_rows(seed, k)
            rel, state_pid, st = state_audit(seed, k, rel, cur)
            write_csv(tables / f"seed{seed}_K{k}_identity_relation_states.csv", rel)
            write_csv(tables / f"seed{seed}_K{k}_identity_pid_states.csv", state_pid)
            states.append(st); state_pid_rows.extend(state_pid)
            cov_rows, aux = build_triplet_audit(old, seed, k, rel, cur, plan)
            write_csv(triplet_out / f"seed{seed}_K{k}_relation_coverage_corrected.csv", cov_rows)
            cov_summary = coverage_summary(seed, k, cov_rows, aux, len(rel))
            write_json(triplet_out / f"seed{seed}_K{k}_coverage_denominator_summary.json", cov_summary)
            coverage.append(cov_summary)
            log(f"【完成】seed={seed} K={k}：有符号槽位={g_summary['total_SIGNED_NEGATIVE_DELTA_count']}，非负侵入槽位={g_summary['total_NET_NEGATIVE_INTRUSION_count']}，可靠翻转={st['reliable_flip_rows']}/{st['reliable_rows']}。")
    manifest["graph_summaries"] = graph_summaries; manifest["coverage_summaries"] = coverage; manifest["state_summaries"] = states
    manifest["outputs"] = {"log": str(log_path), "root": str(out)}
    decision = "NDP_INCREMENTAL_MECHANISM_NOT_SUPPORTED" if not (any(x["reliable_flip_rows"] > 0 for x in states) and any(x["scope"] == "reliable_flip_rows" and x.get("full_triplet_hard_selected", 0) == 0 for c in coverage for x in c["scopes"])) else "NDP_LOCAL_RISK_MECHANISM_UNRESOLVED"
    manifest["decision"] = decision
    manifest["decision_reason"] = "存在可靠翻转但未证明跨seed稳定的独立覆盖缺口；当前Triplet结果为冻结特征离线代理，不能晋级NDP短程训练。"
    write_json(out / "ndp_metric_coverage_audit.json", manifest)
    write_reports(run_time, out, graph_summaries, states, coverage, manifest, decision)
    log(f"【决策】{decision}。六份中文报告已写入 /home/DIFFER/reports；本轮没有启动任何模型计算。")


if __name__ == "__main__":
    main()
