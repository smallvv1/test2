#!/usr/bin/env python3.10
"""NDP P0--P4 独立冻结特征审计。

本脚本只读取已经导出的 train-only 特征和 update0 冻结的 difficult-K5 图。
它重构 R/V/D 三个互不替代的量，比较 A--E 五种关系保护目标，并在
update200 的特征空间计算一阶局部影响。脚本不加载训练模型、不调用
optimizer.step、不写入任何原始实验目录。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


HISTORY_UPDATES = (0, 20, 50, 100)
CURRENT_UPDATE = 200
FUTURE_UPDATE = 400
PRE_REGISTERED = {
    "history_updates": list(HISTORY_UPDATES),
    "minimum_history_nodes": 4,
    "stable_min_margin": 0.05,
    "max_history_std": 0.05,
    "vulnerability_margin": 0.05,
    "vulnerability_tail_margin": 0.08,
    "vulnerability_tail_count": 2,
    "directional_asymmetry_margin": 0.05,
    "degradation_epsilon": 0.02,
    "safe_delta_min": 0.05,
    "safe_epsilon_ref": 0.02,
    "diagnostic_eta": 1.0e-3,
    "gradient_budget_norm": 1.0,
    "random_repeats": 200,
    "bootstrap_repeats": 2000,
    "bootstrap_seed": 20261023,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_graph(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in payload["relations"]
        if row.get("graph_type") == "difficult" and int(row.get("k_pid", -1)) == 5
    ]
    rows.sort(key=lambda row: (int(row["anchor_order"]), int(row["negative_pid"])))
    if len(rows) != 512 * 5:
        raise RuntimeError(f"固定 difficult-K5 图应有2560条关系，实际 {len(rows)}: {path}")
    keys = [(int(row["anchor_image_index"]), int(row["negative_pid"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("关系图包含重复 anchor-negative PID")
    return rows


def cosine_margin(features: np.ndarray, row: dict[str, Any], image_to_pos: dict[int, int]) -> float:
    anchor = features[image_to_pos[int(row["anchor_image_index"])]].astype(np.float32)
    pos = features[[image_to_pos[int(x)] for x in row["positive_indices"]]].astype(np.float32)
    neg = features[[image_to_pos[int(x)] for x in row["negative_indices"]]].astype(np.float32)
    anchor = anchor / max(float(np.linalg.norm(anchor)), 1.0e-12)
    pos = pos / np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1.0e-12)
    neg = neg / np.maximum(np.linalg.norm(neg, axis=1, keepdims=True), 1.0e-12)
    # 本轮主定义固定为 mean-positive/max-negative；Q25 只用于构造
    # 历史安全边界，不改变关系 Margin 的语义。
    return float(np.mean(pos @ anchor) - np.max(neg @ anchor))


def legal_labels(row: dict[str, Any], pid_by_image: dict[int, int]) -> bool:
    anchor = int(row["anchor_image_index"])
    anchor_pid = int(row["anchor_pid"])
    if pid_by_image.get(anchor) != anchor_pid:
        return False
    positives = [int(x) for x in row["positive_indices"]]
    negatives = [int(x) for x in row["negative_indices"]]
    if not positives or not negatives:
        return False
    return (
        all(pid_by_image.get(x) == anchor_pid and x != anchor for x in positives)
        and all(pid_by_image.get(x) == int(row["negative_pid"]) and pid_by_image.get(x) != anchor_pid for x in negatives)
    )


def historical_reliable(history: Iterable[float], labels_legal: bool = True) -> bool:
    """预注册 R：只接受 update0/20/50/100，不读取当前或未来节点。"""
    values = np.asarray(list(history), dtype=np.float64)
    return bool(
        labels_legal
        and values.size >= PRE_REGISTERED["minimum_history_nodes"]
        and np.all(values > 0.0)
        and values.min() >= PRE_REGISTERED["stable_min_margin"]
        and values.std() <= PRE_REGISTERED["max_history_std"]
    )


def current_vulnerable(current_margin: float, tail_count: int, wrong_count: int, asymmetry: float | None) -> bool:
    """预注册 V：只使用 update200 当前局部风险，不用历史方差作唯一判据。"""
    return bool(
        current_margin <= PRE_REGISTERED["vulnerability_margin"]
        or tail_count >= PRE_REGISTERED["vulnerability_tail_count"]
        or wrong_count >= 1
        or (asymmetry is not None and asymmetry >= PRE_REGISTERED["directional_asymmetry_margin"])
    )


def historical_safe_boundary(history: Iterable[float]) -> float:
    values = np.asarray(list(history), dtype=np.float64)
    if values.size == 0:
        raise ValueError("历史窗口不能为空")
    return float(max(PRE_REGISTERED["safe_delta_min"], np.quantile(values, 0.25) - PRE_REGISTERED["safe_epsilon_ref"]))


def first_order_delta(grad_margin: torch.Tensor, grad_loss: torch.Tensor, eta: float | None = None) -> torch.Tensor:
    """沿 -grad(loss) 的一阶 Margin 变化，正数表示预计改善。"""
    step = PRE_REGISTERED["diagnostic_eta"] if eta is None else float(eta)
    return -step * torch.sum(grad_margin * grad_loss)


def bootstrap_rate(values: np.ndarray, rng: np.random.Generator) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    point = float(np.mean(values))
    draws = rng.choice(values, size=(PRE_REGISTERED["bootstrap_repeats"], values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi)


def gini(counts: Iterable[int]) -> float:
    x = np.asarray(list(counts), dtype=np.float64)
    if x.size == 0 or float(x.sum()) == 0.0:
        return 0.0
    x.sort()
    n = x.size
    return float((2.0 * np.sum((np.arange(1, n + 1) * x)) / (n * np.sum(x))) - (n + 1) / n)


def relation_values(row: dict[str, Any], key: str) -> list[int]:
    value = row[key]
    if isinstance(value, str):
        value = json.loads(value)
    return [int(x) for x in value]


def row_stats(rows: list[dict[str, Any]], indices: np.ndarray, total_flips: int, rng: np.random.Generator) -> dict[str, Any]:
    selected = [rows[int(i)] for i in indices]
    flips = np.asarray([int(x["validation_flip"]) for x in selected], dtype=np.float64)
    current_errors = [x for x in selected if int(x["current_correct"]) == 0]
    negative_counts = Counter(int(x["negative_pid"]) for x in selected)
    anchor_counts = Counter(int(x["anchor_pid"]) for x in selected)
    flip_rate, flip_lo, flip_hi = bootstrap_rate(flips, rng)
    return {
        "selected": len(selected),
        "active_safe_boundary": int(sum(int(x["safe_boundary_active"]) for x in selected)),
        "future_flips": int(sum(int(x["validation_flip"]) for x in selected)),
        "future_flip_coverage": int(sum(int(x["validation_flip"]) for x in selected)),
        "future_flip_coverage_rate_of_all": float(sum(int(x["validation_flip"]) for x in selected) / total_flips) if total_flips else None,
        "future_flip_rate": flip_rate,
        "future_flip_rate_bootstrap95": [flip_lo, flip_hi],
        "current_errors": len(current_errors),
        "current_error_negative_pids": len({int(x["negative_pid"]) for x in current_errors}),
        "anchor_pid_count": len(anchor_counts),
        "negative_pid_count": len(negative_counts),
        "negative_pid_exposure_max": max(negative_counts.values(), default=0),
        "negative_pid_exposure_gini": gini(negative_counts.values()),
        "anchor_pid_exposure_max": max(anchor_counts.values(), default=0),
        "current_margin_mean": float(np.mean([float(x["current_margin"]) for x in selected])) if selected else None,
        "current_margin_median": float(np.median([float(x["current_margin"]) for x in selected])) if selected else None,
        "safe_active_gradient_norm": float(np.mean([float(x["safe_gradient_norm"]) for x in selected])) if selected else None,
    }


def torch_relation_margin(x: torch.Tensor, row: dict[str, Any], local: dict[int, int]) -> torch.Tensor:
    anchor = x[local[int(row["anchor_image_index"])]]
    pos = x[[local[int(v)] for v in relation_values(row, "positive_indices")]]
    neg = x[[local[int(v)] for v in relation_values(row, "negative_indices")]]
    za = F.normalize(anchor.float(), dim=-1)
    zp = F.normalize(pos.float(), dim=-1)
    zn = F.normalize(neg.float(), dim=-1)
    return (zp @ za).quantile(0.25) - (zn @ za).max()


def safe_boundary_gradient_norms(
    raw_features: np.ndarray,
    rows: list[dict[str, Any]],
    mask_map: dict[str, np.ndarray],
    image_to_pos: dict[int, int],
) -> dict[str, float]:
    """统一 raw feat[0]、FP32 cosine、相同 safe boundary 的冻结特征梯度。"""
    torch.set_num_threads(min(4, torch.get_num_threads()))
    x = torch.tensor(raw_features, dtype=torch.float32, requires_grad=True)
    margins = []
    for row in rows:
        local = {int(v): int(image_to_pos[int(v)]) for v in row["_all_image_indices"]}
        margins.append(torch_relation_margin(x, row, local))
    current = torch.stack(margins)
    boundary = torch.tensor([float(row["safe_boundary"]) for row in rows], dtype=torch.float32)
    raw = F.relu(boundary - current)
    result: dict[str, float] = {}
    for name, mask in mask_map.items():
        selected = torch.tensor(mask, dtype=torch.bool)
        active = selected & (raw > 0)
        if bool(active.any()):
            loss = raw[active].mean()
        else:
            loss = current.sum() * 0.0
        grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
        result[name] = float(torch.linalg.vector_norm(grad))
    return result


def priority_indices(rows: list[dict[str, Any]], method: str, budget: int, rng: np.random.Generator) -> np.ndarray:
    n = len(rows)
    if budget <= 0:
        return np.asarray([], dtype=np.int64)
    if method == "B0_random":
        return np.sort(rng.choice(n, size=budget, replace=False))
    if method == "B1_current_hard":
        return np.asarray(sorted(range(n), key=lambda i: (float(rows[i]["current_margin"]), i))[:budget], dtype=np.int64)
    keys = {
        "B2_reliability": "reliable_history",
        "B3_reliability_vulnerability": "current_vulnerability",
        "B4_reliability_degradation": "observed_degradation",
        "B5_safe_boundary_activation": "safe_boundary_active",
        "B6_reliability_multi_pid": "multi_pid_risk",
        "B7_reliability_bidirectional": "bidirectional_risk",
    }
    key = keys[method]
    # 预算匹配版本：先按预注册门控排序，再用当前 margin 作为确定性 tie-break。
    # update400 不参与 score。
    def score(i: int) -> tuple[int, float, int]:
        row = rows[i]
        gate = bool(row["reliable_history"]) and bool(row[key]) if method != "B2_reliability" else bool(row[key])
        return (0 if gate else 1, float(row["current_margin"]), i)
    return np.asarray(sorted(range(n), key=score)[:budget], dtype=np.int64)


def local_impact_for_target(
    rows: list[dict[str, Any]],
    by_anchor: dict[int, list[int]],
    reverse_by_pid_pair: dict[tuple[int, int], int],
    raw_bank: np.ndarray,
    image_to_pos: dict[int, int],
    target_index: int,
) -> list[dict[str, Any]]:
    target = rows[target_index]
    anchor_image = int(target["anchor_image_index"])
    same_anchor = by_anchor[anchor_image]
    reverse_index = reverse_by_pid_pair.get((int(target["negative_pid"]), int(target["anchor_pid"])))
    relation_indices = list(same_anchor)
    if reverse_index is not None and reverse_index not in relation_indices:
        relation_indices.append(reverse_index)
    image_ids = sorted({int(v) for rid in relation_indices for v in rows[rid]["_all_image_indices"]})
    local = {v: i for i, v in enumerate(image_ids)}
    x0 = torch.tensor(raw_bank[[image_to_pos[v] for v in image_ids]], dtype=torch.float32)
    x = x0.detach().clone().requires_grad_(True)
    margin_t = {rid: torch_relation_margin(x, rows[rid], local) for rid in relation_indices}

    def margin_loss(rid: int, kind: str) -> torch.Tensor:
        m = margin_t[rid]
        if kind == "margin_hinge":
            return F.relu(torch.tensor(PRE_REGISTERED["vulnerability_margin"], dtype=torch.float32) - m)
        return F.relu(torch.tensor(float(rows[rid]["safe_boundary"]), dtype=torch.float32) - m)

    # Triplet is the original raw Euclidean local Batch-Hard proxy: max positive,
    # min negative, SoftMarginLoss(dist_an-dist_ap)=softplus(dist_ap-dist_an).
    all_pos = sorted({int(v) for rid in same_anchor for v in relation_values(rows[rid], "positive_indices")})
    all_neg = sorted({int(v) for rid in same_anchor for v in relation_values(rows[rid], "negative_indices")})
    anchor = x[local[anchor_image]]
    pos = x[[local[v] for v in all_pos]]
    neg = x[[local[v] for v in all_neg]]
    d_pos = torch.linalg.vector_norm(pos - anchor, dim=1)
    d_neg = torch.linalg.vector_norm(neg - anchor, dim=1)
    triplet_loss = F.softplus(d_pos.max() - d_neg.min())

    loss_by_method: dict[str, torch.Tensor] = {
        "triplet_batchhard": triplet_loss,
        "margin_hinge_target": margin_loss(target_index, "margin_hinge"),
        "safe_boundary_target": margin_loss(target_index, "safe"),
        "multi_pid_safe": torch.stack([margin_loss(rid, "safe") for rid in same_anchor]).mean(),
    }
    if reverse_index is not None:
        loss_by_method["bidirectional_safe"] = torch.stack([margin_loss(target_index, "safe"), margin_loss(reverse_index, "safe")]).mean()
    else:
        loss_by_method["bidirectional_safe"] = loss_by_method["safe_boundary_target"]

    gradients: dict[str, torch.Tensor] = {}
    for method, loss in loss_by_method.items():
        g = torch.autograd.grad(loss, x, retain_graph=True, allow_unused=False)[0]
        norm = torch.linalg.vector_norm(g)
        if float(norm) > 0.0 and bool(torch.isfinite(norm)):
            gradients[method] = g * (PRE_REGISTERED["gradient_budget_norm"] / norm)
        else:
            gradients[method] = torch.zeros_like(g)

    neighbors = list(same_anchor)
    if reverse_index is not None and reverse_index not in neighbors:
        neighbors.append(reverse_index)
    out: list[dict[str, Any]] = []
    for method, g in gradients.items():
        for neighbor_index in neighbors:
            gm = torch.autograd.grad(margin_t[neighbor_index], x, retain_graph=True, allow_unused=False)[0]
            impact = first_order_delta(gm, g)
            if neighbor_index == target_index:
                role = "target"
            elif neighbor_index == reverse_index:
                role = "reverse_direction"
            else:
                role = "other_same_anchor_negative_pid"
            out.append({
                "target_relation_index": target_index,
                "target_anchor_image_index": anchor_image,
                "target_anchor_pid": int(target["anchor_pid"]),
                "target_negative_pid": int(target["negative_pid"]),
                "neighbor_relation_index": neighbor_index,
                "neighbor_role": role,
                "neighbor_anchor_pid": int(rows[neighbor_index]["anchor_pid"]),
                "neighbor_negative_pid": int(rows[neighbor_index]["negative_pid"]),
                "neighbor_current_margin": float(rows[neighbor_index]["current_margin"]),
                "neighbor_current_correct": int(rows[neighbor_index]["current_correct"]),
                "method": method,
                "gradient_budget_norm": PRE_REGISTERED["gradient_budget_norm"],
                "raw_loss_gradient_norm": float(torch.linalg.vector_norm(torch.autograd.grad(loss_by_method[method], x, retain_graph=True)[0])),
                "predicted_delta_margin": float(impact.detach()),
                "finite": bool(torch.isfinite(impact)),
                "safe_boundary": float(rows[neighbor_index]["safe_boundary"]),
                "neighbor_would_cross_zero": bool(float(rows[neighbor_index]["current_margin"]) > 0 and float(impact) < -float(rows[neighbor_index]["current_margin"])),
            })
    return out


def finite_difference_check(
    rows: list[dict[str, Any]],
    by_anchor: dict[int, list[int]],
    raw_bank: np.ndarray,
    image_to_pos: dict[int, int],
    target_index: int,
) -> dict[str, Any]:
    target = rows[target_index]
    same_anchor = by_anchor[int(target["anchor_image_index"])]
    image_ids = sorted({int(v) for rid in same_anchor for v in rows[rid]["_all_image_indices"]})
    local = {v: i for i, v in enumerate(image_ids)}
    x = torch.tensor(raw_bank[[image_to_pos[v] for v in image_ids]], dtype=torch.float32, requires_grad=True)
    m = torch_relation_margin(x, target, local)
    loss = F.relu(torch.tensor(float(target["safe_boundary"]), dtype=torch.float32) - m)
    g = torch.autograd.grad(loss, x, retain_graph=True)[0]
    norm = torch.linalg.vector_norm(g)
    if float(norm) == 0.0:
        return {"target_relation_index": target_index, "active": False, "reason": "zero_gradient"}
    direction = g / norm
    before = float(m.detach())
    after = float(torch_relation_margin((x.detach() - PRE_REGISTERED["diagnostic_eta"] * direction), target, local))
    predicted = float(first_order_delta(torch.autograd.grad(m, x, retain_graph=True)[0], direction))
    return {
        "target_relation_index": target_index,
        "active": True,
        "margin_before": before,
        "margin_after": after,
        "actual_delta": after - before,
        "predicted_delta": predicted,
        "absolute_error": abs((after - before) - predicted),
    }


def audit_seed(seed: int, graph_path: Path, feature_path: Path, output_root: Path) -> dict[str, Any]:
    rows = read_graph(graph_path)
    bank = np.load(feature_path, allow_pickle=False)
    updates = [int(x) for x in bank["updates"]]
    if not set(HISTORY_UPDATES + (CURRENT_UPDATE, FUTURE_UPDATE)).issubset(updates):
        raise RuntimeError(f"特征节点缺少预注册节点: {updates}")
    image_ids = [int(x) for x in bank["image_index"]]
    image_to_pos = {v: i for i, v in enumerate(image_ids)}
    pid_by_image = {int(i): int(p) for i, p in zip(bank["image_index"], bank["pid"])}
    feature_by_update = {u: bank["features"][updates.index(u)].astype(np.float32) for u in updates}
    raw_by_update = {u: bank["raw_features"][updates.index(u)].astype(np.float32) for u in updates}

    by_anchor: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_anchor[int(row["anchor_image_index"])].append(i)
    reverse_by_pid_pair: dict[tuple[int, int], int] = {}
    for i, row in enumerate(rows):
        reverse_by_pid_pair.setdefault((int(row["anchor_pid"]), int(row["negative_pid"])), i)

    current_by_anchor: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        current_by_anchor[int(row["anchor_image_index"])].append(cosine_margin(feature_by_update[CURRENT_UPDATE], row, image_to_pos))

    audit_rows: list[dict[str, Any]] = []
    for relation_index, row in enumerate(rows):
        margins = {u: cosine_margin(feature_by_update[u], row, image_to_pos) for u in updates}
        hist = np.asarray([margins[u] for u in HISTORY_UPDATES], dtype=np.float64)
        current = float(margins[CURRENT_UPDATE])
        future = float(margins[FUTURE_UPDATE])
        anchor_margins = np.asarray(current_by_anchor[int(row["anchor_image_index"])], dtype=np.float64)
        tail_count = int(np.sum(anchor_margins <= PRE_REGISTERED["vulnerability_tail_margin"]))
        wrong_count = int(np.sum(anchor_margins <= 0.0))
        reverse_index = reverse_by_pid_pair.get((int(row["negative_pid"]), int(row["anchor_pid"])))
        reverse_current = float(margins[CURRENT_UPDATE]) if reverse_index == relation_index else None
        if reverse_index is not None:
            reverse_row = rows[reverse_index]
            reverse_current = cosine_margin(feature_by_update[CURRENT_UPDATE], reverse_row, image_to_pos)
        asym = abs(current - reverse_current) if reverse_current is not None else None
        valid = legal_labels(row, pid_by_image)
        reliable = historical_reliable(hist, labels_legal=valid)
        vulnerability_reasons: list[str] = []
        if current <= PRE_REGISTERED["vulnerability_margin"]:
            vulnerability_reasons.append("current_margin_le_0.05")
        if tail_count >= PRE_REGISTERED["vulnerability_tail_count"]:
            vulnerability_reasons.append("multiple_negative_pids_near_boundary")
        if wrong_count >= 1:
            vulnerability_reasons.append("current_anchor_has_wrong_negative_pid")
        if asym is not None and asym >= PRE_REGISTERED["directional_asymmetry_margin"]:
            vulnerability_reasons.append("bidirectional_current_asymmetry")
        vulnerability = current_vulnerable(current, tail_count, wrong_count, asym)
        reference = float(np.mean(hist))
        q25 = float(np.quantile(hist, 0.25))
        safe_boundary = historical_safe_boundary(hist)
        degradation = reference - current
        old_reliable = bool(valid and np.all(np.asarray([*hist, current]) > 0.0) and min(float(hist.min()), current) >= 0.05 and np.std(np.asarray([*hist, current])) <= 0.05)
        old_exclusion: list[str] = []
        if current <= 0.0:
            old_exclusion.append("old_gate_includes_update200_and_current_is_wrong")
        elif current < 0.05:
            old_exclusion.append("old_gate_includes_update200_and_current_margin_below_0.05")
        if np.std(np.asarray([*hist, current])) > 0.05:
            old_exclusion.append("old_gate_history_std_with_current_exceeds_0.05")
        if not old_exclusion and not old_reliable:
            old_exclusion.append("old_reliability_conditions_not_met")
        safe_raw = max(0.0, safe_boundary - current)
        values = {
            "A_fixed_update0": max(0.0, margins[0] - current - PRE_REGISTERED["degradation_epsilon"]),
            "B_last_pre200": max(0.0, margins[100] - current - PRE_REGISTERED["degradation_epsilon"]),
            "C_historical_safe_boundary": safe_raw,
            "D_current_fixed_margin": max(0.0, PRE_REGISTERED["vulnerability_margin"] - current),
            "E_batchhard_softmargin_proxy": float(np.logaddexp(0.0, -current)),
        }
        all_images = sorted({int(v) for v in [row["anchor_image_index"], *row["positive_indices"], *row["negative_indices"]]})
        audit_rows.append({
            "seed": seed,
            "relation_index": relation_index,
            "anchor_image_index": int(row["anchor_image_index"]),
            "anchor_pid": int(row["anchor_pid"]),
            "anchor_clothes_id": int(bank["clothes"][image_to_pos[int(row["anchor_image_index"])] ]),
            "negative_pid": int(row["negative_pid"]),
            "negative_indices": json.dumps([int(x) for x in row["negative_indices"]]),
            "positive_indices": json.dumps([int(x) for x in row["positive_indices"]]),
            "positive_clothes_ids": json.dumps(sorted({int(bank["clothes"][image_to_pos[int(x)]]) for x in row["positive_indices"]})),
            "negative_clothes_ids": json.dumps(sorted({int(bank["clothes"][image_to_pos[int(x)]]) for x in row["negative_indices"]})),
            "M0": margins[0], "M20": margins[20], "M50": margins[50], "M100": margins[100], "M200": current, "M400": future,
            "reference_margin": margins[0], "current_margin": current, "validation_margin": future,
            "history_correct": int(np.all(hist > 0.0)),
            "history_correct_rate": float(np.mean(hist > 0.0)),
            "history_min_margin": float(hist.min()),
            "history_std": float(hist.std()),
            "history_node_count": len(hist),
            "history_labels_legal": int(valid),
            "reliable_history": int(reliable),
            "current_correct": int(current > 0.0),
            "current_vulnerability": int(vulnerability),
            "vulnerability_reasons": ";".join(vulnerability_reasons),
            "current_negative_pid_near_tail_count": tail_count,
            "current_wrong_negative_pid_count": wrong_count,
            "reverse_relation_index": reverse_index if reverse_index is not None else "",
            "bidirectional_pair_exists": int(reverse_index is not None),
            "bidirectional_current_asymmetry": asym if asym is not None else "",
            "reference_margin_mean_history": reference,
            "degradation_D": degradation,
            "observed_degradation": int(degradation > PRE_REGISTERED["degradation_epsilon"]),
            "safe_boundary": safe_boundary,
            "safe_boundary_raw_hinge": safe_raw,
            "safe_boundary_active": int(safe_raw > 0.0),
            "A_raw_hinge": values["A_fixed_update0"], "A_active": int(values["A_fixed_update0"] > 0.0),
            "B_raw_hinge": values["B_last_pre200"], "B_active": int(values["B_last_pre200"] > 0.0),
            "C_raw_hinge": values["C_historical_safe_boundary"], "C_active": int(values["C_historical_safe_boundary"] > 0.0),
            "D_raw_hinge": values["D_current_fixed_margin"], "D_active": int(values["D_current_fixed_margin"] > 0.0),
            "E_softmargin_value": values["E_batchhard_softmargin_proxy"], "E_active": 1,
            "old_reliable": int(old_reliable),
            "old_reliability_exclusion_reason": ";".join(old_exclusion),
            "validation_flip": int(current > 0.0 and future <= 0.0),
            "validation_new_error_pid": int(current > 0.0 and future <= 0.0),
            "multi_pid_risk": int(tail_count >= PRE_REGISTERED["vulnerability_tail_count"] or wrong_count >= 1),
            "bidirectional_risk": int(reverse_index is not None and asym is not None and asym >= PRE_REGISTERED["directional_asymmetry_margin"]),
            "positive_cross_clothes": int(bool(row.get("positive_cross_clothes", False))),
            "negative_cross_camera": int(bool(row.get("negative_cross_camera", False))),
            "_all_image_indices": all_images,
        })

    # Internal-only fields are removed from CSV after gradients are computed.
    for row in audit_rows:
        row["safe_gradient_norm"] = 0.0
    mask_map = {
        "B0_random": np.ones(len(audit_rows), dtype=bool),
        "B1_current_hard": np.ones(len(audit_rows), dtype=bool),
        "B2_reliability": np.asarray([bool(r["reliable_history"]) for r in audit_rows]),
        "B3_reliability_vulnerability": np.asarray([bool(r["reliable_history"]) and bool(r["current_vulnerability"]) for r in audit_rows]),
        "B4_reliability_degradation": np.asarray([bool(r["reliable_history"]) and bool(r["observed_degradation"]) for r in audit_rows]),
        "B5_safe_boundary_activation": np.asarray([bool(r["reliable_history"]) and bool(r["safe_boundary_active"]) for r in audit_rows]),
        "B6_reliability_multi_pid": np.asarray([bool(r["reliable_history"]) and bool(r["multi_pid_risk"]) for r in audit_rows]),
        "B7_reliability_bidirectional": np.asarray([bool(r["reliable_history"]) and bool(r["bidirectional_risk"]) for r in audit_rows]),
    }
    gradient_norms = safe_boundary_gradient_norms(raw_by_update[CURRENT_UPDATE], audit_rows, mask_map, image_to_pos)
    for name, mask in mask_map.items():
        for i, row in enumerate(audit_rows):
            if bool(mask[i]):
                row["safe_gradient_norm"] = gradient_norms[name]

    seed_out = output_root / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    csv_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in audit_rows]
    write_csv(seed_out / ("relation_evidence_seed%d.csv" % seed), csv_rows)
    flip_rows = [row for row in csv_rows if int(row["validation_flip"])]
    write_csv(seed_out / ("missed_flip_rows_seed%d.csv" % seed), flip_rows)

    total_flips = len(flip_rows)
    rng = np.random.default_rng(PRE_REGISTERED["bootstrap_seed"] + seed)
    native_masks = {
        "B0_random": np.ones(len(audit_rows), dtype=bool),
        "B1_current_hard": np.asarray([float(r["current_margin"]) <= PRE_REGISTERED["vulnerability_margin"] for r in audit_rows]),
        "B2_reliability": mask_map["B2_reliability"],
        "B3_reliability_vulnerability": mask_map["B3_reliability_vulnerability"],
        "B4_reliability_degradation": mask_map["B4_reliability_degradation"],
        "B5_safe_boundary_activation": mask_map["B5_safe_boundary_activation"],
        "B6_reliability_multi_pid": mask_map["B6_reliability_multi_pid"],
        "B7_reliability_bidirectional": mask_map["B7_reliability_bidirectional"],
    }
    relation_count = len(audit_rows)
    positive_native = [int(v.sum()) for v in native_masks.values() if int(v.sum()) > 0]
    budget = min(positive_native) if positive_native else 0
    selection_rows: list[dict[str, Any]] = []
    selection_summary: dict[str, Any] = {"budget": budget, "native": {}, "budget_matched": {}, "random_controls": {}}
    for method, mask in native_masks.items():
        if method == "B0_random":
            idx = priority_indices(audit_rows, method, budget, rng)
        else:
            idx = np.flatnonzero(mask)
        stats = row_stats(audit_rows, idx, total_flips, rng)
        stats["native_mask_definition"] = method
        selection_summary["native"][method] = stats
        for i in idx:
            selection_rows.append({"kind": "native", "method": method, "relation_index": int(i), "selected": 1})
        if budget > 0:
            if method == "B0_random":
                matched_idx = priority_indices(audit_rows, method, budget, rng)
            elif method == "B1_current_hard":
                matched_idx = priority_indices(audit_rows, method, budget, rng)
            else:
                matched_idx = priority_indices(audit_rows, method, budget, rng)
            selection_summary["budget_matched"][method] = row_stats(audit_rows, matched_idx, total_flips, rng)
            for i in matched_idx:
                selection_rows.append({"kind": "budget_matched", "method": method, "relation_index": int(i), "selected": 1})
    for repeat in range(PRE_REGISTERED["random_repeats"]):
        if budget <= 0:
            break
        control_rng = np.random.default_rng(PRE_REGISTERED["bootstrap_seed"] + seed * 10000 + repeat)
        idx = np.sort(control_rng.choice(relation_count, size=budget, replace=False))
        stats = row_stats(audit_rows, idx, total_flips, rng)
        stats["repeat"] = repeat
        selection_rows.append({"kind": "random_control", "method": "B0_random_budget", "repeat": repeat, **stats})
    write_csv(seed_out / ("selection_rows_seed%d.csv" % seed), selection_rows)
    json_dump(seed_out / ("selection_summary_seed%d.json" % seed), selection_summary)

    # P4 local collateral: focus on the newly meaningful R∩V cell, but keep all
    # five PID directions around each selected anchor.
    targets = np.flatnonzero(mask_map["B3_reliability_vulnerability"])
    impact_rows: list[dict[str, Any]] = []
    finite_difference_rows: list[dict[str, Any]] = []
    for target_index in targets:
        impact_rows.extend(local_impact_for_target(audit_rows, by_anchor, reverse_by_pid_pair, raw_by_update[CURRENT_UPDATE], image_to_pos, int(target_index)))
        if len(finite_difference_rows) < 32:
            finite_difference_rows.append(finite_difference_check(audit_rows, by_anchor, raw_by_update[CURRENT_UPDATE], image_to_pos, int(target_index)))
    write_csv(seed_out / ("neighborhood_collateral_rows_seed%d.csv" % seed), impact_rows)
    json_dump(seed_out / ("finite_difference_rows_seed%d.json" % seed), finite_difference_rows)
    impact_summary: dict[str, Any] = {"targets": int(len(targets)), "rows": len(impact_rows), "methods": {}}
    for method in sorted({str(r["method"]) for r in impact_rows}):
        vals = np.asarray([float(r["predicted_delta_margin"]) for r in impact_rows if r["method"] == method], dtype=np.float64)
        others = np.asarray([float(r["predicted_delta_margin"]) for r in impact_rows if r["method"] == method and r["neighbor_role"] != "target"], dtype=np.float64)
        impact_summary["methods"][method] = {
            "rows": int(vals.size),
            "target_mean_delta": float(np.mean([float(r["predicted_delta_margin"]) for r in impact_rows if r["method"] == method and r["neighbor_role"] == "target"])) if any(r["method"] == method and r["neighbor_role"] == "target" for r in impact_rows) else None,
            "other_mean_delta": float(others.mean()) if others.size else None,
            "other_harm_count": int(np.sum(others < 0.0)),
            "other_wrong_after_predicted": int(sum(bool(r["neighbor_would_cross_zero"]) for r in impact_rows if r["method"] == method)),
            "finite": bool(np.isfinite(vals).all()) if vals.size else True,
        }
    json_dump(seed_out / ("neighborhood_collateral_summary_seed%d.json" % seed), impact_summary)

    summary = {
        "status": "PASS_OFFLINE_P0_P4_REDESIGN",
        "seed": seed,
        "relation_count": relation_count,
        "anchor_count": len(by_anchor),
        "negative_pid_per_anchor": 5,
        "pre_registered": PRE_REGISTERED,
        "counts": {
            "reliable_history": int(sum(bool(r["reliable_history"]) for r in audit_rows)),
            "current_vulnerable": int(sum(bool(r["current_vulnerability"]) for r in audit_rows)),
            "reliable_and_vulnerable": int(sum(bool(r["reliable_history"]) and bool(r["current_vulnerability"]) for r in audit_rows)),
            "observed_degradation": int(sum(bool(r["observed_degradation"]) for r in audit_rows)),
            "reliable_and_observed_degradation": int(sum(bool(r["reliable_history"]) and bool(r["observed_degradation"]) for r in audit_rows)),
            "current_errors": int(sum(not bool(r["current_correct"]) for r in audit_rows)),
            "future_flips": total_flips,
            "old_reliable": int(sum(bool(r["old_reliable"]) for r in audit_rows)),
            "reliable_newly_included_by_decoupling": int(sum(bool(r["reliable_history"]) and not bool(r["old_reliable"]) for r in audit_rows)),
        },
        "quadrants": {
            "high_reliable_low_vulnerable": int(sum(bool(r["reliable_history"]) and not bool(r["current_vulnerability"]) for r in audit_rows)),
            "high_reliable_high_vulnerable": int(sum(bool(r["reliable_history"]) and bool(r["current_vulnerability"]) for r in audit_rows)),
            "low_reliable_low_vulnerable": int(sum(not bool(r["reliable_history"]) and not bool(r["current_vulnerability"]) for r in audit_rows)),
            "low_reliable_high_vulnerable": int(sum(not bool(r["reliable_history"]) and bool(r["current_vulnerability"]) for r in audit_rows)),
        },
        "gradient_norms_safe_boundary": gradient_norms,
        "selection_budget": budget,
        "graph": str(graph_path),
        "graph_sha256": sha256_file(graph_path),
        "features": str(feature_path),
        "features_sha256": sha256_file(feature_path),
        "relation_csv": str(seed_out / ("relation_evidence_seed%d.csv" % seed)),
        "flip_csv": str(seed_out / ("missed_flip_rows_seed%d.csv" % seed)),
        "selection_summary": str(seed_out / ("selection_summary_seed%d.json" % seed)),
        "neighborhood_summary": str(seed_out / ("neighborhood_collateral_summary_seed%d.json" % seed)),
        "future_update400_used_only_as_evaluation_label": True,
        "optimizer_step_executed": False,
        "official_query_gallery_used": False,
    }
    json_dump(seed_out / ("redesign_summary_seed%d.json" % seed), summary)
    return {"summary": summary, "rows": audit_rows, "flip_rows": flip_rows, "selection": selection_summary, "impact": impact_summary}


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def write_reports(results: dict[int, dict[str, Any]], report_dir: Path, command: str, manifest: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for seed, result in results.items():
        c = result["summary"]["counts"]
        q = result["summary"]["quadrants"]
        table.append([seed, c["reliable_history"], c["current_vulnerable"], c["reliable_and_vulnerable"], c["observed_degradation"], c["reliable_and_observed_degradation"], c["current_errors"], c["future_flips"]])
    common = (
        f"\n- 运行命令：`{command}`\n"
        f"- 远端 main：`{manifest['remote_main']}`；本地工作树无 `.git`，因此不伪造本地 commit。\n"
        f"- 独立研究代码 SHA256：`{manifest['audit_script_sha256']}`。\n"
        f"- 数据来源和 SHA256：见 `RUN_MANIFEST.json`；固定 checkpoint SHA256：见已有 `outputs/ndp_triplet_alignment_20261021_134148/RUN_MANIFEST.json`。\n"
        "- 协议：只读 train-only 特征；update400 只作事后标签；无 optimizer.step、无官方 query/gallery。\n"
    )
    (report_dir / "NDP_RELIABILITY_VULNERABILITY_REDESIGN.md").write_text(
        "# NDP Reliability / Vulnerability / Degradation 重构\n\n"
        "本报告将 R、V、D 分为三个独立概念。R 只看 update0/20/50/100 四个预注册历史节点；V 只看 update200 的当前 margin、同 anchor 的多负 PID 尾部、当前错误 PID 和可观察双向不对称；D 为历史均值参考减当前 Margin。\n\n"
        + markdown_table(["seed", "R", "V", "R∩V", "D", "R∩D", "当前错误", "200→400翻转"], table)
        + "\n\n四个象限：\n\n"
        + markdown_table(["seed", "R高/V低", "R高/V高", "R低/V低", "R低/V高"], [[s, r["summary"]["quadrants"]["high_reliable_low_vulnerable"], r["summary"]["quadrants"]["high_reliable_high_vulnerable"], r["summary"]["quadrants"]["low_reliable_low_vulnerable"], r["summary"]["quadrants"]["low_reliable_high_vulnerable"]] for s, r in results.items()])
        + "\n\n关键判定：若 R∩V 非零，则互斥来自旧门控定义而非数据分布；若某 seed 的某一格为零，仅能解释为该 seed 在预注册阈值下没有落入该格。历史最小节点数固定为4，未用 update400 调阈值。\n"
        + common,
        encoding="utf-8",
    )

    candidate_rows = []
    for seed, result in results.items():
        rows = result["rows"]
        rmask = np.asarray([bool(x["reliable_history"]) for x in rows])
        for name, raw_key, active_key in [("A 固定 update0", "A_raw_hinge", "A_active"), ("B 历史最后可信节点", "B_raw_hinge", "B_active"), ("C 历史 Q25 安全边界", "C_raw_hinge", "C_active"), ("D 当前固定0.05", "D_raw_hinge", "D_active"), ("E Batch-Hard SoftMargin代理", "E_softmargin_value", "E_active")]:
            chosen = [x for i, x in enumerate(rows) if rmask[i]]
            candidate_rows.append([seed, name, len(chosen), sum(float(x[raw_key]) > 0 for x in chosen), f"{np.mean([float(x[raw_key]) for x in chosen]):.6g}" if chosen else "NA", f"{np.linalg.norm([float(x[raw_key]) for x in chosen]):.6g}" if chosen else "NA"])
    (report_dir / "NDP_HISTORICAL_SAFE_BOUNDARY_AUDIT.md").write_text(
        "# Historical Safe Boundary 候选审计\n\n"
        "预注册候选：`M_safe=max(0.05,Q25(M0,M20,M50,M100)-0.02)`；C 只对历史可靠关系计算。A 使用固定 update0 Margin，B 使用 update100，D 使用固定当前阈值0.05，E 使用 `softplus(-M_current)` 的 Batch-Hard 关系代理。A/B/C 的 reference 均停止梯度；当前改善（当前 Margin 高于目标）对应零 hinge。\n\n"
        + markdown_table(["seed", "候选", "R内选择", "R内激活", "平均原始值", "原始值L2"], candidate_rows)
        + "\n\nC 若与 A/B 只有激活边界不同，不能称为独立机制；必须同时查看目标激活集合、梯度范数和 P4 局部影响。结果未执行更新，因此不构成训练收益证据。\n" + common,
        encoding="utf-8",
    )

    flip_table = []
    for seed, result in results.items():
        for x in result["flip_rows"]:
            flip_table.append([seed, x["relation_index"], x["anchor_image_index"], x["anchor_pid"], x["anchor_clothes_id"], x["negative_pid"], x["M0"], x["M20"], x["M50"], x["M100"], x["M200"], x["M400"], x["reliable_history"], x["current_vulnerability"], x["safe_boundary_active"], x["bidirectional_pair_exists"], x["current_negative_pid_near_tail_count"], x["old_reliability_exclusion_reason"]])
    (report_dir / "NDP_MISSED_FLIP_ROOT_CAUSE.md").write_text(
        "# 200→400 翻转遗漏逐关系归因\n\n"
        "以下 CSV 是全部事后观察到的 `M200>0` 且 `M400<=0` 关系；这些行只用于回顾性诊断，不参与 update200 的 mask、阈值或排序。\n\n"
        + markdown_table(["seed", "rel", "anchor img", "anchor PID", "clothes", "negative PID", "M0", "M20", "M50", "M100", "M200", "M400", "新R", "V", "safe激活", "双向", "tail", "旧门控排除"], flip_table)
        + "\n\n归因类别定义：历史错误/不可靠关系不应强制恢复；历史可靠但当前已恶化属于 R∩V；历史可靠且当前尚未恶化、随后在400翻转属于事后不可用的 future-risk 类型；一直接近零或波动关系属于当前风险但不满足长期可靠。正负图像索引、衣物 ID 和完整关系信息保存在每个 seed 的 `missed_flip_rows_seed*.csv`。\n"
        + common,
        encoding="utf-8",
    )

    selection_table = []
    for seed, result in results.items():
        for kind in ("native", "budget_matched"):
            for method, stat in result["selection"][kind].items():
                selection_table.append([seed, kind, method, stat["selected"], stat["active_safe_boundary"], stat["future_flips"], stat["future_flip_coverage_rate_of_all"], stat["current_errors"], stat["anchor_pid_count"], stat["negative_pid_count"], f"{stat['negative_pid_exposure_gini']:.4f}"])
    (report_dir / "NDP_REDESIGNED_SELECTION_BENCHMARK.md").write_text(
        "# P3 修复后预算匹配关系选择\n\n"
        "候选池固定为每 seed 512 anchors × 5 negative PID = 2560 条。native 表保留方法本身的自然选中量；budget_matched 表以各 seed 非空 native 方法中的最小关系数为共同预算，并且 B0/B1/B2--B7 均不看 update400。B3--B7 的预算版是‘门控优先、当前 Margin 作 tie-break’的可比版本；原生 mask 仍单独报告，避免把补齐后的版本误称为纯交集。随机控制使用200个固定 RNG。\n\n"
        + markdown_table(["seed", "版本", "方法", "selected", "active", "flip", "覆盖全部flip", "当前错误", "anchor PID", "negative PID", "neg Gini"], selection_table)
        + "\n\n梯度范数只在同一 safe-boundary 关系代理上测量；未把少量 active 的 native mask 与大 active 集合直接做 raw norm 结论。原始和预算匹配统计的 bootstrap 95% CI 保存在 `selection_summary_seed*.json`。\n"
        + common,
        encoding="utf-8",
    )

    impact_table = []
    for seed, result in results.items():
        summary = result["impact"]["methods"]
        for method, stat in summary.items():
            impact_table.append([seed, method, stat["rows"], stat["target_mean_delta"], stat["other_mean_delta"], stat["other_harm_count"], stat["other_wrong_after_predicted"], stat["finite"]])
    (report_dir / "NDP_NEIGHBORHOOD_COLLATERAL_RISK_AUDIT.md").write_text(
        "# P4 邻域附带损害与双侧风险审计\n\n"
        "目标关系为新定义的 R∩V；每个 anchor 同时展开五个不同负 PID，方向 A→B 与 B→A 使用各自合法正样本。`predicted_delta_margin=-eta<grad(M),grad(L)>`，eta=1e-3；每个候选 loss 的梯度先归一到相同 L2 budget=1，因此这是统一预算的一阶方向诊断，不是网络更新。\n\n"
        + markdown_table(["seed", "method", "rows", "target ΔM均值", "其他 ΔM均值", "其他关系恶化数", "预测跨零数", "finite"], impact_table)
        + "\n\n`triplet_batchhard` 使用原始 raw feat[0] 的欧氏 Batch-Hard 局部代理；`margin_hinge_target` 是单侧固定阈值；`safe_boundary_target` 是历史边界；`multi_pid_safe` 同 anchor 五个 PID；`bidirectional_safe` 加入可匹配的反向身份关系。若目标改善同时伴随其他正确关系负 ΔM 或跨零，说明单侧保护存在 collateral risk。有限差分复核保存在每个 seed 的 JSON。\n"
        + common,
        encoding="utf-8",
    )

    json_dump(report_dir / "RUN_MANIFEST.json", manifest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--remote-main", default="d6e24867a9bf33423d3342682c96d053d37ff190")
    args = ap.parse_args()
    feature_base = Path("/home/DIFFER/outputs/ndp_margin_consistency_20261021_220823/feature_export")
    graph_base = Path("/home/DIFFER/outputs/ndp_true_multipid_20261021_103952/reference_graph")
    results: dict[int, dict[str, Any]] = {}
    for seed in (1234, 4321):
        results[seed] = audit_seed(
            seed,
            graph_base / f"seed{seed}/multi_pid_reference_graph.json",
            feature_base / f"seed{seed}/k5_trajectory_features_seed{seed}.npz",
            args.output_root,
        )
    manifest = {
        "run_id": args.run_id,
        "remote_main": args.remote_main,
        "python": "/usr/local/bin/python3.10",
        "torch": torch.__version__,
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "command": " ".join(["/usr/local/bin/python3.10", "-u", str(Path(__file__).resolve()), *[f"{k}={v}" for k, v in []]]),
        "pre_registered": PRE_REGISTERED,
        "seeds": [1234, 4321],
        "optimizer_step_executed": False,
        "official_query_gallery_used": False,
        "gpc_modified": False,
        "baseline_hash_manifest": "/home/DIFFER/outputs/ndp_margin_consistency_20261021_220823/RUN_MANIFEST.json",
    }
    command = f"/usr/local/bin/python3.10 -u {Path(__file__).resolve()} --run-id {args.run_id} --output-root {args.output_root} --report-dir {args.report_dir} --remote-main {args.remote_main}"
    manifest["command"] = command
    json_dump(args.output_root / "RUN_MANIFEST.json", manifest)
    write_reports(results, args.report_dir, command, manifest)
    print(json.dumps({seed: result["summary"]["counts"] for seed, result in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
