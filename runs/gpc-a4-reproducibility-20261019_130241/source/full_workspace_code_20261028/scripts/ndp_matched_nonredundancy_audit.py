#!/usr/bin/env python3
"""预算匹配的 NDP/Triplet 关系与冻结特征梯度审计。

本脚本不执行模型 forward、optimizer.step 或修改 GPC。它把两类证据分开：

* 200->400 的固定关系事件用于比较关系覆盖（这是回顾性诊断）；
* update0 冻结特征上的 autograd 只用于检查损失接口、梯度有限性和关系掩码
  的差异，不能解释为真实网络参数更新收益。

真实 Batch-Hard 覆盖来自 ndp_real_triplet_miner_coverage.py 的只读重建，
没有历史逐步 miner 日志时，结果必须标记 reconstructed/approximate。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1, eps=1e-12)


def feature_probe(features: np.ndarray, graph_rows, mask, mode: str):
    """在 update0 特征上做一次纯张量梯度检查。"""
    x = torch.tensor(features, dtype=torch.float32, requires_grad=True)
    anchors = torch.tensor([int(r["anchor_local_index"]) for r in graph_rows], dtype=torch.long)
    positives = torch.tensor([[int(v) for v in r["positive_local_indices"]] for r in graph_rows], dtype=torch.long)
    negatives = torch.tensor([[int(v) for v in r["negative_local_indices"]] for r in graph_rows], dtype=torch.long)
    a = F.normalize(x[anchors], dim=-1)
    p = F.normalize(x[positives], dim=-1)
    n = F.normalize(x[negatives], dim=-1)
    pos_s = cosine(a[:, None, :].expand_as(p), p)
    neg_s = cosine(a[:, None, :].expand_as(n), n)
    current = torch.quantile(pos_s, 0.25, dim=1) - neg_s.max(dim=1).values
    ref = torch.tensor([float(r["reference_margin"]) for r in graph_rows], dtype=torch.float32)
    m = torch.tensor(mask, dtype=torch.bool)
    if mode == "fixed_margin":
        raw = F.relu(torch.tensor(0.20) - current)
    elif mode == "reactive_margin":
        raw = F.relu(ref - current - 0.02)
    else:
        raise ValueError(mode)
    selected = raw[m]
    loss = selected.mean() if selected.numel() else current.sum() * 0.0
    grad = torch.autograd.grad(loss, x, allow_unused=True)[0]
    finite = bool(torch.isfinite(loss).item() and grad is not None and torch.isfinite(grad).all().item())
    return {
        "loss": float(loss.detach()),
        "active_relation_count": int((raw > 0).sum().item()),
        "selected_relation_count": int(m.sum().item()),
        "gradient_finite": finite,
        "gradient_norm": float(grad.norm().item()) if grad is not None else 0.0,
        "network_parameter_gradient": False,
        "feature_space_only": True,
        "reference_detached": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--transition-rows", type=Path, required=True)
    ap.add_argument("--reliability-rows", type=Path, required=True)
    ap.add_argument("--miner-rows", type=Path, required=True)
    ap.add_argument("--bidirectional-rows", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bank = np.load(args.features, allow_pickle=False)
    features = bank["features"].astype(np.float32)
    if features.ndim != 2 or features.shape[1] != 1024:
        raise RuntimeError("update0 feature shape is not [N,1024]")
    payload = read_json(args.graph)
    graph_rows = [r for r in payload["relations"] if r["graph_type"] == "difficult" and int(r["k_pid"]) == 5]
    graph_rows.sort(key=lambda r: (int(r["anchor_order"]), int(r["negative_pid"])))
    trans = [r for r in read_csv(args.transition_rows) if int(r["prev_update"]) == 200 and int(r["target_update"]) == 400]
    trans.sort(key=lambda r: int(r["relation_index"]))
    if len(graph_rows) != len(trans):
        raise RuntimeError(f"graph/transition count mismatch {len(graph_rows)} != {len(trans)}")
    # graph.reference_margin 是 update0 的参考；transition.prev_margin 是
    # update200 的当前状态。二者本来就不应相等，分别保留，不能把时间
    # 节点错配当作代码错误。
    reference_vs_200_max_abs_delta = max(
        abs(float(g["reference_margin"]) - float(t["prev_margin"]))
        for g, t in zip(graph_rows, trans)
    )

    rel = {int(r["relation_index"]): r for r in read_csv(args.reliability_rows)}
    miner = {(
        int(r["anchor"]), int(r["negative_pid"])
    ): r for r in read_csv(args.miner_rows)}
    bidi = read_csv(args.bidirectional_rows)
    bidir_indices = set()
    for r in bidi:
        bidir_indices.add(int(r["relation_index_a"]))
        bidir_indices.add(int(r["relation_index_b"]))

    masks = {}
    # B0 是实际 Batch-Hard 覆盖（当前只读重建的直接证据）。
    masks["B0_reconstructed_batch_hard"] = np.array([
        int(miner[(int(g["anchor_local_index"]), int(g["negative_pid"]))]["full_miner_selected"]) > 0
        for g in graph_rows
    ], dtype=bool)
    # B1/B4 是明确的 margin 代理，不能冒充原始 Triplet。
    prev_margin = np.array([float(t["prev_margin"]) for t in trans], dtype=np.float64)
    masks["B1_fixed_margin_proxy"] = np.ones(len(trans), dtype=bool)
    masks["B2_reference_margin_proxy"] = prev_margin <= 0.20
    masks["B3_random_count_matched_to_B5"] = np.zeros(len(trans), dtype=bool)
    masks["B3_random_count_matched_to_B6"] = np.zeros(len(trans), dtype=bool)
    masks["B3_random_count_matched_to_B7"] = np.zeros(len(trans), dtype=bool)
    masks["B4_current_hard_margin_proxy"] = prev_margin <= 0.05
    masks["B5_reliability_vulnerability"] = np.array([
        rel[int(t["relation_index"])] ["reliability"] == "HIGH_RELIABILITY" and
        rel[int(t["relation_index"])] ["vulnerability"] == "HIGH_VULNERABILITY"
        for t in trans
    ], dtype=bool)
    masks["B6_bidirectional_image_relation"] = np.array([i in bidir_indices for i in range(len(trans))], dtype=bool)
    masks["B7_reliability_plus_bidirectional"] = masks["B5_reliability_vulnerability"] & masks["B6_bidirectional_image_relation"]
    rng = np.random.default_rng(20261021 + args.seed)
    for source_name, random_name in (
        ("B5_reliability_vulnerability", "B3_random_count_matched_to_B5"),
        ("B6_bidirectional_image_relation", "B3_random_count_matched_to_B6"),
        ("B7_reliability_plus_bidirectional", "B3_random_count_matched_to_B7"),
    ):
        n = int(masks[source_name].sum())
        if n:
            masks[random_name][rng.choice(len(trans), size=n, replace=False)] = True

    events = np.array([t["new_negative_pid"] == "True" for t in trans], dtype=bool)
    relation_summary = []
    for name, mask in masks.items():
        relation_summary.append({
            "method": name, "relation_count": int(mask.sum()),
            "event_count_200_400": int((mask & events).sum()),
            "event_rate_within_method": float((mask & events).sum() / mask.sum()) if mask.sum() else None,
            "pid_count": len({int(trans[i]["anchor_pid"]) for i in np.flatnonzero(mask)}),
            "direct_triplet_relation_overlap": float((mask & masks["B0_reconstructed_batch_hard"]).sum() / mask.sum()) if mask.sum() else None,
        })
    with (args.output_dir / "matched_relation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(relation_summary[0])); w.writeheader(); w.writerows(relation_summary)

    gradient_summary = {}
    for name, mask in masks.items():
        if name == "B0_reconstructed_batch_hard":
            gradient_summary[name] = {"not_run": True, "reason": "no direct relation selected in reconstructed miner"}
            continue
        gradient_summary[name] = {
            "fixed_margin_update0": feature_probe(features, graph_rows, mask, "fixed_margin"),
            "reactive_margin_update0": feature_probe(features, graph_rows, mask, "reactive_margin"),
        }
    (args.output_dir / "matched_gradient_summary.json").write_text(json.dumps(gradient_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS_BUDGET_MATCHED_OFFLINE_AUDIT",
        "seed": args.seed,
        "features_sha256": sha256(args.features), "graph_sha256": sha256(args.graph),
        "transition_sha256": sha256(args.transition_rows), "reliability_sha256": sha256(args.reliability_rows),
        "miner_sha256": sha256(args.miner_rows), "bidirectional_sha256": sha256(args.bidirectional_rows),
        "relation_count": len(trans),
        "update0_reference_vs_update200_max_abs_delta": reference_vs_200_max_abs_delta,
        "masks": relation_summary, "gradient_probe": "update0 frozen-feature autograd only; no network parameter gradient",
        "real_batch_hard_history_available": False, "reconstructed_sampler_only": True,
        "future_event_used_for_mask_definition": False,
        "official_query_gallery_used": False, "optimizer_step_executed": False,
        "interpretation": "B0 is actual-code semantic reconstruction; B1-B7 are budget/mask diagnostics, not training efficacy evidence",
    }
    (args.output_dir / "matched_nonredundancy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
