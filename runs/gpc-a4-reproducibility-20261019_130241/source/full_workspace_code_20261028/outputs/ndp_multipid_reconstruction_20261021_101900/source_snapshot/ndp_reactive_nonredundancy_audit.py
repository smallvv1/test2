#!/usr/bin/env python3
"""在固定多 PID 关系上比较反应式 NDP 与 Triplet 型 hinge。

这是冻结特征/边界张量的机制审计，不是网络训练，也不代表参数更新收益。由于离线
关系已经固定，无法重建真实 Batch-Hard miner；因此 T0/T1 的等价性会被明确记录。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ndp_reactive_prototype import reactive_boundary_loss


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if float(na) == 0 or float(nb) == 0:
        return None
    return float(torch.dot(a, b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid-rows", type=Path, required=True)
    ap.add_argument("--history-rows", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prev-update", type=int, default=200)
    ap.add_argument("--target-update", type=int, default=400)
    ap.add_argument("--history-class", default="A_STABLE_CORRECT")
    ap.add_argument("--triplet-margin", type=float, default=0.3)
    ap.add_argument("--epsilon", type=float, default=0.0)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.pid_rows.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with args.history_rows.open(encoding="utf-8", newline="") as fh:
        history = list(csv.DictReader(fh))
    rows = [r for r in rows if int(r["prev_update"]) == args.prev_update and int(r["target_update"]) == args.target_update]
    hist_keys = {(int(r["anchor"]), int(r["negative_pid"])): r["reliability_class"]
                 for r in history if int(r["update"]) == args.prev_update}
    if not rows:
        raise RuntimeError("指定区间没有固定关系")
    ref = torch.tensor([float(r["m_prev"]) for r in rows], dtype=torch.float64)
    cur = torch.tensor([float(r["m_target"]) for r in rows], dtype=torch.float64, requires_grad=True)
    all_mask = torch.ones(len(rows), dtype=torch.bool)
    reliable = torch.tensor([hist_keys.get((int(r["anchor"]), int(r["negative_pid"]))) == args.history_class for r in rows])
    # T0/T1 在固定关系上都只能实现为同一固定-margin hinge；没有真实 miner 轨迹，
    # 不把它冒充为完整 Batch-Hard 梯度。
    losses = {
        "T0_HARD_TRIPLET_FIXED_RELATION_PROXY": torch.relu(float(args.triplet_margin) - cur),
        "T1_FIXED_MARGIN_HINGE": torch.relu(float(args.triplet_margin) - cur),
        "T2_REFERENCE_ADAPTIVE_HINGE": torch.relu(ref - cur - float(args.epsilon)),
        "T3_REACTIVE_STABLE_HISTORY": torch.relu(ref[reliable] - cur[reliable] - float(args.epsilon)),
        "T4_DIRECTED_ALL_RELIABLE": torch.relu(ref[reliable] - cur[reliable] - float(args.epsilon)),
        "T5_MULTI_PID_ALL_RELATIONS": torch.relu(ref[all_mask] - cur[all_mask] - float(args.epsilon)),
    }
    results = []
    grads = {}
    for name, value in losses.items():
        active = value.detach() > 0
        scalar = value.mean() if value.numel() else cur.sum() * 0.0
        grad = torch.autograd.grad(scalar, cur, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(cur)
        grads[name] = grad.detach()
        results.append({"method": name, "n_relation": int(value.numel()), "active_relations": int(active.sum()),
                        "active_fraction": float(active.double().mean()) if value.numel() else 0.0,
                        "loss": float(scalar.detach()), "reliable_relations": int(reliable.sum())})
    base = grads["T0_HARD_TRIPLET_FIXED_RELATION_PROXY"]
    for row in results:
        row["gradient_cosine_vs_T0"] = cosine(grads[row["method"]], base)
    t3_keys = {(int(rows[i]["anchor"]), int(rows[i]["negative_pid"])) for i in range(len(rows)) if bool(reliable[i])}
    # 当前离线数据没有额外的 reciprocal-only 子集；T4 明确使用全部有向可靠关系，
    # 因而可审计地与 T3 比较，而不是用一个恒真的布尔表达式。
    t4_keys = {(int(rows[i]["anchor"]), int(rows[i]["negative_pid"])) for i in range(len(rows)) if bool(reliable[i])}
    out = {"status": "PASS_FROZEN_FEATURE_MECHANISM_AUDIT", "transition": f"{args.prev_update}->{args.target_update}",
           "results": results, "reliable_mask_count": int(reliable.sum()),
           "unique_negative_pid": len({int(r["negative_pid"]) for r in rows}),
           "relation_overlap": {"T3_T4_same_in_this_fixed_directed_audit": t3_keys == t4_keys,
                                "T3_T5_subset": t3_keys.issubset({(int(r["anchor"]), int(r["negative_pid"])) for r in rows}),
                                "T3_relation_keys": sorted([list(x) for x in t3_keys]),
                                "T4_relation_keys": sorted([list(x) for x in t4_keys])},
           "warning": "T0/T1为固定关系proxy；未获得真实Batch-Hard逐步miner与网络参数梯度，不能解释为训练因果收益。",
           "pid_rows": str(args.pid_rows), "history_rows": str(args.history_rows),
           "official_query_gallery_used": False, "optimizer_step_executed": False}
    (args.output_dir / "reactive_nonredundancy_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "reactive_nonredundancy_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
