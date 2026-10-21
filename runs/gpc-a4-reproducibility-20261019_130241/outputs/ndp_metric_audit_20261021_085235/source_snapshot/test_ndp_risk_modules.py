#!/usr/bin/env python3
"""NDP P1/P2/P5 离线模块单测。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ndp_intrusion_loss import bidirectional_soft_intrusion_loss, soft_intrusion_loss, soft_intrusion_score
from ndp_p2_p5_consistency import p2_score, p5_score
from ndp_risk_benchmark import auc_rank, average_precision, top_fraction_metrics
from ndp_risk_common import positive_indices, top_negative_indices


def main():
    pids = np.array([0, 0, 1, 2])
    sims = np.array([1.0, .8, .7, .6])
    assert 0 not in positive_indices(sims, 0, pids, 4)
    assert not set(top_negative_indices(sims, 0, pids, 3)) & {0, 1}
    assert len(top_negative_indices(sims, 0, pids, 3)) == 2

    torch.manual_seed(7)
    ref_a = torch.randn(2, 8)
    ref_p = torch.randn(2, 2, 8)
    ref_n = torch.randn(2, 3, 8)
    cur_a = ref_a.clone().requires_grad_(True)
    cur_p = ref_p.clone().requires_grad_(True)
    cur_n = ref_n.clone().requires_grad_(True)
    loss, info = soft_intrusion_loss(cur_a, cur_p, cur_n, ref_a, ref_p, ref_n, tau=.2, epsilon=.01)
    assert float(loss) == 0.0
    loss.backward()
    assert torch.isfinite(cur_a.grad).all() and torch.isfinite(cur_p.grad).all() and torch.isfinite(cur_n.grad).all()
    assert ref_a.grad is None and ref_p.grad is None and ref_n.grad is None

    # 人工增强异身份相似度，必须产生非零软侵入梯度。
    cur2_a = ref_a.clone().requires_grad_(True)
    cur2_p = ref_p.clone().requires_grad_(True)
    cur2_n = (ref_n * .2 + ref_a[:, None, :] * .8).requires_grad_(True)
    loss2, info2 = soft_intrusion_loss(cur2_a, cur2_p, cur2_n, ref_a, ref_p, ref_n, tau=.2, epsilon=0.0)
    assert float(loss2) >= 0.0
    loss2.backward()
    assert torch.isfinite(cur2_a.grad).all() and int(info2["active_count"]) >= 0

    # 两个方向均有限；不强制把重复无向关系再计一次，由调用方传入去重后的方向。
    bi, bi_info = bidirectional_soft_intrusion_loss(
        (cur_a.detach().requires_grad_(True), ref_p, ref_n, ref_a, ref_p, ref_n),
        (cur2_a.detach().requires_grad_(True), ref_p, ref_n, ref_a, ref_p, ref_n),
        tau=.2,
    )
    assert torch.isfinite(bi) and bi_info["directions"] == 2
    assert soft_intrusion_score(ref_a, ref_p, ref_n, .2).shape == (2,)

    # active_count 必须对应 raw_delta > epsilon，而不是错误的 > 2*epsilon。
    one_a = torch.tensor([[1.0, 0.0]])
    one_p = torch.tensor([[[1.0, 0.0]]])
    one_n = torch.tensor([[[0.0, 1.0]]])
    probe, pinfo = soft_intrusion_loss(one_a, one_p, one_n, one_a, one_p, one_n, tau=.2, epsilon=.01)
    assert float(probe) == 0.0 and int(pinfo["active_count"]) == 0

    # ROC-AUC: 理想、反向、完全并列，检查并列平均秩。
    assert abs(auc_rank(np.array([1, 1, 0, 0]), np.array([4., 3., 2., 1.])) - 1.0) < 1e-12
    assert abs(auc_rank(np.array([1, 1, 0, 0]), np.array([1., 2., 3., 4.])) - 0.0) < 1e-12
    assert abs(auc_rank(np.array([1, 0, 1, 0]), np.ones(4)) - 0.5) < 1e-12
    assert abs(average_precision(np.array([1, 0, 1, 0]), np.ones(4)) - .5) < 1e-12
    metrics = top_fraction_metrics(np.array([1, 0, 1, 0]), np.ones(4), .5)
    assert metrics["precision"] == .5 and metrics["recall"] == .5
    assert abs(auc_rank(np.array([1, 0, 1, 0]), np.array([1., np.nan, 0., -1.])) - 1.0) < 1e-12

    # P2 与 P5 的明确反例：一个负身份越过边界，但其余安全负身份
    # 足以降低聚合平均风险；因此 P2/P5 都不能被当作 edge 标签的同义词。
    pos_ref = np.full(4, .80)
    neg_ref = np.full(20, .70)
    pos_cur = np.full(4, .90)
    neg_cur = np.asarray([.92] + [.10] * 19)
    u_ref, p2_ref = p2_score(pos_ref, neg_ref, .07)
    u_cur, p2_cur = p2_score(pos_cur, neg_cur, .07)
    p5_ref = p5_score(pos_ref, neg_ref, .07)
    p5_cur = p5_score(pos_cur, neg_cur, .07)
    assert u_ref <= .8 and u_cur > u_ref
    assert pos_ref.min() - neg_ref.max() > 0 and pos_cur.min() - neg_cur.max() <= 0
    assert p2_cur < p2_ref and p5_cur < p5_ref
    print("NDP_RISK_MODULE_TEST_PASS")


if __name__ == "__main__":
    main()
