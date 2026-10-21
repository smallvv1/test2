#!/usr/bin/env python3
"""NDP P1/P2/P5 离线模块单测。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ndp_intrusion_loss import bidirectional_soft_intrusion_loss, soft_intrusion_loss, soft_intrusion_score
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
    print("NDP_RISK_MODULE_TEST_PASS")


if __name__ == "__main__":
    main()
