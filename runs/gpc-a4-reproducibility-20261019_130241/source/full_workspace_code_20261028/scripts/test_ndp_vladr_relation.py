#!/usr/bin/env python3
"""VLADR/NDP 原型的 CPU 数学单测；不加载模型、不触碰 GPC。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ndp_vladr_relation import (  # noqa: E402
    add_ndp_to_base,
    local_advantage_preservation,
    local_identity_advantage,
    multi_positive_supcon_control,
    row_relation_distribution,
    vladr_relation_kl,
)


def assert_finite(x: torch.Tensor, name: str) -> None:
    assert torch.isfinite(x).all().item(), name


def main() -> None:
    torch.manual_seed(20261021)
    d = 16
    a = torch.nn.functional.normalize(torch.randn(4, d), dim=1).requires_grad_()
    p = torch.nn.functional.normalize(torch.randn(4, 3, d), dim=2).requires_grad_()
    n = torch.nn.functional.normalize(torch.randn(4, 5, d), dim=2).requires_grad_()
    ref_a = a.detach().clone().requires_grad_()
    ref_p = p.detach().clone().requires_grad_()
    ref_n = n.detach().clone().requires_grad_()

    # 1) 关系矩阵：非对角线有效，逐行和为1；单样本安全。
    prob, valid = row_relation_distribution(a, temperature=0.05, exclude_diagonal=True)
    assert valid.all().item()
    assert torch.allclose(prob.sum(1), torch.ones(4), atol=1e-6)
    assert torch.allclose(torch.diagonal(prob), torch.zeros(4), atol=1e-8)
    single = torch.randn(1, d)
    single_prob, single_valid = row_relation_distribution(single, temperature=0.05, exclude_diagonal=True)
    assert not single_valid.any().item()
    assert torch.equal(single_prob, torch.zeros_like(single_prob))

    # 2) 官方方向：current == reference 时 KL 为0且参考不接收梯度。
    kl, kl_info = vladr_relation_kl(a, ref_a, temperature=0.05, exclude_diagonal=True)
    assert_finite(kl, "KL finite")
    assert abs(float(kl)) < 1e-6
    kl.backward()
    assert ref_a.grad is None

    # 3) 同状态 NDP 且 epsilon>0 不激活，当前梯度为0，参考不反向。
    a.grad = None
    ref_a.grad = None
    loss0, info0 = local_advantage_preservation(
        a, p, n, ref_a, ref_p, ref_n,
        temperature=0.2, epsilon=0.1, reference_reliability=0.0,
        reliable_mask=torch.ones(4, dtype=torch.bool),
    )
    assert abs(float(loss0)) < 1e-7
    loss0.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and float(a.grad.abs().sum()) == 0.0
    assert ref_a.grad is None

    # 4) 人工降低当前正负优势，NDP 激活且只对当前分支产生有限梯度。
    a2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    p2 = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]], requires_grad=True)
    n2 = torch.tensor([[[0.4, 0.916515, 0.0, 0.0], [0.3, 0.953939, 0.0, 0.0]]], requires_grad=True)
    ra2 = a2.detach().clone().requires_grad_()
    rp2 = p2.detach().clone().requires_grad_()
    rn2 = n2.detach().clone().requires_grad_()
    # 使当前正样本变差、负样本变近，得到严格正的违反量。
    cur_p2 = (p2 * 0.6 + n2 * 0.4).requires_grad_()
    cur_n2 = (n2 * 0.6 + p2 * 0.4).requires_grad_()
    loss2, info2 = local_advantage_preservation(
        a2, cur_p2, cur_n2, ra2, rp2, rn2,
        temperature=0.5, epsilon=0.01, reference_reliability=0.0,
        reliable_mask=torch.ones(1, dtype=torch.bool),
    )
    assert int(info2["active_anchor_count"]) == 1
    assert int(info2["violating_anchor_count"]) == 1
    assert float(loss2) > 0
    loss2.backward()
    assert_finite(loss2, "NDP loss finite")
    assert torch.isfinite(a2.grad).all() and float(a2.grad.abs().sum()) > 0
    assert ra2.grad is None and rp2.grad is None and rn2.grad is None

    # 5) 多邻居与单三元组路径均有限；空 mask 安全返回0。
    supcon, sup_info = multi_positive_supcon_control(a, p, n, temperature=0.2)
    assert int(sup_info["valid_anchor_count"]) == 4
    assert_finite(supcon, "SupCon control finite")
    empty = torch.zeros(4, 3, dtype=torch.bool)
    empty_loss, empty_info = local_advantage_preservation(
        a, p, n, ref_a, ref_p, ref_n, positive_mask=empty,
        negative_mask=torch.ones(4, 5, dtype=torch.bool), epsilon=0.0,
    )
    assert float(empty_loss) == 0.0
    assert int(empty_info["active_anchor_count"]) == 0

    # 6) lambda=0 必须保留原 base tensor 身份；非零只增加候选项。
    base = torch.tensor(3.0, requires_grad=True)
    ndp = torch.tensor(2.0, requires_grad=True)
    assert add_ndp_to_base(base, ndp, 0.0) is base
    assert abs(float(add_ndp_to_base(base, ndp, 0.1)) - 3.2) < 1e-6

    print("NDP_VLADR_RELATION_UNIT_TEST_PASS")


if __name__ == "__main__":
    main()
