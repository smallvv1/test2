#!/usr/local/bin/python3.10
"""NDP-LRD 数学、mask、停止梯度和独立性反例单测。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ndp_lrd_candidate import local_lrd_loss, lrd_from_logits  # noqa: E402


def main() -> None:
    torch.manual_seed(20261021)
    a = torch.nn.functional.normalize(torch.randn(3, 8), dim=1).requires_grad_()
    p = torch.nn.functional.normalize(torch.randn(3, 2, 8), dim=2).requires_grad_()
    n = torch.nn.functional.normalize(torch.randn(3, 3, 8), dim=2).requires_grad_()
    ra = a.detach().clone().requires_grad_()
    rp = p.detach().clone().requires_grad_()
    rn = n.detach().clone().requires_grad_()

    # 同状态且 epsilon>0 不激活；当前图仍保持可反向，参考分支无梯度。
    loss0, info0 = local_lrd_loss(a, p, n, ra, rp, rn, temperature=0.2, epsilon_mass=0.02, reference_reliability=0.0, reference_positive_mass_min=0.0)
    assert float(loss0) == 0.0 and int(info0["active_anchor_count"]) == 0
    loss0.backward()
    assert ra.grad is None and rp.grad is None and rn.grad is None
    assert a.grad is not None and torch.isfinite(a.grad).all()

    # 人工降低正身份概率质量；必须激活并给当前分支有限梯度。
    cur = torch.tensor([[2.0, 1.0, 0.5, 0.0]], requires_grad=True)
    ref = cur.detach().clone().requires_grad_()
    changed = torch.tensor([[0.0, 1.0, 2.0, 1.0]], requires_grad=True)
    loss1, info1 = lrd_from_logits(changed, ref, positive_count=2, epsilon_mass=0.02)
    assert int(info1["active_mask"].sum()) == 1 and float(loss1) > 0
    loss1.backward()
    assert changed.grad is not None and torch.isfinite(changed.grad).all()
    assert ref.grad is None

    # 严格反例：两个当前多邻居集合的 A=logmeanexp(pos)-logmeanexp(neg)
    # 均约为1，但正/负内部概率形状不同；自适应 margin 只看到同一个标量，
    # LRD 的 KL/概率质量梯度不同。
    ref_logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    cur_a = torch.tensor([[3.0, 2.0, 2.0, 1.0]], requires_grad=True)
    cur_b = torch.tensor([[4.0, 1.0, 3.0, 0.0]], requires_grad=True)
    la, ia = lrd_from_logits(cur_a, ref_logits, positive_count=2, epsilon_mass=0.0)
    lb, ib = lrd_from_logits(cur_b, ref_logits, positive_count=2, epsilon_mass=0.0)
    assert int(ia["active_mask"].sum()) == 1 and int(ib["active_mask"].sum()) == 1
    assert torch.allclose(cur_a[:, :2].logsumexp(1) - torch.log(torch.tensor(2.0)) - (cur_a[:, 2:].logsumexp(1) - torch.log(torch.tensor(2.0))), torch.tensor([1.0]), atol=1e-6)
    assert torch.allclose(cur_b[:, :2].logsumexp(1) - torch.log(torch.tensor(2.0)) - (cur_b[:, 2:].logsumexp(1) - torch.log(torch.tensor(2.0))), torch.tensor([1.0]), atol=1e-6)
    ga = torch.autograd.grad(la, cur_a, retain_graph=True)[0]
    gb = torch.autograd.grad(lb, cur_b, retain_graph=True)[0]
    assert not torch.allclose(ga, gb, atol=1e-6)

    # 空/无激活路径安全返回有限零。
    none, ni = lrd_from_logits(torch.zeros(2, 4, requires_grad=True), torch.zeros(2, 4), positive_count=2, epsilon_mass=0.02)
    assert float(none) == 0.0 and torch.isfinite(none)
    none.backward()
    print("NDP_LRD_UNIT_TEST_PASS")


if __name__ == "__main__":
    main()
