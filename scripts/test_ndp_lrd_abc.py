#!/usr/bin/env python3
"""LRD-A/B/C 的纯张量回归测试；不加载模型、不访问测试集。"""
from __future__ import annotations

import torch

from ndp_lrd_abc import lrd_a_positive_mass, lrd_b_full_kl, lrd_c_cross_identity_kl


def main() -> None:
    torch.manual_seed(123)
    ref_pos = torch.tensor([[0.80, 0.70], [0.60, 0.55]], dtype=torch.float32)
    ref_neg = torch.tensor([[0.20, 0.10, 0.00], [0.50, 0.45, 0.40]], dtype=torch.float32)
    cur_pos = ref_pos.clone().requires_grad_(True)
    cur_neg = ref_neg.clone().requires_grad_(True)

    # 同状态：三种损失均为零或数值零，且不应人为制造梯度。
    for fn in (lrd_a_positive_mass, lrd_b_full_kl, lrd_c_cross_identity_kl):
        cur_pos.grad = None
        cur_neg.grad = None
        kwargs = {"tau": 0.07}
        if fn is lrd_a_positive_mass:
            kwargs["epsilon"] = 0.02
        loss = fn(ref_pos, ref_neg, cur_pos, cur_neg, **kwargs)
        assert torch.isfinite(loss) and abs(float(loss)) < 1e-6, (fn.__name__, loss)

    # 只在正负组内部换序：A/C 不应感知组内置换，B 应感知完整关系变化。
    cur_pos2 = ref_pos[:, [1, 0]].clone().requires_grad_(True)
    cur_neg2 = ref_neg[:, [2, 1, 0]].clone().requires_grad_(True)
    a = lrd_a_positive_mass(ref_pos, ref_neg, cur_pos2, cur_neg2, tau=0.07)
    b = lrd_b_full_kl(ref_pos, ref_neg, cur_pos2, cur_neg2, tau=0.07)
    c = lrd_c_cross_identity_kl(ref_pos, ref_neg, cur_pos2, cur_neg2, tau=0.07)
    assert float(a) < 1e-6 and float(c) < 1e-6, (a, c)
    assert float(b) > 1e-6, "完整KL应看到固定关系槽位的内部重排，A/C不应看到"

    # 保持正负总质量近似不变，但负组内部重排，完整 KL 产生额外惩罚；A/C不惩罚组内形状。
    cur_neg3 = torch.tensor([[0.19, 0.11, 0.00], [0.48, 0.47, 0.40]], requires_grad=True)
    a3 = lrd_a_positive_mass(ref_pos, ref_neg, cur_pos, cur_neg3, tau=0.07)
    c3 = lrd_c_cross_identity_kl(ref_pos, ref_neg, cur_pos, cur_neg3, tau=0.07)
    b3 = lrd_b_full_kl(ref_pos, ref_neg, cur_pos, cur_neg3, tau=0.07)
    assert float(b3) > float(c3) + 1e-8, (a3, b3, c3)

    # 退化输入应激活，并且当前分支有有限梯度；参考分支不反传。
    cur_pos4 = torch.tensor([[0.30, 0.25], [0.60, 0.55]], requires_grad=True)
    cur_neg4 = torch.tensor([[0.65, 0.60, 0.55], [0.50, 0.45, 0.40]], requires_grad=True)
    loss4 = lrd_c_cross_identity_kl(ref_pos, ref_neg, cur_pos4, cur_neg4, tau=0.07)
    loss4.backward()
    assert float(loss4) > 0 and cur_pos4.grad is not None and torch.isfinite(cur_pos4.grad).all()
    assert cur_neg4.grad is not None and torch.isfinite(cur_neg4.grad).all()
    assert ref_pos.grad is None and ref_neg.grad is None
    print("LRD_ABC_UNIT_TEST_PASS")


if __name__ == "__main__":
    main()
