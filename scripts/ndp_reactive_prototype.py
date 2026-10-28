#!/usr/bin/env python3
"""反应式 NDP 的独立数学原型。

本模块只定义冻结参考关系上的单侧边界保护，不接入 DIFFER 训练入口，也不
执行 optimizer.step。关系是有向的：A→B 与 B→A 必须作为两条独立记录传入，
函数不会把它们错误合并成一个无向关系。

对每条可靠关系 r：

    h_r = relu(M_ref(r) - M_cur(r) - epsilon)

只有当前边界相对历史可靠边界过度下降时才产生梯度。边界改善时 h_r=0，
因此该原型不会把特征强行拉回历史状态。参考边界和权重均停止梯度。
"""
from __future__ import annotations

from typing import Optional

import torch


def reactive_boundary_loss(
    current_margin: torch.Tensor,
    reference_margin: torch.Tensor,
    reliable_mask: Optional[torch.Tensor] = None,
    epsilon: float = 0.0,
    weights: Optional[torch.Tensor] = None,
    max_weight: Optional[float] = None,
) -> torch.Tensor:
    """计算有向可靠关系的单侧反应式边界保护损失。

    Args:
        current_margin: 当前模型的 M(i,p,n)，形状 [R]，应保留计算图。
        reference_margin: 冻结参考状态的 M(i,p,n)，形状 [R]。
        reliable_mask: 预先冻结的可靠关系掩码；缺省为全部有效。
        epsilon: 允许的边界下降量，单位与 margin 相同。
        weights: 可选的 stop-gradient 可靠性权重，形状 [R]。
        max_weight: 可选的权重上限，用于限制 hub/关系重复造成的单条关系支配。

    Returns:
        标量有限损失。没有有效关系时返回与 current_margin 连接但数值为零的
        张量，保证可以安全反向而不产生 NaN。
    """
    if current_margin.ndim != 1 or reference_margin.ndim != 1:
        raise ValueError("current_margin/reference_margin 必须是一维 [R] 张量")
    if current_margin.shape != reference_margin.shape:
        raise ValueError("current_margin 与 reference_margin 形状不一致")
    if epsilon < 0:
        raise ValueError("epsilon 必须非负")

    ref = reference_margin.detach().to(dtype=current_margin.dtype, device=current_margin.device)
    finite = torch.isfinite(current_margin) & torch.isfinite(ref)
    if reliable_mask is None:
        mask = torch.ones_like(finite, dtype=torch.bool)
    else:
        if reliable_mask.shape != current_margin.shape:
            raise ValueError("reliable_mask 形状不一致")
        mask = reliable_mask.detach().to(device=current_margin.device, dtype=torch.bool)

    if weights is None:
        w = torch.ones_like(current_margin)
    else:
        if weights.shape != current_margin.shape:
            raise ValueError("weights 形状不一致")
        w = weights.detach().to(dtype=current_margin.dtype, device=current_margin.device)
        finite = finite & torch.isfinite(w)
        w = torch.clamp(w, min=0)
        if max_weight is not None:
            if max_weight <= 0:
                raise ValueError("max_weight 必须为正")
            w = torch.clamp(w, max=max_weight)

    active = mask & finite
    if not bool(active.any()):
        # current_margin 可能含 NaN/Inf；不能用 sum()*0 作为安全零值，
        # 因为 NaN*0 仍为 NaN。用有限元素构造与计算图相连的零。
        safe_current = torch.where(torch.isfinite(current_margin), current_margin, torch.zeros_like(current_margin))
        return safe_current.sum() * 0.0

    hinge = torch.relu(ref - current_margin - float(epsilon))[active]
    selected_w = w[active]
    weight_sum = selected_w.sum()
    if not bool(torch.isfinite(weight_sum)) or float(weight_sum.detach().cpu()) <= 0:
        safe_current = torch.where(torch.isfinite(current_margin), current_margin, torch.zeros_like(current_margin))
        return safe_current.sum() * 0.0
    # 采用加权平均，避免关系数量或 hubness 重复直接改变损失尺度。
    return (hinge * selected_w).sum() / weight_sum


def directed_reactive_boundary_loss(
    current_margin: torch.Tensor,
    reference_margin: torch.Tensor,
    reliable_mask: Optional[torch.Tensor] = None,
    epsilon: float = 0.0,
    weights: Optional[torch.Tensor] = None,
    max_weight: Optional[float] = None,
) -> torch.Tensor:
    """有向关系接口别名，明确保留 A→B 与 B→A 的独立梯度。"""
    return reactive_boundary_loss(
        current_margin=current_margin,
        reference_margin=reference_margin,
        reliable_mask=reliable_mask,
        epsilon=epsilon,
        weights=weights,
        max_weight=max_weight,
    )


__all__ = ["reactive_boundary_loss", "directed_reactive_boundary_loss"]
