#!/usr/bin/env python3
"""P5 研究性软侵入损失；不接入正式训练。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


def _check(anchor, positive, negative):
    if anchor.ndim != 2 or positive.ndim != 3 or negative.ndim != 3:
        raise ValueError("anchor=[A,D], positive/negative=[A,M,D]")
    if anchor.shape[0] != positive.shape[0] or anchor.shape[0] != negative.shape[0]:
        raise ValueError("anchor 数量不一致")
    if positive.shape[1] == 0 or negative.shape[1] == 0:
        raise ValueError("正负邻居不能为空")


def soft_intrusion_score(anchor, positive, negative, tau=0.07):
    _check(anchor, positive, negative)
    if tau <= 0:
        raise ValueError("tau 必须大于0")
    a = F.normalize(anchor.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    n = F.normalize(negative.float(), dim=-1)
    pos = torch.einsum("ad,amd->am", a, p)
    neg = torch.einsum("ad,amd->am", a, n)
    u = torch.logsumexp(pos / tau, dim=1) - torch.log(torch.tensor(pos.shape[1], device=pos.device, dtype=pos.dtype))
    v = torch.logsumexp(neg / tau, dim=1) - torch.log(torch.tensor(neg.shape[1], device=neg.device, dtype=neg.dtype))
    # 用 logit 差构造 sigmoid 型软侵入质量，数值稳定且范围为(0,1)。
    return torch.sigmoid(v - u)


def soft_intrusion_loss(current_anchor, current_positive, current_negative,
                        reference_anchor, reference_positive, reference_negative,
                        tau=0.07, epsilon=0.0, reliable=None, weight=None):
    with torch.no_grad():
        ref = soft_intrusion_score(reference_anchor.detach(), reference_positive.detach(), reference_negative.detach(), tau)
    cur = soft_intrusion_score(current_anchor, current_positive, current_negative, tau)
    violation = F.relu(cur - ref - float(epsilon))
    if reliable is None:
        reliable = torch.ones_like(violation, dtype=torch.bool)
    else:
        reliable = reliable.to(device=violation.device, dtype=torch.bool)
    if weight is None:
        weight = torch.ones_like(violation)
    else:
        weight = weight.to(device=violation.device, dtype=violation.dtype).detach()
    selected = violation[reliable]
    selected_weight = weight[reliable]
    if selected.numel() == 0:
        loss = _zero(current_anchor)
    else:
        loss = (selected * selected_weight).sum() / selected_weight.sum().clamp_min(1e-12)
    return loss, {
        "reference_intrusion": ref.detach(),
        "current_intrusion": cur,
        "violation": violation,
        "raw_delta": cur - ref,
        "reliable": reliable.detach(),
        # violation 已经包含 epsilon 门槛；再次与 epsilon 比较会把真实
        # 条件 cur-ref>epsilon 错误收紧为 cur-ref>2*epsilon。
        "active_count": (violation[reliable] > 0).sum().detach(),
    }


def bidirectional_soft_intrusion_loss(*directions, **kwargs):
    """对已去重的有向关系求平均；调用方负责保证 (i,j)/(j,i) 去重策略。"""
    if not directions:
        raise ValueError("至少需要一个方向")
    losses = []
    infos = []
    for args in directions:
        loss, info = soft_intrusion_loss(*args, **kwargs)
        losses.append(loss); infos.append(info)
    return torch.stack(losses).mean(), {"directions": len(directions), "infos": infos}
