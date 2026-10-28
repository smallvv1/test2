#!/usr/bin/env python3
"""研究性 LRD-A/B/C 张量定义。

本文件只实现离线/单元测试所需的局部关系损失，不接入正式 DIFFER 训练。
输入为相同候选顺序下的正、负相似度 logits；参考分支始终 detach。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _check(pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[0] != neg.shape[0]:
        raise ValueError("pos/neg 必须为 [anchors, neighbors] 且 anchor 数一致")
    if pos.shape[1] == 0 or neg.shape[1] == 0:
        raise ValueError("每个 anchor 至少需要一个正样本和一个负样本")
    return pos.float(), neg.float()


def local_advantage(pos: torch.Tensor, neg: torch.Tensor, tau: float) -> torch.Tensor:
    pos, neg = _check(pos, neg)
    if tau <= 0:
        raise ValueError("tau 必须为正")
    return torch.logsumexp(pos / tau, dim=1) - torch.log(torch.tensor(pos.shape[1], device=pos.device)) - (
        torch.logsumexp(neg / tau, dim=1) - torch.log(torch.tensor(neg.shape[1], device=neg.device))
    )


def _probabilities(pos: torch.Tensor, neg: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos, neg = _check(pos, neg)
    logits = torch.cat([pos, neg], dim=1) / tau
    probs = torch.softmax(logits, dim=1)
    return probs, probs[:, : pos.shape[1]].sum(dim=1), probs[:, pos.shape[1] :].sum(dim=1)


def lrd_a_positive_mass(
    ref_pos: torch.Tensor,
    ref_neg: torch.Tensor,
    cur_pos: torch.Tensor,
    cur_neg: torch.Tensor,
    tau: float = 0.07,
    epsilon: float = 0.0,
    reliable: torch.Tensor | None = None,
) -> torch.Tensor:
    """LRD-A：只保护正身份概率总质量，不约束组内分布。"""
    _, ref_mass, _ = _probabilities(ref_pos.detach(), ref_neg.detach(), tau)
    _, cur_mass, _ = _probabilities(cur_pos, cur_neg, tau)
    raw = F.relu(ref_mass - cur_mass - epsilon)
    if reliable is not None:
        raw = raw[reliable.bool()]
    return raw.mean() if raw.numel() else cur_pos.sum() * 0.0


def lrd_b_full_kl(
    ref_pos: torch.Tensor,
    ref_neg: torch.Tensor,
    cur_pos: torch.Tensor,
    cur_neg: torch.Tensor,
    tau: float = 0.07,
    reliable: torch.Tensor | None = None,
) -> torch.Tensor:
    """LRD-B：完整 q_ref || q_cur，保留组内正负排序结构。"""
    ref_q, _, _ = _probabilities(ref_pos.detach(), ref_neg.detach(), tau)
    cur_q, _, _ = _probabilities(cur_pos, cur_neg, tau)
    raw = F.kl_div(cur_q.clamp_min(1e-12).log(), ref_q, reduction="none").sum(dim=1)
    if reliable is not None:
        raw = raw[reliable.bool()]
    return raw.mean() if raw.numel() else cur_pos.sum() * 0.0


def lrd_c_cross_identity_kl(
    ref_pos: torch.Tensor,
    ref_neg: torch.Tensor,
    cur_pos: torch.Tensor,
    cur_neg: torch.Tensor,
    tau: float = 0.07,
    reliable: torch.Tensor | None = None,
) -> torch.Tensor:
    """LRD-C：只蒸馏 [正身份质量, 异身份质量] 两组，不保持组内次序。"""
    _, ref_pos_mass, ref_neg_mass = _probabilities(ref_pos.detach(), ref_neg.detach(), tau)
    _, cur_pos_mass, cur_neg_mass = _probabilities(cur_pos, cur_neg, tau)
    ref_group = torch.stack([ref_pos_mass, ref_neg_mass], dim=1).clamp_min(1e-12)
    cur_group = torch.stack([cur_pos_mass, cur_neg_mass], dim=1).clamp_min(1e-12)
    raw = F.kl_div(cur_group.log(), ref_group, reduction="none").sum(dim=1)
    if reliable is not None:
        raw = raw[reliable.bool()]
    return raw.mean() if raw.numel() else cur_pos.sum() * 0.0


def detached_reference_gradient_check(loss: torch.Tensor, ref_pos: torch.Tensor, ref_neg: torch.Tensor) -> bool:
    """参考输入不应有梯度；当前分支梯度检查由调用方执行。"""
    return ref_pos.grad is None and ref_neg.grad is None and bool(torch.isfinite(loss).item())
