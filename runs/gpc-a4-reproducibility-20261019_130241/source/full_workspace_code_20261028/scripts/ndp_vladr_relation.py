#!/usr/bin/env python3
"""独立的 VLADR 关系保持 / NDP 研究原型。

本文件不导入、不修改 DIFFER 的 GPC、Triplet、Bio 或训练 processor。
它只提供可单测的张量函数，供 train-only 诊断和后续受控探针使用。

实现分成三层：

* ``vladr_relation_kl``：复现官方 VLADR 的关系分布蒸馏对照；
* ``adaptive_margin_relation_loss``：单正/单负的自适应 margin 对照；
* ``local_advantage_preservation``：NDP 候选，比较多正、多负邻域的
  log-mean-exp 身份优势，参考分支永不反向传播。

所有当前特征在 FP32 中计算，参考特征显式 detach。空候选或没有可靠
anchor 时返回与当前图相连的零，保证正式 processor 未来接入时不会出现
无梯度的 ``torch.tensor(0.)`` 图断裂。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _zero_from(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


def _check_2d(x: torch.Tensor, name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} 必须是 [N,D]，实际为 {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise ValueError(f"{name} 不能为空；空集合请由上层 mask 跳过")


def _normalize(x: torch.Tensor) -> torch.Tensor:
    # 统一在 FP32 计算相似度，避免 AMP 下参考/当前分支的 dtype 不一致。
    return F.normalize(x.float(), dim=-1, eps=1e-12)


def _as_neighbor_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    """将 [A,D] 变为 [A,1,D]，保留 [A,M,D]。"""
    if x.ndim == 2:
        _check_2d(x, name)
        return x.unsqueeze(1)
    if x.ndim == 3:
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError(f"{name} 的 anchor/neighbor 维不能为空")
        return x
    raise ValueError(f"{name} 必须是 [A,D] 或 [A,M,D]，实际为 {tuple(x.shape)}")


def row_relation_distribution(
    features: torch.Tensor,
    temperature: float = 0.005,
    exclude_diagonal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造逐行关系概率分布。

    返回 ``(probabilities, valid_rows)``。官方 VLADR 实现不排除对角线；
    NDP 默认排除对角线，避免自相似把身份关系保持变成恒等项。
    """
    _check_2d(features, "features")
    if temperature <= 0:
        raise ValueError("temperature 必须大于0")
    z = _normalize(features)
    logits = z @ z.t() / float(temperature)
    valid = torch.ones_like(logits, dtype=torch.bool)
    if exclude_diagonal:
        valid.fill_diagonal_(False)
    row_has_value = valid.any(dim=1)
    masked = logits.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(masked, dim=1)
    # 单样本等空行在 softmax 后可能成为 NaN；把它们安全置零。
    probs = torch.where(row_has_value.unsqueeze(1), probs, torch.zeros_like(probs))
    return probs, row_has_value


def vladr_relation_kl(
    current: torch.Tensor,
    reference: torch.Tensor,
    temperature: float = 0.005,
    exclude_diagonal: bool = True,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VLADR 风格 ``KL(p_current || p_reference)``。

    官方代码用 ``KLDivLoss(batchmean)``，输入 ``log(p_new)``、target
    ``p_old.detach()``；本函数沿用这一方向。参考特征及其分布完全停止梯度。
    """
    _check_2d(current, "current")
    _check_2d(reference, "reference")
    if current.shape != reference.shape:
        raise ValueError("current/reference shape 不一致")
    p_cur, valid_cur = row_relation_distribution(current, temperature, exclude_diagonal)
    with torch.no_grad():
        p_ref, valid_ref = row_relation_distribution(reference.detach(), temperature, exclude_diagonal)
    valid = valid_cur & valid_ref
    if not bool(valid.any()):
        loss = _zero_from(current)
    else:
        per_row = F.kl_div(
            p_cur.clamp_min(1e-8).log(),
            p_ref.clamp_min(1e-8),
            reduction="none",
        ).sum(dim=1)
        loss = per_row[valid].mean()
    return loss, {
        "active_rows": valid.sum().detach(),
        "row_count": torch.tensor(current.shape[0], device=current.device),
        "current_distribution": p_cur,
        "reference_distribution": p_ref.detach(),
    }


def _masked_logmeanexp(values: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 [A,M] 值做 mask 后 log-mean-exp，返回值和有效 anchor。"""
    if values.ndim != 2:
        raise ValueError("values 必须是 [A,M]")
    if mask is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    else:
        valid = mask.to(device=values.device, dtype=torch.bool)
        if valid.shape != values.shape:
            raise ValueError("邻居 mask 形状不匹配")
    has = valid.any(dim=1)
    masked = values.masked_fill(~valid, float("-inf"))
    count = valid.sum(dim=1).clamp_min(1).to(values.dtype)
    out = torch.logsumexp(masked, dim=1) - count.log()
    out = torch.where(has, out, torch.zeros_like(out))
    return out, has


def local_identity_advantage(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = 0.07,
    positive_mask: Optional[torch.Tensor] = None,
    negative_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算多邻居身份优势 A=logmeanexp(pos)-logmeanexp(neg)。"""
    _check_2d(anchor, "anchor")
    pos = _as_neighbor_tensor(positive, "positive")
    neg = _as_neighbor_tensor(negative, "negative")
    if pos.shape[0] != anchor.shape[0] or neg.shape[0] != anchor.shape[0]:
        raise ValueError("anchor 与邻居 anchor 数不一致")
    if temperature <= 0:
        raise ValueError("temperature 必须大于0")
    a = _normalize(anchor)
    p = _normalize(pos)
    n = _normalize(neg)
    pos_sim = torch.einsum("ad,amd->am", a, p) / float(temperature)
    neg_sim = torch.einsum("ad,amd->am", a, n) / float(temperature)
    pos_lme, pos_has = _masked_logmeanexp(pos_sim, positive_mask)
    neg_lme, neg_has = _masked_logmeanexp(neg_sim, negative_mask)
    return pos_lme - neg_lme, pos_has & neg_has


def local_advantage_preservation(
    current_anchor: torch.Tensor,
    current_positive: torch.Tensor,
    current_negative: torch.Tensor,
    reference_anchor: torch.Tensor,
    reference_positive: torch.Tensor,
    reference_negative: torch.Tensor,
    temperature: float = 0.07,
    epsilon: float = 0.02,
    reference_reliability: float = 0.05,
    reliable_mask: Optional[torch.Tensor] = None,
    positive_mask: Optional[torch.Tensor] = None,
    negative_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """NDP 候选：只保护可靠参考下跌超过 epsilon 的多邻居身份优势。

    ``reference_*`` 只参与无梯度的可靠性和目标计算；当前分支保留完整
    梯度。由于优势在相似度除以 temperature 后的 log-mean-exp 域中，
    epsilon 的单位也是该优势域，不能直接把 cosine 域的 0.02 生搬过来。
    """
    if epsilon < 0 or reference_reliability < 0:
        raise ValueError("epsilon/reliability 必须非负")
    with torch.no_grad():
        ref_adv, ref_has = local_identity_advantage(
            reference_anchor.detach(), reference_positive.detach(), reference_negative.detach(),
            temperature, positive_mask, negative_mask,
        )
    cur_adv, cur_has = local_identity_advantage(
        current_anchor, current_positive, current_negative,
        temperature, positive_mask, negative_mask,
    )
    if reliable_mask is None:
        reliable = ref_has & (ref_adv > float(reference_reliability))
    else:
        reliable = reliable_mask.to(device=cur_adv.device, dtype=torch.bool) & ref_has & cur_has
    reliable = reliable & cur_has
    violation = F.relu(ref_adv.detach() - cur_adv - float(epsilon))
    selected = violation[reliable]
    loss = selected.mean() if selected.numel() else _zero_from(current_anchor)
    return loss, {
        "reference_advantage": ref_adv.detach(),
        "current_advantage": cur_adv,
        "violation": violation,
        "reliable_mask": reliable.detach(),
        "active_anchor_count": reliable.sum().detach(),
        "violating_anchor_count": (selected > 0).sum().detach(),
    }


def adaptive_margin_relation_loss(
    current_margin: torch.Tensor,
    reference_margin: torch.Tensor,
    epsilon: float = 0.02,
    reliable_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """与单三元组/自适应 margin 对照使用的纯标量版本。"""
    if current_margin.shape != reference_margin.shape or current_margin.ndim != 1:
        raise ValueError("margin 必须是相同形状的 [A]")
    if reliable_mask is None:
        reliable = reference_margin.detach() > 0.05
    else:
        reliable = reliable_mask.bool()
    violation = F.relu(reference_margin.detach() - current_margin - float(epsilon))
    selected = violation[reliable]
    loss = selected.mean() if selected.numel() else _zero_from(current_margin)
    return loss, {"reliable_mask": reliable, "violation": violation, "active_count": reliable.sum()}


def multi_positive_supcon_control(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = 0.07,
    positive_mask: Optional[torch.Tensor] = None,
    negative_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """固定相同多邻居集合的 SupCon/N-pair 风格控制。

    它不使用参考分支，目标是比较 NDP 的参考单侧保护是否只是普通多正
    样本对比学习的重加权。
    """
    _check_2d(anchor, "anchor")
    pos = _as_neighbor_tensor(positive, "positive")
    neg = _as_neighbor_tensor(negative, "negative")
    a = _normalize(anchor)
    p = _normalize(pos)
    n = _normalize(neg)
    pos_sim = torch.einsum("ad,amd->am", a, p) / float(temperature)
    neg_sim = torch.einsum("ad,amd->am", a, n) / float(temperature)
    pos_lme, pos_has = _masked_logmeanexp(pos_sim, positive_mask)
    neg_lse, neg_has = _masked_logmeanexp(neg_sim, negative_mask)
    valid = pos_has & neg_has
    value = F.softplus(neg_lse - pos_lme)
    loss = value[valid].mean() if bool(valid.any()) else _zero_from(anchor)
    return loss, {"valid_anchor_count": valid.sum().detach(), "per_anchor": value}


def add_ndp_to_base(base_loss: torch.Tensor, ndp_loss: torch.Tensor, lambda_ndp: float) -> torch.Tensor:
    """独立组合接口；lambda=0 返回同一个 base tensor，便于回归测试。"""
    if lambda_ndp == 0.0:
        return base_loss
    return base_loss + float(lambda_ndp) * ndp_loss
