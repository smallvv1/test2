#!/usr/local/bin/python3.10
"""NDP-LRD（Local Relational Distillation）研究性候选。

这是独立的数学/梯度原型，不接入正式 DIFFER processor，不修改 GPC-v1、CE、
Triplet 或 Bio。当前特征保留梯度；参考特征、可靠性和激活门控均停止梯度。

核心区别：不是对单个正负对施加固定或自适应 margin，而是在固定的多邻居集合
上蒸馏参考局部关系分布，并只在参考可靠且当前正身份概率质量下降超过容忍度时
激活。单正单负时它退化为二类 softmax/KL 对照，不能据此宣称多邻居独立性。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _zero(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


def _as_3d(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim != 3 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} 必须是 [A,M,D] 且 A/M 非空，实际为 {tuple(value.shape)}")
    return value


def _mask(value: Optional[torch.Tensor], shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    result = value.to(device=device, dtype=torch.bool)
    if tuple(result.shape) != shape:
        raise ValueError(f"mask 形状错误：期望 {shape}，实际为 {tuple(result.shape)}")
    return result


def _logits(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, temperature: float, positive_mask: Optional[torch.Tensor], negative_mask: Optional[torch.Tensor]):
    if temperature <= 0:
        raise ValueError("temperature 必须大于0")
    if anchor.ndim != 2:
        raise ValueError(f"anchor 必须是 [A,D]，实际为 {tuple(anchor.shape)}")
    positive = _as_3d(positive, "positive")
    negative = _as_3d(negative, "negative")
    if positive.shape[0] != anchor.shape[0] or negative.shape[0] != anchor.shape[0]:
        raise ValueError("anchor/positive/negative 的 A 维不一致")
    a = F.normalize(anchor.float(), dim=-1, eps=1e-12)
    p = F.normalize(positive.float(), dim=-1, eps=1e-12)
    n = F.normalize(negative.float(), dim=-1, eps=1e-12)
    pos = torch.einsum("ad,amd->am", a, p) / float(temperature)
    neg = torch.einsum("ad,amd->am", a, n) / float(temperature)
    pm = _mask(positive_mask, tuple(pos.shape), pos.device)
    nm = _mask(negative_mask, tuple(neg.shape), neg.device)
    logits = torch.cat([pos, neg], dim=1)
    valid = torch.cat([pm, nm], dim=1)
    return logits, valid, pm, nm


def _distribution(logits: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    has = valid.any(dim=1)
    masked = logits.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(masked, dim=1)
    probs = torch.where(has[:, None], probs, torch.zeros_like(probs))
    return probs, has


def _group_logmeanexp(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    has = mask.any(dim=1)
    masked = logits.masked_fill(~mask, float("-inf"))
    count = mask.sum(dim=1).clamp_min(1).to(logits.dtype)
    values = torch.logsumexp(masked, dim=1) - count.log()
    return torch.where(has, values, torch.zeros_like(values)), has


def local_lrd_loss(
    current_anchor: torch.Tensor,
    current_positive: torch.Tensor,
    current_negative: torch.Tensor,
    reference_anchor: torch.Tensor,
    reference_positive: torch.Tensor,
    reference_negative: torch.Tensor,
    temperature: float = 0.07,
    epsilon_mass: float = 0.02,
    reference_reliability: float = 0.05,
    reference_positive_mass_min: float = 0.50,
    reliable_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """选择性局部关系分布蒸馏。

    对每个 anchor，将正/负邻居拼为一个关系分布 q。参考可靠条件为：

    * ``A_ref = LME(pos_ref) - LME(neg_ref) > reference_reliability``；
    * ``Q_ref = sum(q_ref[pos]) >= reference_positive_mass_min``；
    * 可选的外部 mask 为真。

    仅当 ``Q_ref - Q_cur > epsilon_mass`` 时激活。激活门控 detach，损失为
    ``KL(q_ref || q_cur)``。因此梯度作用于当前 anchor/邻居，参考分支无梯度，
    且多邻居概率质量和分布形状都会产生信号。空候选返回与当前 anchor 相连的0。
    """
    if epsilon_mass < 0 or reference_reliability < 0:
        raise ValueError("epsilon_mass/reference_reliability 必须非负")
    cur_logits, cur_valid, cur_pm, cur_nm = _logits(
        current_anchor, current_positive, current_negative, temperature, None, None,
    )
    ref_logits, ref_valid, ref_pm, ref_nm = _logits(
        reference_anchor.detach(), reference_positive.detach(), reference_negative.detach(), temperature, None, None,
    )
    # 当前/参考使用同一固定邻居索引；这里不接受两套 mask，避免偷偷改变关系集合。
    if cur_logits.shape != ref_logits.shape:
        raise ValueError("current/reference 邻居集合形状不一致")
    if reliable_mask is None:
        external = torch.ones(cur_logits.shape[0], dtype=torch.bool, device=cur_logits.device)
    else:
        external = reliable_mask.to(device=cur_logits.device, dtype=torch.bool)
        if external.ndim != 1 or external.shape[0] != cur_logits.shape[0]:
            raise ValueError("reliable_mask 必须是 [A]")
    q_cur, cur_has = _distribution(cur_logits, cur_valid)
    with torch.no_grad():
        q_ref, ref_has = _distribution(ref_logits, ref_valid)
    pos_count = current_positive.shape[1] if current_positive.ndim == 3 else 1
    pos_mass_cur = q_cur[:, :pos_count].sum(dim=1)
    pos_mass_ref = q_ref[:, :pos_count].sum(dim=1).detach()
    pos_cur_lme, pos_cur_has = _group_logmeanexp(cur_logits[:, :pos_count], cur_valid[:, :pos_count])
    neg_cur_lme, neg_cur_has = _group_logmeanexp(cur_logits[:, pos_count:], cur_valid[:, pos_count:])
    with torch.no_grad():
        pos_ref_lme, pos_ref_has = _group_logmeanexp(ref_logits[:, :pos_count], ref_valid[:, :pos_count])
        neg_ref_lme, neg_ref_has = _group_logmeanexp(ref_logits[:, pos_count:], ref_valid[:, pos_count:])
    adv_ref = (pos_ref_lme - neg_ref_lme).detach()
    reliable = external & cur_has & ref_has & pos_cur_has & neg_cur_has & pos_ref_has & neg_ref_has
    reliable = reliable & (adv_ref > float(reference_reliability)) & (pos_mass_ref >= float(reference_positive_mass_min))
    active = reliable & ((pos_mass_ref - pos_mass_cur).detach() > float(epsilon_mass))
    log_q_cur = q_cur.clamp_min(1e-8).log()
    q_ref_safe = q_ref.clamp_min(1e-8)
    per_anchor_kl = F.kl_div(log_q_cur, q_ref_safe, reduction="none").masked_fill(~ref_valid, 0.0).sum(dim=1)
    selected = per_anchor_kl[active]
    loss = selected.mean() if selected.numel() else _zero(current_anchor)
    return loss, {
        "reference_advantage": adv_ref,
        "current_advantage": pos_cur_lme - neg_cur_lme,
        "reference_positive_mass": pos_mass_ref,
        "current_positive_mass": pos_mass_cur,
        "per_anchor_kl": per_anchor_kl,
        "reliable_mask": reliable.detach(),
        "active_mask": active.detach(),
        "valid_anchor_count": reliable.sum().detach(),
        "active_anchor_count": active.sum().detach(),
    }


def lrd_from_logits(current_logits: torch.Tensor, reference_logits: torch.Tensor, positive_count: int, epsilon_mass: float = 0.02) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """供数学单测使用的 logits 版本；不涉及模型或数据。"""
    if current_logits.shape != reference_logits.shape or current_logits.ndim != 2:
        raise ValueError("current/reference logits 必须同形状 [A,M]")
    valid = torch.ones_like(current_logits, dtype=torch.bool)
    q_cur = torch.softmax(current_logits, dim=1)
    with torch.no_grad():
        q_ref = torch.softmax(reference_logits.detach(), dim=1)
    ref_mass = q_ref[:, :positive_count].sum(1)
    cur_mass = q_cur[:, :positive_count].sum(1)
    active = (ref_mass - cur_mass).detach() > epsilon_mass
    kl = F.kl_div(q_cur.clamp_min(1e-8).log(), q_ref.clamp_min(1e-8), reduction="none").sum(1)
    selected = kl[active]
    loss = selected.mean() if selected.numel() else _zero(current_logits)
    return loss, {"reference_positive_mass": ref_mass, "current_positive_mass": cur_mass, "active_mask": active, "per_anchor_kl": kl, "valid": valid}
