"""统一的 NDP 局部身份 Margin 定义。

本模块只做独立关系数学，不修改 DIFFER+GPC 训练入口。参考值和当前值必须
使用同一个 mode、同一正负索引集合，避免把历史 mean Margin 与当前 single
sample Margin 混在一起。
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

MarginMode = Literal["q25", "mean", "single"]


def _finite_vector(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 1 or x.numel() == 0:
        raise ValueError(f"{name} 必须是非空一维张量")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} 含 NaN/Inf")
    return x


def aggregate_positive(similarities: torch.Tensor, mode: MarginMode) -> torch.Tensor:
    sims = _finite_vector(similarities, "positive similarities")
    if mode == "q25":
        return torch.quantile(sims, 0.25)
    if mode == "mean":
        return sims.mean()
    if mode == "single":
        return sims[0]
    raise ValueError(f"未知 positive aggregation: {mode}")


def aggregate_negative(similarities: torch.Tensor, mode: MarginMode) -> torch.Tensor:
    sims = _finite_vector(similarities, "negative similarities")
    if mode == "single":
        return sims[0]
    # A/B 都使用同一组负身份中的最危险样本；C 才是单正单负对照。
    return sims.max()


def relation_margin(
    anchor: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    mode: MarginMode,
) -> torch.Tensor:
    """计算一个有向 anchor→negative-PID 关系的可微 Margin。

    输入可以是未归一化特征；相似度在这里统一定义为 cosine。返回标量并
    保留 current 分支计算图。mode=q25/mean/single 分别对应：
    Q25(pos)-max(neg)、mean(pos)-max(neg)、pos[0]-neg[0]。
    """
    if anchor.ndim != 1 or positives.ndim != 2 or negatives.ndim != 2:
        raise ValueError("anchor/positives/negatives 形状应为 [D]/[P,D]/[N,D]")
    if positives.shape[1] != anchor.shape[0] or negatives.shape[1] != anchor.shape[0]:
        raise ValueError("特征维度不一致")
    if positives.shape[0] == 0 or negatives.shape[0] == 0:
        raise ValueError("正负集合不能为空")
    z_a = F.normalize(anchor.float(), dim=-1)
    z_p = F.normalize(positives.float(), dim=-1)
    z_n = F.normalize(negatives.float(), dim=-1)
    pos_sim = z_p @ z_a
    neg_sim = z_n @ z_a
    return aggregate_positive(pos_sim, mode) - aggregate_negative(neg_sim, mode)


def reactive_hinge(
    current_margin: torch.Tensor,
    reference_margin: torch.Tensor,
    mask: torch.Tensor | None = None,
    epsilon: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (loss, raw_hinge, active_mask)，参考分支停止梯度。"""
    if current_margin.shape != reference_margin.shape:
        raise ValueError("reference/current Margin 形状不一致")
    ref = reference_margin.detach().to(device=current_margin.device, dtype=current_margin.dtype)
    finite = torch.isfinite(current_margin) & torch.isfinite(ref)
    valid = finite if mask is None else (mask.detach().to(torch.bool) & finite)
    raw = torch.relu(ref - current_margin - float(epsilon))
    active = valid & (raw > 0)
    if not bool(valid.any()):
        safe = torch.where(torch.isfinite(current_margin), current_margin, torch.zeros_like(current_margin))
        return safe.sum() * 0.0, raw, active
    loss = raw[valid].mean()
    return loss, raw, active


__all__ = ["MarginMode", "aggregate_positive", "aggregate_negative", "relation_margin", "reactive_hinge"]
