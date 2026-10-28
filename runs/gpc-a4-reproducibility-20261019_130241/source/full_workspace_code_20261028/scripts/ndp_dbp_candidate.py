#!/usr/bin/env python3
"""NDP-DBP 研究性双尺度原型；不接入正式训练。"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def edge_margin(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    if pos_logits.ndim != 2 or neg_logits.ndim != 2 or pos_logits.shape[0] != neg_logits.shape[0]:
        raise ValueError("pos/neg 必须为 [anchors, neighbors]")
    if pos_logits.shape[1] == 0 or neg_logits.shape[1] == 0:
        raise ValueError("正负邻居不能为空")
    return pos_logits.min(dim=1).values - neg_logits.max(dim=1).values


def dbp_edge_loss(ref_pos, ref_neg, cur_pos, cur_neg, epsilon=0.02, reliable=None):
    with torch.no_grad():
        ref_edge = edge_margin(ref_pos.detach(), ref_neg.detach())
    cur_edge = edge_margin(cur_pos, cur_neg)
    if reliable is None:
        reliable = ref_edge > 0
    reliable = reliable.bool() & (ref_edge > 0)
    violation = F.relu(ref_edge - cur_edge - epsilon)
    selected = violation[reliable]
    loss = selected.mean() if selected.numel() else cur_pos.sum() * 0.0
    return loss, {"reference_edge": ref_edge.detach(), "current_edge": cur_edge, "violation": violation, "reliable": reliable, "active": (selected > 0).sum()}
