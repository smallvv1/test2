#!/usr/bin/env python3
"""Diagnostic-only NDP local structure preservation losses.

This module is deliberately independent from the released DIFFER+GPC loss.
It provides the two candidate penalties used by the NDP design pilot:

* margin preservation: preserve a reliable reference positive-vs-negative
  margin up to an allowed tolerance;
* negative intrusion: prevent a reliable reference negative similarity from
  increasing beyond its tolerance.

Reference tensors are detached by construction. Current tensors remain in the
autograd graph. The functions do not select neighbours or GPC pairs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _validate_pair_inputs(
    current_anchor: torch.Tensor,
    current_positive: torch.Tensor,
    current_negative: torch.Tensor,
    reference_anchor: torch.Tensor,
    reference_positive: torch.Tensor,
    reference_negative: torch.Tensor,
) -> None:
    tensors = (
        current_anchor,
        current_positive,
        current_negative,
        reference_anchor,
        reference_positive,
        reference_negative,
    )
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("NDP features must have shape [N, D]")
    shape = current_anchor.shape
    if any(t.shape != shape for t in (current_positive, current_negative)):
        raise ValueError("current anchor/positive/negative shapes differ")
    if any(t.shape != shape for t in (reference_anchor, reference_positive, reference_negative)):
        raise ValueError("reference and current feature shapes differ")


def cosine_similarity_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return row-wise cosine similarity in FP32 without detaching input."""
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("cosine_similarity_rows expects equal [N, D] tensors")
    return F.cosine_similarity(left.float(), right.float(), dim=1)


def reference_margin(
    reference_anchor: torch.Tensor,
    reference_positive: torch.Tensor,
    reference_negative: torch.Tensor,
) -> torch.Tensor:
    """Compute detached reference positive-minus-negative margins."""
    with torch.no_grad():
        return cosine_similarity_rows(reference_anchor, reference_positive) - cosine_similarity_rows(
            reference_anchor, reference_negative
        )


def current_margin(
    current_anchor: torch.Tensor,
    current_positive: torch.Tensor,
    current_negative: torch.Tensor,
) -> torch.Tensor:
    """Compute current positive-minus-negative margins with gradients."""
    return cosine_similarity_rows(current_anchor, current_positive) - cosine_similarity_rows(
        current_anchor, current_negative
    )


def ndp_margin_loss(
    current_anchor: torch.Tensor,
    current_positive: torch.Tensor,
    current_negative: torch.Tensor,
    reference_anchor: torch.Tensor,
    reference_positive: torch.Tensor,
    reference_negative: torch.Tensor,
    epsilon: float = 0.0,
    reliable_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve reliable reference margins up to ``epsilon``.

    The reduction is over the selected reliable rows only. If no row is
    selected, a differentiable zero is returned and ``active_count`` is zero.
    """
    _validate_pair_inputs(
        current_anchor,
        current_positive,
        current_negative,
        reference_anchor,
        reference_positive,
        reference_negative,
    )
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    ref_m = reference_margin(reference_anchor, reference_positive, reference_negative)
    cur_m = current_margin(current_anchor, current_positive, current_negative)
    mask = torch.ones_like(ref_m, dtype=torch.bool) if reliable_mask is None else reliable_mask.bool()
    if mask.shape != ref_m.shape:
        raise ValueError("reliable_mask must have shape [N]")
    violation = F.relu(ref_m - cur_m - float(epsilon))
    selected = violation[mask]
    loss = selected.mean() if selected.numel() else current_anchor.sum() * 0.0
    return loss, {
        "reference_margin": ref_m.detach(),
        "current_margin": cur_m,
        "violation": violation,
        "active_count": mask.sum().detach(),
        "violating_count": (selected > 0).sum().detach(),
    }


def ndp_negative_intrusion_loss(
    current_anchor: torch.Tensor,
    current_negative: torch.Tensor,
    reference_anchor: torch.Tensor,
    reference_negative: torch.Tensor,
    epsilon_negative: float = 0.0,
    reliable_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize increases in reliable anchor-negative similarity."""
    if current_anchor.ndim != 2 or current_negative.ndim != 2:
        raise ValueError("current features must have shape [N, D]")
    if current_anchor.shape != current_negative.shape or reference_anchor.shape != current_anchor.shape or reference_negative.shape != current_anchor.shape:
        raise ValueError("reference/current negative feature shapes differ")
    if epsilon_negative < 0:
        raise ValueError("epsilon_negative must be non-negative")
    with torch.no_grad():
        ref_s = cosine_similarity_rows(reference_anchor, reference_negative)
    cur_s = cosine_similarity_rows(current_anchor, current_negative)
    mask = torch.ones_like(ref_s, dtype=torch.bool) if reliable_mask is None else reliable_mask.bool()
    if mask.shape != ref_s.shape:
        raise ValueError("reliable_mask must have shape [N]")
    violation = F.relu(cur_s - ref_s - float(epsilon_negative))
    selected = violation[mask]
    loss = selected.mean() if selected.numel() else current_anchor.sum() * 0.0
    return loss, {
        "reference_negative_similarity": ref_s.detach(),
        "current_negative_similarity": cur_s,
        "violation": violation,
        "active_count": mask.sum().detach(),
        "violating_count": (selected > 0).sum().detach(),
    }
