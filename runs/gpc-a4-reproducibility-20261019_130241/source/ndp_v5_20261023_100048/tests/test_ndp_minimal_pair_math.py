import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ndp_margin_consistency import reactive_hinge, relation_margin


def test_improvement_is_zero_and_solely_current_branch_has_gradient():
    reference = torch.tensor([0.40, 0.30], dtype=torch.float32)
    current = torch.tensor([0.50, 0.10], dtype=torch.float32, requires_grad=True)
    loss, raw, active = reactive_hinge(current, reference, epsilon=0.02)
    loss.backward()
    assert float(raw[0]) == 0.0
    assert bool(active[0]) is False
    assert bool(active[1]) is True
    assert float(current.grad[0]) == 0.0
    assert float(current.grad[1]) < 0.0


def test_reference_and_current_share_one_margin_definition():
    anchor = torch.tensor([1.0, 0.0])
    positives = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
    negatives = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
    expected = relation_margin(anchor, positives, negatives, "mean")
    live_anchor = anchor.clone().requires_grad_(True)
    current = relation_margin(live_anchor, positives, negatives, "mean")
    assert torch.allclose(expected, current.detach())
    current.backward()
    assert live_anchor.grad is not None
    assert torch.isfinite(live_anchor.grad).all()
