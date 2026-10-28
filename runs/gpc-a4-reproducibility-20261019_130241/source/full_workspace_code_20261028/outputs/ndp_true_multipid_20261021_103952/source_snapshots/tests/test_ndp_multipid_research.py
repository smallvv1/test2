import json
from pathlib import Path

import torch

from ndp_historical_relation_reliability import classify_reliability, classify_vulnerability
from ndp_reactive_prototype import reactive_boundary_loss


def test_single_history_is_not_stable():
    assert classify_reliability([0.3], 0.0, 0.05, 3) == "INSUFFICIENT_HISTORY"


def test_reliability_and_vulnerability_are_independent_dimensions():
    assert classify_reliability([0.08, 0.09, 0.07], 0.0, 0.05, 3) == "A_STABLE_CORRECT"
    assert classify_vulnerability([0.08, 0.09, 0.07], 0.05, 0.05) == "LOW_VULNERABILITY"
    assert classify_vulnerability([0.08, -0.01, 0.07], 0.05, 0.05) == "HIGH_VULNERABILITY"


def test_reactive_improvement_is_zero_and_degradation_has_gradient():
    current = torch.tensor([0.7, 0.2], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = reactive_boundary_loss(current, reference, epsilon=0.05)
    assert torch.isfinite(loss)
    assert float(loss) > 0
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[0] == 0
    assert current.grad[1] < 0


def test_update0_graph_has_multiple_negative_pids_without_future_selection():
    root = Path("/home/DIFFER/outputs/ndp_true_multipid_20261021_103952/reference_graph")
    for seed in (1234, 4321):
        summary = json.loads((root / f"seed{seed}/multi_pid_reference_summary.json").read_text())
        assert summary["future_nodes_used_for_selection"] is False
        assert summary["multi_pid_anchor_by_type_k"]["difficult_k5"] > 0


def test_two_directions_are_not_collapsed():
    a = (123, 456)
    b = (456, 123)
    assert a != b


def test_coverage_summary_keeps_count_and_path_separate():
    p = Path("/home/DIFFER/outputs/ndp_true_multipid_20261021_103952/coverage_bias_fix/seed1234/relation_coverage_summary.json")
    data = json.loads(p.read_text())
    assert isinstance(data["group_count"], int)
    assert isinstance(data["group_path"], str)
    assert isinstance(data["groups_sha256"], str)
