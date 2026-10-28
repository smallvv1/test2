import numpy as np
import torch

from ndp_reliability_safe_boundary_audit import (
    PRE_REGISTERED,
    current_vulnerable,
    first_order_delta,
    historical_reliable,
    historical_safe_boundary,
    priority_indices,
)


def test_reliability_excludes_current_margin_and_allows_reliable_vulnerable():
    history = [0.20, 0.18, 0.16, 0.15]
    assert historical_reliable(history)
    assert current_vulnerable(0.01, 1, 0, None)
    assert historical_reliable(history) and current_vulnerable(0.01, 1, 0, None)


def test_safe_boundary_has_no_future_and_improvement_is_zero_penalty():
    history = [0.20, 0.18, 0.16, 0.15]
    boundary = historical_safe_boundary(history)
    assert boundary == max(PRE_REGISTERED["safe_delta_min"], np.quantile(history, 0.25) - PRE_REGISTERED["safe_epsilon_ref"])
    assert max(0.0, boundary - 0.30) == 0.0


def test_historical_wrong_relation_is_not_reliable():
    assert not historical_reliable([0.20, -0.01, 0.16, 0.15])


def test_first_order_sign_and_budget_are_explicit():
    gm = torch.tensor([1.0, 0.0])
    gl = torch.tensor([1.0, 0.0])
    assert np.isclose(float(first_order_delta(gm, gl, eta=0.1)), -0.1)
    assert np.isclose(float(first_order_delta(gm, -gl, eta=0.1)), 0.1)


def test_selection_does_not_read_future_label():
    rows = [
        {"current_margin": 0.1, "reliable_history": True, "current_vulnerability": False, "observed_degradation": False,
         "safe_boundary_active": False, "multi_pid_risk": False, "bidirectional_risk": False, "validation_flip": False},
        {"current_margin": 0.2, "reliable_history": True, "current_vulnerability": True, "observed_degradation": True,
         "safe_boundary_active": True, "multi_pid_risk": True, "bidirectional_risk": True, "validation_flip": True},
    ]
    rng = np.random.default_rng(9)
    a = priority_indices(rows, "B3_reliability_vulnerability", 1, rng)
    rows[1]["validation_flip"] = False
    b = priority_indices(rows, "B3_reliability_vulnerability", 1, rng)
    assert np.array_equal(a, b)
