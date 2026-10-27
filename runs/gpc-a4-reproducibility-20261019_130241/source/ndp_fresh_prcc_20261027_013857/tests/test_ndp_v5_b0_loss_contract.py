import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ndp_v5_prcc_runner import compose_total_loss, gradient_gate_from_grads, ndp_target_margin, ndyn_should_select, ndyn_top5_target_margin, nrg_should_select  # noqa: E402


def test_b0_total_is_main_total_and_has_no_ndp_gradient():
    main = torch.tensor(3.0, requires_grad=True)
    ndp = torch.tensor(2.0, requires_grad=True)
    total = compose_total_loss(main, ndp, "B0")

    assert total is main
    assert torch.equal(total, main)
    total.backward()
    assert main.grad.item() == 1.0
    assert ndp.grad is None


def test_n5_total_contains_only_registered_ndp_weight():
    main = torch.tensor(3.0, requires_grad=True)
    ndp = torch.tensor(2.0, requires_grad=True)
    total = compose_total_loss(main, ndp, "N5")

    assert torch.isclose(total, torch.tensor(3.1))
    total.backward()
    assert torch.isclose(main.grad, torch.tensor(1.0))
    assert torch.isclose(ndp.grad, torch.tensor(0.05))


def test_nf0_uses_zero_floor_for_small_positive_reference_margin():
    assert ndp_target_margin(0.01, 0.02, "N5") == -0.01
    assert ndp_target_margin(0.01, 0.02, "NF0") == 0.0
    assert ndp_target_margin(0.08, 0.02, "NF0") == 0.06


def test_nf0_total_contains_only_registered_ndp_weight():
    main = torch.tensor(3.0, requires_grad=True)
    ndp = torch.tensor(2.0, requires_grad=True)
    total = compose_total_loss(main, ndp, "NF0")

    assert torch.isclose(total, torch.tensor(3.1))
    total.backward()
    assert torch.isclose(ndp.grad, torch.tensor(0.05))


def test_nrg_selects_only_historically_safe_current_near_boundary_relations():
    assert nrg_should_select(0.02, 0.01)
    assert nrg_should_select(0.08, 0.019999)
    assert not nrg_should_select(0.019999, 0.01)
    assert not nrg_should_select(0.08, 0.0)
    assert not nrg_should_select(0.08, 0.02)


def test_nrg_has_positive_floor_and_fixed_loss_weight():
    assert ndp_target_margin(0.01, 0.02, "NRG") == 0.01
    assert ndp_target_margin(0.08, 0.02, "NRG") == 0.06
    main = torch.tensor(3.0, requires_grad=True)
    ndp = torch.tensor(2.0, requires_grad=True)
    total = compose_total_loss(main, ndp, "NRG")
    total.backward()
    assert torch.isclose(total, torch.tensor(3.1))
    assert torch.isclose(ndp.grad, torch.tensor(0.05))


def test_nrg_hinge_has_current_margin_gradient_when_target_is_violated():
    current_margin = torch.tensor(0.005, requires_grad=True)
    target = torch.tensor(ndp_target_margin(0.08, 0.02, "NRG"))
    loss = torch.relu(target - current_margin)
    loss.backward()
    assert torch.isclose(loss, torch.tensor(0.055))
    assert torch.isclose(current_margin.grad, torch.tensor(-1.0))


def test_ncap_caps_reference_target_without_current_risk_gate():
    assert ndp_target_margin(0.01, 0.02, "NCAP") == 0.0
    assert ndp_target_margin(0.08, 0.02, "NCAP") == 0.02
    assert ndp_target_margin(0.04, 0.02, "NCAP") == 0.02
    assert torch.isclose(torch.tensor(ndp_target_margin(0.021, 0.02, "NCAP")), torch.tensor(0.001))


def test_ncap_total_contains_only_registered_ndp_weight():
    main = torch.tensor(3.0, requires_grad=True)
    ndp = torch.tensor(2.0, requires_grad=True)
    total = compose_total_loss(main, ndp, "NCAP")
    total.backward()
    assert torch.isclose(total, torch.tensor(3.1))
    assert torch.isclose(ndp.grad, torch.tensor(0.05))


def test_ndyn_selects_only_positive_recently_degraded_relations():
    assert ndyn_should_select(0.08, 0.04)
    assert not ndyn_should_select(0.08, 0.075)
    assert not ndyn_should_select(0.08, 0.0)
    assert not ndyn_should_select(-0.01, 0.01)
    assert not ndyn_should_select(0.08, 0.06)


def test_ndyn_uses_t1_minus_fixed_delta_and_keeps_ndp_weight():
    assert torch.isclose(torch.tensor(ndp_target_margin(0.08, 0.02, "NDYN")), torch.tensor(0.075))
    assert ndp_target_margin(0.003, 0.02, "NDYN") == 0.0
    main = torch.tensor(3.0, requires_grad=True)
    ndp = torch.tensor(2.0, requires_grad=True)
    total = compose_total_loss(main, ndp, "NDYN")
    total.backward()
    assert torch.isclose(total, torch.tensor(3.1))
    assert torch.isclose(ndp.grad, torch.tensor(0.05))


def test_ndyn_top5_target_recovers_only_half_of_observed_drop():
    assert torch.isclose(torch.tensor(ndyn_top5_target_margin(0.04, 0.02)), torch.tensor(0.045))
    assert torch.isclose(torch.tensor(ndyn_top5_target_margin(0.04, 0.20)), torch.tensor(0.045))
    assert torch.isclose(torch.tensor(ndyn_top5_target_margin(0.04, -0.01)), torch.tensor(0.04))


def test_ndyn_top5_loss_has_registered_weight_and_gradient():
    main = torch.tensor(3.0, requires_grad=True)
    ndp = torch.tensor(2.0, requires_grad=True)
    total = compose_total_loss(main, ndp, "NDYN_TOP5")
    total.backward()
    assert torch.isclose(total, torch.tensor(3.1))
    assert torch.isclose(ndp.grad, torch.tensor(0.05))


def test_gradient_gate_keeps_nonnegative_cosine_and_rejects_conflict():
    keep, cosine, status = gradient_gate_from_grads(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))
    assert keep and cosine > 0.99 and status == "kept"
    keep, cosine, status = gradient_gate_from_grads(torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.0]))
    assert not keep and cosine < -0.99 and status == "conflict"


def test_gradient_gate_rejects_missing_or_zero_norm_gradient():
    assert gradient_gate_from_grads(None, torch.tensor([1.0]))[2] == "missing_gradient"
    assert gradient_gate_from_grads(torch.tensor([0.0]), torch.tensor([1.0]))[2] == "zero_norm"


def test_ngate_uses_same_ndp_weight_without_changing_main_loss():
    main = torch.tensor(3.0, requires_grad=True)
    ndp = torch.tensor(2.0, requires_grad=True)
    total = compose_total_loss(main, ndp, "NGATE")
    assert torch.isclose(total, torch.tensor(3.1))
    total.backward()
    assert torch.isclose(main.grad, torch.tensor(1.0))
    assert torch.isclose(ndp.grad, torch.tensor(0.05))
