import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ndp_v5_prcc_runner import compose_total_loss, ndp_target_margin  # noqa: E402


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
