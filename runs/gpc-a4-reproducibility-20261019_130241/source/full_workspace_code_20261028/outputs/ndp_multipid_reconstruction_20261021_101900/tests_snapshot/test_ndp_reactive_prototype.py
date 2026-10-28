import torch

from ndp_reactive_prototype import directed_reactive_boundary_loss, reactive_boundary_loss


def test_improvement_is_not_punished_and_reference_is_detached():
    current = torch.tensor([0.80, 0.60], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.50, 0.50], dtype=torch.float64, requires_grad=True)
    loss = reactive_boundary_loss(current, reference, epsilon=0.01)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(current.grad == 0)
    assert reference.grad is None


def test_deterioration_has_one_sided_gradient():
    current = torch.tensor([0.20], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.50], dtype=torch.float64)
    loss = reactive_boundary_loss(current, reference, epsilon=0.10)
    assert abs(loss.item() - 0.20) < 1e-12
    loss.backward()
    assert current.grad.item() < 0
    assert torch.isfinite(current.grad).all()


def test_mask_and_empty_relation_are_safe():
    current = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.5, 0.6], dtype=torch.float64)
    loss = reactive_boundary_loss(current, reference, reliable_mask=torch.tensor([False, False]))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(current.grad == 0)


def test_weight_cap_and_weighted_reduction_are_finite():
    current = torch.tensor([0.0, 0.4], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.5, 0.5], dtype=torch.float64)
    weights = torch.tensor([1000.0, 1.0], dtype=torch.float64)
    loss = reactive_boundary_loss(current, reference, weights=weights, max_weight=2.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(current.grad).all()


def test_directed_sides_are_not_collapsed():
    # A→B 恶化，B→A 改善；无向去重会错误地抹掉第一侧。
    current = torch.tensor([0.1, 0.8], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = directed_reactive_boundary_loss(current, reference, epsilon=0.0)
    assert abs(loss.item() - 0.2) < 1e-12
    loss.backward()
    assert current.grad[0].item() < 0
    assert current.grad[1].item() == 0.0


def test_nonfinite_rows_return_finite_zero_without_nan_poisoning():
    current = torch.tensor([float("nan"), float("inf")], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.5, 0.4], dtype=torch.float64)
    loss = reactive_boundary_loss(current, reference)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(current.grad).all()
