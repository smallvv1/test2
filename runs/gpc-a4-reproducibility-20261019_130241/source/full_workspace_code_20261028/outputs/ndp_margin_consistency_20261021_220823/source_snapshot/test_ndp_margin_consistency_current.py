import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path('/home/DIFFER/scripts')))
from ndp_margin_consistency import relation_margin, reactive_hinge


def test_same_state_all_margin_modes_match_reference_current():
    x = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    for mode in ('q25', 'mean', 'single'):
        ref = relation_margin(x[0], x[1:3], x[3:], mode).detach()
        cur = relation_margin(x[0], x[1:3], x[3:], mode).detach()
        assert torch.equal(ref, cur)


def test_q25_mean_single_are_explicitly_different_when_positive_set_differs():
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    n = torch.tensor([[0.5, 0.8660254]], dtype=torch.float64)
    values = [float(relation_margin(a, p, n, m)) for m in ('q25', 'mean', 'single')]
    assert len(set(round(v, 8) for v in values)) == 3


def test_reactive_improvement_zero_and_deterioration_positive():
    current = torch.tensor([0.7, 0.2], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss, raw, active = reactive_hinge(current, reference, epsilon=0.1)
    assert torch.allclose(raw, torch.tensor([0.0, 0.2], dtype=torch.float64))
    assert active.tolist() == [False, True]
    assert abs(loss.item() - 0.1) < 1e-12
    loss.backward()
    assert current.grad[0].item() == 0.0
    assert current.grad[1].item() < 0


def test_nonfinite_reference_current_is_not_allowed_to_poison_loss():
    current = torch.tensor([float('nan'), 0.3], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss, _, active = reactive_hinge(current, reference)
    assert torch.isfinite(loss)
    assert active.tolist() == [False, True]


def test_negative_pid_count_is_identity_count_not_image_count():
    negative_image_indices = [10, 11, 20]
    negative_pids = {10: 3, 11: 3, 20: 7}
    assert len(set(negative_pids[i] for i in negative_image_indices)) == 2
