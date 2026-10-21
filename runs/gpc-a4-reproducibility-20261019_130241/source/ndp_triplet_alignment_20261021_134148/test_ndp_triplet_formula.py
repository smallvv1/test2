import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/DIFFER")
from loss.triplet_loss import TripletLoss, euclidean_dist, hard_example_mining


def test_softmargin_formula_matches_pytorch_and_correct_sign():
    difference = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    expected = torch.logaddexp(torch.zeros_like(difference), -difference)
    actual = torch.nn.SoftMarginLoss(reduction="none")(difference, torch.ones_like(difference))
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    grad = torch.autograd.grad(actual.sum(), difference)[0]
    expected_grad = -torch.sigmoid(-difference.detach())
    assert torch.allclose(grad, expected_grad, atol=1e-12, rtol=1e-12)
    assert float(actual[2]) < float(actual[0])


def test_original_hard_mining_is_euclidean_and_indices_are_explicit():
    features = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    dist = euclidean_dist(features, features)
    dap, dan, pidx, nidx = hard_example_mining(dist, labels, return_inds=True)
    assert pidx.tolist() == [1, 0, 3, 2]
    assert nidx.tolist() == [3, 2, 1, 0]
    loss, dap2, dan2 = TripletLoss()(features, labels, normalize_feature=False)
    assert torch.allclose(dap, dap2)
    assert torch.allclose(dan, dan2)
    assert torch.isfinite(loss)


def test_normalization_is_a_distinct_contract():
    source = Path("/home/DIFFER/scripts/ndp_real_triplet_miner_coverage.py").read_text()
    assert "feature_normalization_in_loss\": False" in source
    assert "SoftMarginLoss(dist_an - dist_ap)=softplus(-(dist_an-dist_ap))" in source
    assert "soft_margin_gradient_d_difference" in source


def test_baseline_hashes_unchanged():
    expected = {
        "/data/DIFFER/scripts/gpc_v2_train.py": "d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5",
        "/data/DIFFER/loss/make_loss.py": "9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043",
        "/data/DIFFER/configs/ltcc/eva02_l_bio.yml": "c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97",
    }
    for path, want in expected.items():
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert h == want
