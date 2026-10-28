import torch

from ndp_aa_operator_analysis import compare


def test_cosine_uses_float64_and_is_bounded_for_identical_float32_vectors():
    x = torch.full((10000,), 0.1, dtype=torch.float32)
    result = compare(x, x.clone())
    assert result["cosine_raw"] <= 1.0 + 1e-10
    assert result["cosine"] <= 1.0
    assert abs(result["cosine"] - 1.0) < 1e-12
