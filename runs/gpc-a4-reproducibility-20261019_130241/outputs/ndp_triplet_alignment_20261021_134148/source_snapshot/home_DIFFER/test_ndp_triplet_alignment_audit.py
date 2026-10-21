import json
from pathlib import Path

import torch


ROOT = Path("/home/DIFFER")
RUN = ROOT / "outputs/ndp_triplet_alignment_20261021_134148"


def test_softmargin_formula_and_gradient_direction():
    d = torch.tensor([-20.0, -2.0, 0.0, 2.0, 20.0], dtype=torch.float64, requires_grad=True)
    loss = torch.nn.SoftMarginLoss(reduction="none")(d, torch.ones_like(d))
    loss.sum().backward()
    expected = torch.logaddexp(torch.zeros_like(d.detach()), -d.detach())
    expected_grad = -torch.sigmoid(-d.detach())
    assert torch.allclose(loss.detach(), expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(d.grad, expected_grad, atol=1e-12, rtol=1e-12)
    assert loss[0] > loss[-1]


def test_raw_feature_alignment_keeps_training_contract_and_detects_normalized_change():
    raw = json.loads((RUN / "seed4321/triplet_raw_feature_alignment_seed4321.json").read_text())
    assert "normalize_feature=False" in raw["original_training_input_contract"]
    assert raw["raw_feature"]["shape"] == [8, 1024]
    assert raw["triplet_raw"]["gradient_finite"]
    assert raw["triplet_normalized"]["gradient_finite"]
    assert raw["raw_vs_norm_hard_positive_same"] is True
    # The normalized retrieval bank is not the raw Triplet input contract.
    assert raw["raw_vs_norm_hard_negative_same"] is False


def test_multilevel_coverage_separates_pid_and_image_levels():
    for seed in (1234, 4321):
        p = RUN / f"miner_coverage/seed{seed}/triplet_miner_coverage_summary.json"
        data = json.loads(p.read_text())
        assert data["reconstructed_sampler"] is True
        assert data["historical_batch_log_available"] is False
        assert data["ndp_relations_with_hard_negative_pid"] > 0
        assert data["ndp_relations_with_full_miner_selection"] == 0
        assert "random_pid_matched_control" in data


def test_first_flip_denominator_excludes_already_wrong_relations():
    for seed in (1234, 4321):
        p = RUN / f"reliability_incremental/seed{seed}/reliability_incremental_rows.csv"
        import csv
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows
        assert all(float(row["prev_margin"]) > 0 for row in rows)


def test_reactive_active_count_is_selected_mask_count():
    for seed in (1234, 4321):
        data = json.loads((RUN / f"reactive_feature_gradient/seed{seed}/reactive_feature_gradient_summary.json").read_text())
        for name, row in data["masks"].items():
            assert row["active_relations"] <= row["selected_relations"]
            assert row["all_candidate_active_relations"] >= row["active_relations"]
            assert row["gradient_finite"] is True


def test_fixed_gpc_fingerprints_unchanged():
    expected = {
        "/data/DIFFER/scripts/gpc_v2_train.py": "d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5",
        "/data/DIFFER/loss/make_loss.py": "9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043",
        "/data/DIFFER/configs/ltcc/eva02_l_bio.yml": "c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97",
    }
    import hashlib
    for name, digest in expected.items():
        h = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        assert h == digest
