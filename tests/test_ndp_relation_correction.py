from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, "/home/DIFFER/scripts")
from ndp_bidirectional_relation_correction import identity_stats, image_pairs


def _row(idx, anchor, anchor_pid, neg_pid, negs, delta="-0.1"):
    return {
        "prev_update": "0", "target_update": "20", "relation_index": str(idx),
        "anchor_local_index": str(anchor), "anchor_pid": str(anchor_pid),
        "negative_pid": str(neg_pid), "negative_local_indices": repr(negs),
        "positive_local_indices": repr([anchor + 10]), "delta_margin": delta,
        "strict_flip": "False", "new_negative_pid": "False",
    }


def test_identity_group_keeps_all_image_relations():
    rows = [_row(0, 0, 1, 2, [3]), _row(1, 4, 1, 2, [5])]
    out = identity_stats(rows)
    assert len(out) == 1
    assert out[0]["directed_relation_count"] == 2
    assert out[0]["unique_anchor_images"] == 2


def test_image_reciprocal_requires_exact_same_interval_and_cross_images():
    a = _row(0, 0, 1, 2, [3], delta="0.2")
    b = _row(1, 3, 2, 1, [0], delta="-0.1")
    assert len(image_pairs([a, b])) == 1
    # Missing reverse query is not silently converted into a reciprocal pair.
    assert len(image_pairs([a])) == 0
    c = dict(b); c["target_update"] = "50"
    assert len(image_pairs([a, c])) == 0


def test_fixed_upstream_probe_is_valid_and_old_claim_is_invalid():
    root = Path("/home/DIFFER/outputs/ndp_relation_correction_20261021_192813")
    for name in ("ndp_layernorm_probe.json", "ndp_layernorm_probe_seed4321.json"):
        data = json.loads((root / name).read_text())
        assert data["status"] == "PASS_FIXED_INPUT_UPSTREAM_PROBE"
        for group in data["groups"]:
            assert group["input_hash"] == group["repeat_hashes"][0]["input"]
            assert group["upstream_hash"] == group["repeat_hashes"][0]["upstream"]
            assert all(c["same_input"] and c["same_upstream"] for c in group["comparisons"])
            assert all(c["status"] == "VALID_COMPARISON" for c in group["comparisons"])


def test_reverse_knn_is_explicitly_computed():
    root = Path("/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/reference_graph")
    for seed, minimum in ((1234, 2000), (4321, 2000)):
        data = json.loads((root / f"seed{seed}/multi_pid_reference_summary.json").read_text())
        assert data["reverse_knn_topk_computed"] is True
        assert data["reverse_knn_query_count"] >= minimum


def test_triplet_audit_records_actual_semantics_not_cosine_proxy():
    source = Path("/home/DIFFER/scripts/ndp_real_triplet_miner_coverage.py").read_text()
    triplet = Path("/data/DIFFER/loss/triplet_loss.py").read_text()
    assert "euclidean_dist" in source and "hard_positive" in source and "hard_negative" in source
    assert "SoftMarginLoss" in triplet
    assert "softplus(-(dist_an-dist_ap))" in source
    assert "np.logaddexp(0.0, -difference)" in source
