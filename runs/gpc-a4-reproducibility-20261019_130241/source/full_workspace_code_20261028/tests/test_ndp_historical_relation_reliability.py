import numpy as np

from ndp_historical_relation_reliability import classify_reliability, classify_vulnerability


def test_one_node_is_not_long_term_stable():
    assert classify_reliability(np.asarray([0.4]), 0.05, 0.05, 3) == "INSUFFICIENT_HISTORY"


def test_reliability_and_vulnerability_are_separate_dimensions():
    h = np.asarray([0.20, 0.18, 0.16])
    assert classify_reliability(h, 0.05, 0.05, 3) == "A_STABLE_CORRECT"
    assert classify_vulnerability(h, 0.05, 0.05) == "LOW_VULNERABILITY"
    h2 = np.asarray([0.20, 0.03, 0.02])
    assert classify_reliability(h2, 0.05, 0.05, 3) == "B_CORRECT_NEAR_OR_VARIABLE"
    assert classify_vulnerability(h2, 0.05, 0.05) == "HIGH_VULNERABILITY"
