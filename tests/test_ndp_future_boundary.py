#!/usr/bin/env python3
"""NDP 同轨迹风险标签的纯离线单元测试。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ndp_future_boundary_risk import classify_transition, normalized_change


def test_transition_states_and_event_separation():
    # 正确但仅 margin 下降：不是 flip；可同时是 increment。
    r = classify_transition(0.30, 0.10, 0.01, 0.05, 0.02, 0.02)
    assert r["state"] == "correct_to_correct"
    assert not r["y_flip"]
    assert r["y_increment"]
    assert r["y_onset"]

    # 正确 -> 错误才是严格首次排序翻转的事件形式。
    r = classify_transition(0.01, -0.01, 0.00, 0.03, 0.02, 0.02)
    assert r["state"] == "correct_to_wrong"
    assert r["y_flip"] and r["y_onset"]

    # 错误 -> 正确不能被计作损害。
    r = classify_transition(-0.10, 0.10, 0.04, 0.01, 0.02, 0.02)
    assert r["state"] == "wrong_to_correct"
    assert not r["y_flip"]
    assert not r["y_increment"]


def test_persistent_is_not_onset():
    r = classify_transition(0.2, 0.1, 0.04, 0.05, 0.02, 0.02)
    assert r["y_persistent"]
    assert not r["y_onset"]


def test_interval_normalization():
    assert normalized_change(0.2, 20) == 0.01
    assert normalized_change(0.2, 0) == 0.0


if __name__ == "__main__":
    test_transition_states_and_event_separation()
    test_persistent_is_not_onset()
    test_interval_normalization()
    print("NDP_FUTURE_BOUNDARY_TEST_PASS")
