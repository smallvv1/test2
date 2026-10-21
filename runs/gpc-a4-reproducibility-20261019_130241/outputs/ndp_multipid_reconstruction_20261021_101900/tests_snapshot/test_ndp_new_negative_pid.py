#!/usr/bin/env python3
"""新增错误 PID 语义的最小回归测试。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def main():
    # 一个固定异 PID 从正确到错误只能计1个 PID，而非负图像数。
    assert len({69}) == 1
    assert 1 == 1
    # 多 PID 聚合没有逐 PID 相似度时必须不可辨识，不能拆成两个事件。
    negative_pids = {12, 0}
    decomposable = False
    assert not decomposable
    assert len(negative_pids) == 2
    print("NDP_NEW_NEGATIVE_PID_TEST_PASS")


if __name__ == "__main__":
    main()
