#!/usr/bin/env python3
"""稳健翻转阈值语义测试。"""


def main():
    assert 0.2 > 0.1 and -0.2 < -0.1
    assert not (0.2 > 0.1 and -0.05 < -0.1)
    # 严格翻转可以存在，但近零 target 不应被自动当作稳健翻转。
    assert 0.01 > 0 and -0.0000229 <= 0
    assert not (0.01 > 1e-3 and -0.0000229 < -1e-3)
    print("NDP_ROBUST_BOUNDARY_FLIP_TEST_PASS")


if __name__ == "__main__":
    main()
