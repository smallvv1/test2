#!/usr/bin/env python3
"""双侧边界同一时间区间配对回归测试。"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rows = td / "rows.csv"
        rows2 = td / "rows2.csv"
        groups = td / "groups.json"
        out = td / "out"
        intervals = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 400)]
        fields = ["seed", "anchor", "pid", "feature_update", "target_update", "prev_margin", "target_margin", "delta_drop", "y_flip"]
        with rows.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
            for anchor, pid, base in ((1, 10, .1), (2, 20, .2)):
                for i, (u0, u1) in enumerate(intervals):
                    prev = base - .01 * i
                    target = prev - .05
                    w.writerow({"seed": 1234, "anchor": anchor, "pid": pid,
                                "feature_update": u0, "target_update": u1,
                                "prev_margin": prev, "target_margin": target,
                                "delta_drop": .0, "y_flip": str(target <= 0)})
        rows2.write_text(rows.read_text(encoding="utf-8").replace("1234", "4321"), encoding="utf-8")
        groups.write_text(json.dumps({"groups": [
            {"anchor": 1, "pid": 10, "positives": [11], "negatives": [2]},
            {"anchor": 2, "pid": 20, "positives": [22], "negatives": [1]},
        ]}), encoding="utf-8")
        # 脚本要求两个 seed 输入；第二份仅将 seed 标识改为4321。
        subprocess.run([
            sys.executable, str(root / "scripts/ndp_bidirectional_boundary_audit.py"),
            "--rows", str(rows), str(rows2), "--groups", str(groups), str(groups),
            "--output-dir", str(out),
        ], check=True, capture_output=True, text=True)
        with (out / "bidirectional_boundary_seed1234.csv").open(encoding="utf-8") as fh:
            got = list(csv.DictReader(fh))
        assert len(got) == 5, got
        first = got[0]
        assert (first["feature_update"], first["target_update"]) == ("0", "20")
        assert abs(float(first["m_a_prev"]) - .1) < 1e-12
        assert abs(float(first["m_a_target"]) - .05) < 1e-12
        assert abs(float(first["delta_a_to_b"]) + .05) < 1e-12
        assert (got[-1]["feature_update"], got[-1]["target_update"]) == ("200", "400")
    print("NDP_BIDIRECTIONAL_ALIGNMENT_TEST_PASS")


if __name__ == "__main__":
    main()
