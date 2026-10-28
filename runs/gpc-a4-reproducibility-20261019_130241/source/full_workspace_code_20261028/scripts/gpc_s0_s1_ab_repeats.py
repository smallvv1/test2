#!/usr/local/bin/python3.10
"""S0/S1 多次独立 20 步 A/B 配对复核。

每个重复由独立 child 进程执行一次严格 S0/S1 配对，固定相同输入、关系计划和
初始化；不因 AMP skip 或结果方向选择性重跑。该脚本只做短程诊断。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    overall_code = 0
    script = Path("/home/DIFFER/scripts/gpc_s0_s1_controlled_pair.py")
    for seed in (1234, 4321):
        for repeat in range(1, args.repeats + 1):
            child_output = args.output / f"seed{seed}_repeat{repeat}"
            child_output.mkdir(parents=True, exist_ok=True)
            log_path = child_output / "child.log"
            command = [
                sys.executable, str(script), "--mode", "pair",
                "--run-time", args.run_time, "--seed", str(seed), "--steps", str(args.steps),
                "--output", str(child_output),
            ]
            with log_path.open("w", encoding="utf-8") as stream:
                code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
            result_files = sorted(child_output.glob("s0_s1_seed*_steps*.json"))
            record = {"seed": seed, "repeat": repeat, "returncode": code, "log": str(log_path), "json": str(result_files[0]) if result_files else None}
            records.append(record)
            overall_code = overall_code or code
    summary = {
        "run_time": args.run_time,
        "steps": args.steps,
        "repeats_per_seed": args.repeats,
        "status": "COMPLETED_AB_REPEATS" if overall_code == 0 else "AB_REPEATS_INCLUDE_INVALID_PAIRS",
        "records": records,
        "cpu_model_fallback": False,
        "note": "S0/S1 每次只改变当前 batch 内合法关系；所有成功和失败重复均保留。",
    }
    write_json(args.output / "ab_repeats_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return overall_code


if __name__ == "__main__":
    raise SystemExit(main())
