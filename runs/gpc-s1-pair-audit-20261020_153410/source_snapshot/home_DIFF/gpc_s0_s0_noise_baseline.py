#!/usr/local/bin/python3.10
"""S0/S0 数值噪声基线。

每个 child 进程构建一次冻结 train-only batch/关系计划，然后从相同 checkpoint、
optimizer、GradScaler 和 seed 连续执行两个独立 S0 分支。parent 进程按 seed 和
repeat 顺序启动 child，避免 PPU 争抢。该脚本不保存模型 checkpoint，不读取官方
query/gallery，也不修改正式训练处理器。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path("/home/DIFFER")
HARNESS = ROOT / "scripts/gpc_s0_s1_controlled_pair.py"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finite_number(value):
    return isinstance(value, (int, float)) and value == value and abs(value) != float("inf")


def compare_rows(first, second):
    scalar_fields = [
        "base_loss", "ce_loss", "triplet_loss", "bio_loss", "gpc_loss_raw",
        "gpc_loss_applied", "gpc_feature_grad_norm_raw", "gpc_feature_grad_norm_applied",
        "base_feature_grad_norm", "total_feature_grad_norm",
        "gpc_base_feature_grad_cosine", "tracked_parameter_update_norm",
    ]
    differences = {}
    first_numeric_divergence = None
    first_parameter_hash_divergence = None
    max_abs = {}
    for left, right in zip(first["rows"], second["rows"]):
        step = int(left["step"])
        if left.get("tracked_parameter_state_sha256") != right.get("tracked_parameter_state_sha256") and first_parameter_hash_divergence is None:
            first_parameter_hash_divergence = step
        for field in scalar_fields:
            a, b = left.get(field), right.get(field)
            if not (finite_number(a) and finite_number(b)):
                continue
            delta = abs(float(a) - float(b))
            max_abs[field] = max(max_abs.get(field, 0.0), delta)
            if delta > 0.0 and first_numeric_divergence is None:
                first_numeric_divergence = step
            differences.setdefault(field, []).append({"step": step, "delta_second_minus_first": float(b) - float(a)})
    return {
        "first_numeric_divergence_step": first_numeric_divergence,
        "first_tracked_parameter_hash_divergence_step": first_parameter_hash_divergence,
        "max_absolute_delta": max_abs,
        "stepwise_deltas": differences,
        "same_input_batches": first["input_batch_hashes"] == second["input_batch_hashes"],
        "same_diagnostic_batch": first["diagnostic_batch_hash"] == second["diagnostic_batch_hash"],
        "same_selected_relation_sha256": first["selected_relation_sha256"] == second["selected_relation_sha256"],
        "same_skip_sequence": first["skip_sequence"] == second["skip_sequence"],
        "same_effective_update_sequence": first["effective_update_sequence"] == second["effective_update_sequence"],
        "same_initial_state": first["initial_state"] == second["initial_state"],
        "diagnostic_delta_base_loss_second_minus_first": second["diagnostic_base_loss"] - first["diagnostic_base_loss"],
        "diagnostic_delta_consistency_second_minus_first": (
            None if first["diagnostic_same_pid_consistency_loss"] is None or second["diagnostic_same_pid_consistency_loss"] is None
            else second["diagnostic_same_pid_consistency_loss"] - first["diagnostic_same_pid_consistency_loss"]
        ),
    }


def child(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    import gpc_s0_s1_controlled_pair as harness

    if not torch.cuda.is_available():
        raise RuntimeError("PPU 不可用，拒绝 CPU model fallback")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 2 or any(name != "PPU-ZW810E" for name in devices):
        raise RuntimeError(f"PPU 设备不符合要求: {devices}")

    plan = harness.build_plan_and_batches(args.seed, args.steps)
    device = torch.device("cuda:0")
    print(f"中文进度：seed={args.seed} repeat={args.repeat} 开始 A0", flush=True)
    a0 = harness.run_arm(plan, args.seed, "S0", args.steps, device)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"中文进度：seed={args.seed} repeat={args.repeat} 开始 A1", flush=True)
    a1 = harness.run_arm(plan, args.seed, "S0", args.steps, device)
    comparison = compare_rows(a0, a1)
    pair_valid = all(comparison[key] for key in (
        "same_input_batches", "same_diagnostic_batch", "same_selected_relation_sha256",
        "same_skip_sequence", "same_effective_update_sequence", "same_initial_state",
    ))
    payload = {
        "status": "PASS_AA_PAIR" if pair_valid else "INVALID_AA_PAIR",
        "run_time": args.run_time,
        "seed": args.seed,
        "repeat": args.repeat,
        "steps": args.steps,
        "epoch": harness.EPOCH,
        "device_names": devices,
        "cpu_model_fallback": False,
        "lambda": harness.LAMBDA,
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "config_sha256": plan["config_sha256"],
        "split_sha256": plan["split_sha256"],
        "batch_plan_sha256": plan["batch_plan_sha256"],
        "candidate_relation_sha256": plan["candidate_relation_sha256"],
        "selected_relation_sha256": plan["s0_relation_sha256"],
        "batch_fingerprints": plan["cached_hashes"],
        "relation_profile": harness.relation_profile(plan, "S0", args.steps),
        "a0": a0,
        "a1": a1,
        "comparison": comparison,
        "protocol_note": "A0/A1 都是原始 S0；仅用于测量同方案数值噪声，不是 S1 方法比较。",
    }
    output = args.output / f"aa_seed{args.seed}_repeat{args.repeat}_steps{args.steps}.json"
    write_json(output, payload)
    print(json.dumps({"status": payload["status"], "seed": args.seed, "repeat": args.repeat, "comparison": comparison}, ensure_ascii=False, indent=2), flush=True)
    return 0 if pair_valid else 2


def parent(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    overall_code = 0
    for seed in (1234, 4321):
        for repeat in (1, 2, 3):
            child_output = args.output / f"seed{seed}_repeat{repeat}"
            child_output.mkdir(parents=True, exist_ok=True)
            log_path = child_output / "child.log"
            command = [
                sys.executable, str(Path(__file__).resolve()), "--mode", "child",
                "--run-time", args.run_time, "--seed", str(seed), "--repeat", str(repeat),
                "--steps", str(args.steps), "--output", str(child_output),
            ]
            with log_path.open("w", encoding="utf-8") as stream:
                code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
            result_files = sorted(child_output.glob("aa_seed*_repeat*_steps*.json"))
            record = {"seed": seed, "repeat": repeat, "returncode": code, "log": str(log_path), "json": str(result_files[0]) if result_files else None}
            records.append(record)
            overall_code = overall_code or code
    summary = {
        "run_time": args.run_time,
        "steps": args.steps,
        "repeats_per_seed": 3,
        "status": "COMPLETED_AA_BASELINE" if overall_code == 0 else "AA_BASELINE_HAS_INVALID_REPEATS",
        "records": records,
        "cpu_model_fallback": False,
        "note": "所有成功和失败重复均保留；不因结果方向选择性重跑。",
    }
    write_json(args.output / "aa_baseline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return overall_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parent", "child"), default="parent")
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(parent(args) if args.mode == "parent" else child(args))


if __name__ == "__main__":
    main()
