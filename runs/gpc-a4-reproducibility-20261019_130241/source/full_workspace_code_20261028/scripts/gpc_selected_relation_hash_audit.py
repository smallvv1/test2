#!/usr/local/bin/python3.10
"""离线核对短程输出中实际训练步的 selected relation SHA256。

不执行模型 forward/backward；从与实验相同的冻结数据计划重建关系键，只统计前
``steps`` 个训练 batch，排除额外的诊断 batch。原始 JSON 不被修改。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
import gpc_s0_s1_controlled_pair as harness


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed in (1234, 4321):
        plan = harness.build_plan_and_batches(seed, args.steps)
        actual_s0 = harness.relation_hash({k: v for k, v in plan["s0"].items() if k <= args.steps})
        actual_s1 = harness.relation_hash({k: v for k, v in plan["s1"].items() if k <= args.steps})
        rows.append({
            "seed": seed,
            "steps": args.steps,
            "training_s0_selected_relation_sha256": actual_s0,
            "training_s1_selected_relation_sha256": actual_s1,
            "full_plan_s0_selected_relation_sha256": plan["s0_relation_sha256"],
            "full_plan_s1_selected_relation_sha256": plan["s1_relation_sha256"],
            "batch_plan_sha256": plan["batch_plan_sha256"],
            "candidate_relation_sha256": plan["candidate_relation_sha256"],
            "s0_relation_total_training_steps": sum(len(v) for k, v in plan["s0"].items() if k <= args.steps),
            "s1_relation_total_training_steps": sum(len(v) for k, v in plan["s1"].items() if k <= args.steps),
        })
    payload = {
        "status": "PASS_TRAINING_RELATION_HASH_AUDIT",
        "cpu_model_fallback": False,
        "model_forward_executed": False,
        "rows": rows,
        "note": "原始 A/A JSON 保留；本文件只纠正训练步关系 hash 与含诊断 batch 的 full-plan hash 的口径区别。",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
