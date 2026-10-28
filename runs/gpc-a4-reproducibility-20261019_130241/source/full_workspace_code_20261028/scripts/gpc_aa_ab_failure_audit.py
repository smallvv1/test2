#!/usr/local/bin/python3.10
"""离线审计既有 A/A 与 A/B 子实验。

不执行模型 forward/backward。读取已有 JSON、child.log，并按冻结关系计划计算
20 步内实际关系、共同关系和替换数量。所有原始 JSON 保留不变。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/home/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
import gpc_s0_s1_controlled_pair as harness

AA_ROOT = ROOT / "outputs/gpc_s0s0_noise_20261020_112219/aa_steps20"
AB_ROOT = ROOT / "outputs/gpc_s0s1_ab_repeats_20261020_112219/ab_steps20"

SCALAR_FIELDS = [
    "base_loss", "ce_loss", "triplet_loss", "bio_loss", "gpc_loss_raw",
    "gpc_loss_applied", "gpc_feature_grad_norm_raw", "gpc_feature_grad_norm_applied",
    "base_feature_grad_norm", "total_feature_grad_norm", "gpc_base_feature_grad_cosine",
    "tracked_parameter_update_norm",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def first_stepwise_difference(left_rows, right_rows):
    first_numeric = None
    first_state_hash = None
    max_abs: dict[str, float] = {}
    for left, right in zip(left_rows, right_rows):
        step = int(left["step"])
        if left.get("tracked_parameter_state_sha256") != right.get("tracked_parameter_state_sha256") and first_state_hash is None:
            first_state_hash = step
        for field in SCALAR_FIELDS:
            a, b = left.get(field), right.get(field)
            if not (finite_number(a) and finite_number(b)):
                continue
            delta = abs(float(a) - float(b))
            max_abs[field] = max(max_abs.get(field, 0.0), delta)
            if delta > 0.0 and first_numeric is None:
                first_numeric = step
    return {"first_numeric_divergence_step": first_numeric, "first_parameter_hash_divergence_step": first_state_hash, "max_abs_delta": max_abs}


def nonfinite_events(result: dict[str, Any], arm: str):
    events = []
    for row in result[arm]["rows"]:
        if row.get("found_inf", 0.0) or row.get("skipped"):
            events.append({
                "step": row["step"],
                "found_inf": row.get("found_inf"),
                "scale_before": row.get("amp_scale_before", row.get("scale_before")),
                "scale_after": row.get("amp_scale_after", row.get("scale_after")),
                "optimizer_step_executed": row.get("optimizer_step_executed", row.get("updated")),
                "nonfinite_gradient_names": row.get("nonfinite_gradient_names", []),
                "selected_relation_count": row.get("selected_relation_count", row.get("selected_pairs")),
            })
    return events


def log_error_context(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    traceback = "\n".join(text.splitlines()[max(0, i - 3): i + 12]) if "Traceback" in text else None
    return {"log_exists": log_path.exists(), "traceback": traceback, "has_sigabrt": "SIGABRT" in text or "SIGABRT" in text.upper()}


def load_records(root: Path, kind: str):
    rows = []
    for path in sorted(root.glob("seed*_repeat*/" + ("aa*.json" if kind == "AA" else "s0_s1_*.json"))):
        result = json.loads(path.read_text(encoding="utf-8"))
        repeat_dir = path.parent
        log_path = repeat_dir / "child.log"
        if kind == "AA":
            left, right = "a0", "a1"
            status = result["status"]
            gate = result["comparison"]
            returncode = 0 if status == "PASS_AA_PAIR" else 2
            pair_type = "A/A"
        else:
            left, right = "s0", "s1"
            status = result["status"]
            gate = result["comparison"]
            returncode = 0 if status == "PASS_CONTROLLED_PAIR" else 2
            pair_type = "A/B"
        stepwise = first_stepwise_difference(result[left]["rows"], result[right]["rows"])
        same_protocol = {
            "initial_state": gate.get("same_initial_state"),
            "input_batches": gate.get("same_input_batches"),
            "diagnostic_batch": gate.get("same_diagnostic_batch"),
            "batch_plan": result[left].get("batch_plan_sha256") == result[right].get("batch_plan_sha256"),
            "candidate_plan": result[left].get("candidate_relation_sha256") == result[right].get("candidate_relation_sha256"),
        }
        if kind == "AA":
            same_protocol["selected_relation"] = gate.get("same_selected_relation_sha256", result[left].get("selected_relation_sha256") == result[right].get("selected_relation_sha256"))
        else:
            same_protocol["selected_relation"] = False
            same_protocol["selected_relation_note"] = "A/B 中 S0/S1 关系 hash 按设计不同；不是协议失败。"
        trigger_fields = []
        if not gate.get("same_initial_state", True): trigger_fields.append("same_initial_state=false")
        if not gate.get("same_input_batches", True): trigger_fields.append("same_input_batches=false")
        if not gate.get("same_diagnostic_batch", True): trigger_fields.append("same_diagnostic_batch=false")
        if not gate.get("same_skip_sequence", gate.get("same_amp_skip_sequence", True)): trigger_fields.append("same_amp_skip_sequence=false")
        if not gate.get("same_effective_update_sequence", True): trigger_fields.append("same_effective_update_sequence=false")
        failure_stage = "none" if returncode == 0 else "pair_gate_compare_pair"
        if returncode != 0 and (trigger_fields == ["same_amp_skip_sequence=false"] or set(trigger_fields) == {"same_amp_skip_sequence=false", "same_effective_update_sequence=false"}):
            failure_stage = "AMP_update_trajectory_gate"
        repeat_match = re.search(r"repeat(\d+)", repeat_dir.name)
        repeat = int(repeat_match.group(1)) if repeat_match else result.get("repeat")
        rows.append({
            "seed": result["seed"], "repeat": repeat, "type": pair_type,
            "returncode": returncode, "status": status, "path": str(path), "child_log": str(log_path),
            "failure_stage": failure_stage, "trigger_fields": trigger_fields,
            "protocol_checks": same_protocol,
            "first_numeric_divergence_step": gate.get("first_numeric_divergence_step", stepwise["first_numeric_divergence_step"]),
            "first_parameter_hash_divergence_step": gate.get("first_tracked_parameter_hash_divergence_step", stepwise["first_parameter_hash_divergence_step"]),
            "amp_events_left": nonfinite_events(result, left), "amp_events_right": nonfinite_events(result, right),
            "skip_left": result[left]["skip_sequence"], "skip_right": result[right]["skip_sequence"],
            "updates_left": result[left]["optimizer_updates"], "updates_right": result[right]["optimizer_updates"],
            "diagnostic_base_loss_left": result[left]["diagnostic_base_loss"], "diagnostic_base_loss_right": result[right]["diagnostic_base_loss"],
            "diagnostic_consistency_left": result[left]["diagnostic_same_pid_consistency_loss"], "diagnostic_consistency_right": result[right]["diagnostic_same_pid_consistency_loss"],
            "selected_relation_total_left": result[left]["selected_relation_total"], "selected_relation_total_right": result[right]["selected_relation_total"],
            "selected_relation_sha256_left": result[left]["selected_relation_sha256"], "selected_relation_sha256_right": result[right]["selected_relation_sha256"],
            "same_relation_count": None if kind == "AA" else 0,
            "stepwise_max_abs_delta": stepwise["max_abs_delta"],
            "error_context": log_error_context(log_path),
        })
    return rows


def relation_intervention(steps: int):
    rows = []
    for seed in (1234, 4321):
        plan = harness.build_plan_and_batches(seed, steps)
        s0 = [harness.relation_key(item) for i, values in plan["s0"].items() if i <= steps for item in values]
        s1 = [harness.relation_key(item) for i, values in plan["s1"].items() if i <= steps for item in values]
        common = len(set(s0) & set(s1))
        active0 = [i for i, values in plan["s0"].items() if i <= steps and values]
        active1 = [i for i, values in plan["s1"].items() if i <= steps and values]
        rows.append({
            "seed": seed, "steps": steps,
            "active_steps_s0": active0, "active_steps_s1": active1,
            "active_step_count_s0": len(active0), "active_step_count_s1": len(active1),
            "relation_count_s0": len(s0), "relation_count_s1": len(s1),
            "identical_relation_count": common,
            "replaced_relation_count_s0": len(s0) - common,
            "replaced_relation_count_s1": len(s1) - common,
            "s0_relation_sha256": harness.relation_hash({i: v for i, v in plan["s0"].items() if i <= steps}),
            "s1_relation_sha256": harness.relation_hash({i: v for i, v in plan["s1"].items() if i <= steps}),
            "s0_profile": harness.relation_profile(plan, "S0", steps),
            "s1_profile": harness.relation_profile(plan, "S1", steps),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    aa = load_records(AA_ROOT, "AA")
    ab = load_records(AB_ROOT, "AB")
    intervention = relation_intervention(args.steps)
    summary = {
        "run_time": args.run_time,
        "source_runs": ["20261020_112219"],
        "cpu_model_fallback": False,
        "model_forward_executed": False,
        "structured_json_missing": False,
        "aa_records": aa,
        "ab_records": ab,
        "relation_intervention": intervention,
        "counts": {
            "aa_total": len(aa), "aa_returncode_0": sum(row["returncode"] == 0 for row in aa), "aa_returncode_2": sum(row["returncode"] == 2 for row in aa),
            "ab_total": len(ab), "ab_returncode_0": sum(row["returncode"] == 0 for row in ab), "ab_returncode_2": sum(row["returncode"] == 2 for row in ab),
        },
        "decision_note": "A/A 失败均有完整 JSON；returncode=2 是配对门禁返回值，不是未捕获异常。",
    }
    write_json(args.output / "failure_root_cause_audit.json", summary)
    write_json(args.output / "relation_intervention_audit.json", intervention)
    print(json.dumps({"counts": summary["counts"], "relation_intervention": intervention}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
