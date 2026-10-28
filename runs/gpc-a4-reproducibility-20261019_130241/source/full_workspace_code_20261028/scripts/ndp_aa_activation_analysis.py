#!/usr/local/bin/python3.10
"""汇总本轮 A/A、重复重编码和真实轨迹激活证据；不执行模型计算。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/home/DIFFER")
AA = ROOT / "outputs/ndp_aa_reproducibility_20261021_030947"
TRAJ = ROOT / "outputs/ndp_overnight_recovery_20261021_024248"
REPEAT = ROOT / "outputs/ndp_activation_repeat_20261021_031415"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare(left, right):
    first = None
    max_abs = {}
    rows = []
    for a, b in zip(left["rows"], right["rows"]):
        step = int(a["step"])
        categories = []
        if a["forward_output"]["sha256"] != b["forward_output"]["sha256"]:
            categories.append("forward_output")
        if a["hook_outputs"] != b["hook_outputs"]:
            categories.append("forward_hook")
        if a["gradient_layers"] != b["gradient_layers"]:
            categories.append("gradient")
        if a["found_inf"] != b["found_inf"] or a["amp_scale_after"] != b["amp_scale_after"]:
            categories.append("amp")
        if a["optimizer_state"]["sha256"] != b["optimizer_state"]["sha256"]:
            categories.append("optimizer")
        if a["parameter_layers_after"] != b["parameter_layers_after"]:
            categories.append("parameter_state")
        for field in ("base_loss", "ce_loss", "triplet_loss", "bio_loss", "gpc_loss_raw", "feature_grad_norm", "parameter_grad_norm"):
            value = abs(float(a[field]) - float(b[field]))
            max_abs[field] = max(max_abs.get(field, 0.0), value)
        if first is None and categories:
            first = {"step": step, "categories": categories}
        rows.append({"step": step, "categories": categories, "base_loss_delta": float(b["base_loss"] - a["base_loss"]), "parameter_grad_norm_delta": float(b["parameter_grad_norm"] - a["parameter_grad_norm"])})
    return {
        "same_input_tensor_hashes": all(a["input_tensor_hashes"] == b["input_tensor_hashes"] for a, b in zip(left["rows"], right["rows"])),
        "same_batch_hash": all(a["batch_hash"] == b["batch_hash"] for a, b in zip(left["rows"], right["rows"])),
        "same_relation_hash": all(a["selected_relation_sha256"] == b["selected_relation_sha256"] for a, b in zip(left["rows"], right["rows"])),
        "same_skip_sequence": left["skip_sequence"] == right["skip_sequence"],
        "same_effective_update_count": left["optimizer_updates"] == right["optimizer_updates"],
        "first_difference": first,
        "max_abs_delta": max_abs,
        "rows": rows,
        "left_updates": left["optimizer_updates"],
        "right_updates": right["optimizer_updates"],
        "left_skips": left["skip_sequence"],
        "right_skips": right["skip_sequence"],
    }


def aa_summary():
    pairs = {}
    paths = {
        1234: (AA / "seed1234_aa0/aa0.json", AA / "seed1234_aa1/aa1.json"),
        4321: (AA / "seed4321_retry2/aa0.json", AA / "seed4321_retry3/aa1.json"),
    }
    for seed, (left_path, right_path) in paths.items():
        left = json.loads(left_path.read_text())
        right = json.loads(right_path.read_text())
        pairs[str(seed)] = {
            "left": str(left_path),
            "right": str(right_path),
            "left_sha256": sha256_file(left_path),
            "right_sha256": sha256_file(right_path),
            "comparison": compare(left, right),
            "phase": left["phase"],
            "checkpoint_sha256": left["checkpoint_sha256"],
        }
    return pairs


def activation_summary():
    result = {}
    rows_out = []
    for seed in (1234, 4321):
        path = TRAJ / f"trajectory_seed{seed}/activation_audit/activation_rows.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        by_update = {}
        for update in sorted({int(row["update"]) for row in rows}):
            current = [row for row in rows if int(row["update"]) == update]
            all_active = {int(row["anchor"]) for row in current if float(row["advantage_drop"]) > 0.02}
            reliable = {int(row["anchor"]) for row in current if float(row["advantage_drop"]) > 0.02 and float(row["reference_advantage"]) > 0.05}
            by_update[str(update)] = {"all_active": sorted(all_active), "reliable_active": sorted(reliable), "all_count": len(all_active), "reliable_count": len(reliable), "group_count": len(current)}
            rows_out.append({"seed": seed, "update": update, "groups": len(current), "all_active": len(all_active), "reliable_active": len(reliable), "all_rate": len(all_active) / len(current), "reliable_rate": len(reliable) / len(current)})
        ordered = sorted(by_update, key=int)
        transitions = []
        for first, second in zip(ordered, ordered[1:]):
            a = set(by_update[first]["reliable_active"]); b = set(by_update[second]["reliable_active"])
            transitions.append({"from": int(first), "to": int(second), "intersection": len(a & b), "union": len(a | b), "jaccard": (len(a & b) / len(a | b)) if a | b else None})
        result[str(seed)] = {"rows_source": str(path), "rows_source_sha256": sha256_file(path), "by_update": by_update, "transitions": transitions}
    return result, rows_out


def main() -> None:
    activation, activation_rows = activation_summary()
    repeat = {}
    for seed in (1234, 4321):
        path = REPEAT / f"seed{seed}/repeat_encoding_summary.json"
        repeat[str(seed)] = json.loads(path.read_text())
    output = ROOT / "outputs/ndp_aa_activation_analysis_20261021_031415"
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "run_time": "20261021_031415",
        "aa": aa_summary(),
        "activation": activation,
        "repeat_encoding": repeat,
        "source_runs": ["20261021_030947", "20261021_031415", "20261021_024248"],
        "official_query_gallery_used": False,
        "cpu_model_fallback": False,
        "interpretation": {
            "aa_gate": "FAIL: both seeds show backward/gradient divergence at step1 despite identical materialized inputs and no AMP skip; seed-level pairs completed 20/20 updates.",
            "activation": "natural activation is observed on a single fixed continuation trajectory; activation set is state-dependent and not a causal optimizer-step intervention.",
            "reencode": "same update400 checkpoint repeated twice with exact feature bytes for both seeds; measurement noise was not observed in this probe.",
        },
    }
    write_json(output / "aa_activation_bundle.json", bundle)
    with (output / "activation_stability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(activation_rows[0]))
        writer.writeheader(); writer.writerows(activation_rows)
    print(json.dumps({"output": str(output), "aa_seeds": list(bundle["aa"]), "repeat": {k: (v["status"], v["max_abs_difference"]) for k, v in repeat.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
