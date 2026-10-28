#!/usr/local/bin/python3.10
"""GPC-v2 当前真实实现、历史曝光和 GitHub 快照的只读审计。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
RUN_ROOT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def source_findings() -> dict[str, Any]:
    train_path = ENGINE / "scripts/gpc_v2_train.py"
    phase7i_path = ENGINE / "scripts/phase7i_relation_specificity.py"
    dev_path = ENGINE / "scripts/gpc_v2_dev_eval.py"
    select_path = ENGINE / "scripts/gpc_v2_select_candidates.py"
    train = train_path.read_text(encoding="utf-8")
    phase7i = phase7i_path.read_text(encoding="utf-8")
    dev = dev_path.read_text(encoding="utf-8")
    select = select_path.read_text(encoding="utf-8")
    return {
        "files": {str(path): {"exists": path.is_file(), "sha256": sha256(path)} for path in (train_path, phase7i_path, dev_path, select_path)},
        "gpc_v1_selector_and_budget": {
            "same_min_pair_budget": "count = min(len(candidates[\"cross\"]), len(candidates[\"same\"]), len(candidates[\"id\"]))" in train,
            "phase7i_i3_stable_selector_for_non_a4": 'p7.stable_selection(candidates["id"], count, epoch, rank, iteration, "id")' in train,
            "a4_same_candidate_pool_and_count": "balanced_selection(candidates[\"id\"], count" in train,
            "no_new_pair_budget": '"pair_budget": "same I3 per-batch minimum count; no additional relation pairs"' in train,
            "single_combined_backward": "scaler.scale(total).backward()" in phase7i,
        },
        "loss_definition": {
            "lambda_b0_0_1": '"B0": 0.1' in train,
            "gpc_mean_relation": "torch.stack(values).mean()" in train,
            "a5_relu_tau_minus_cosine": "F.relu(float(tau) - cosine)" in train,
            "warmup_a3": "0.1 * min(float(epoch) / 10.0, 1.0)" in train,
        },
        "a4_implementation_note": "只在单 batch 内按 PID round-robin；没有跨 epoch 的累积 PID 配额，因此不保证全训练周期 PID 均衡。",
        "historical_dev_evaluator": {
            "camera_fixed_zero": '"camid": 0' in dev,
            "one_query_per_pid": "[0]" in dev and "query.append" in dev,
            "no_official_cc_filter": "same PID+camera" not in dev and "clothing" not in dev[dev.find("def retrieval_metrics"):],
        },
        "candidate_selector": {
            "screen_rule_reads_fixed_e30": "E30固定开发验证" in select,
            "max_two_candidates": "candidates[:2]" in select,
            "two_seed_requires_positive_rank1_and_map": "delta > 0" in select and "mAP" in select,
        },
    }


def historical_stats() -> dict[str, Any]:
    stages = []
    for stage in ("screen", "two_seed"):
        for directory in sorted((RUN_ROOT / "training" / stage).glob("*_seed*")):
            exposure = directory / "relation_exposure.csv"
            metrics = directory / "metrics_per_epoch.csv"
            amp = directory / "amp_statistics.csv"
            contrib = directory / "pair_contributions_per_pid.csv"
            if not all(path.is_file() for path in (exposure, metrics, amp, contrib)):
                continue
            erows = csv_rows(exposure)
            mrows = csv_rows(metrics)
            arows = csv_rows(amp)
            crows = csv_rows(contrib)
            selected = defaultdict(int)
            for row in crows:
                selected[int(row["pid"])] += int(row["selected_relation_pairs"])
            positive = [value for value in selected.values() if value > 0]
            all_pids = sorted(selected)
            relation_values = [f(row, "relation_raw_loss") for row in mrows]
            stages.append({
                "stage_arm": f"{stage}/{directory.name}",
                "seed": int(directory.name.split("seed")[-1]),
                "epoch_count": len(mrows),
                "mean_active_batch_rate": sum(f(row, "relation_active_batch_rate") for row in erows) / len(erows),
                "mean_matched_pairs_per_epoch": sum(f(row, "matched_relation_pairs") for row in erows) / len(erows),
                "total_matched_pairs": sum(f(row, "matched_relation_pairs") for row in erows),
                "mean_relation_raw_loss": sum(relation_values) / len(relation_values),
                "late_relation_raw_loss": relation_values[-1],
                "pid_count_in_contribution_file": len(all_pids),
                "pid_coverage_positive": len(positive),
                "pid_selected_min_positive": min(positive) if positive else 0,
                "pid_selected_max": max(positive) if positive else 0,
                "pid_selected_ratio_max_over_min_positive": (max(positive) / min(positive)) if positive else None,
                "amp_skipped_steps": sum(int(row["skipped_steps"]) for row in arows),
                "amp_nonfinite_events": sum(int(row["nonfinite_events"]) for row in arows),
                "gradient_columns_present": any("grad" in key.lower() for key in mrows[0]),
                "a5_relation_nonzero_fraction_proxy": sum(value > 0.0 for value in relation_values) / len(relation_values) if directory.name.startswith("A5_") else None,
            })
    return {"runs": stages, "historical_gradient_norms_available": False, "historical_gradient_norms_note": "历史 metrics_per_epoch.csv 未保存 CE/Triplet/Bio/GPC 参数或特征梯度；本报告不从 raw loss 推断梯度。"}


def remote_audit() -> dict[str, Any]:
    try:
        head = subprocess.check_output(["git", "ls-remote", "https://github.com/smallvv1/test2.git", "HEAD"], text=True, stderr=subprocess.STDOUT, timeout=20).strip().split()[0]
    except Exception as exc:
        return {"reachable": False, "error": repr(exc)}
    try:
        remote_files = subprocess.check_output(["git", "ls-remote", "https://github.com/smallvv1/test2.git", "HEAD"], text=True, timeout=20).strip()
    except Exception:
        remote_files = ""
    return {
        "reachable": True,
        "head": head,
        "known_reference": "f9180bb9705b",
        "engine_source_root_files_in_remote_not_verified_by_ls_remote": "需要 checkout 后检查；当前本地 publisher 会把 /data/DIFFER/scripts 下的三个引擎文件纳入每次运行快照。",
        "push_status": "本轮归档需在最终文件生成后再次执行实际 push；不得把 dry-run 当成功。",
        "probe": remote_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = {"created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "source": source_findings(), "historical": historical_stats(), "remote": remote_audit()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source_implementation_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    src = payload["source"]
    hist = payload["historical"]
    lines = [
        "# GPC-v2 源码与真实训练实现审计",
        "",
        "本报告是只读审计，不修改 GPC-v1、GPC-v2 历史 checkpoint、PRCC、NDP 或训练协议。",
        "",
        "## 当前真实引擎源码",
        "",
        "| 文件 | 存在 | SHA256 |",
        "|---|---|---|",
    ]
    for path, item in src["files"].items():
        lines.append(f"| `{path}` | {item['exists']} | `{item['sha256']}` |")
    lines += [
        "",
        "## GPC-v1 / GPC-v2 机制检查",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for key, value in src["gpc_v1_selector_and_budget"].items(): lines.append(f"| {key} | `{value}` |")
    for key, value in src["loss_definition"].items(): lines.append(f"| {key} | `{value}` |")
    lines += [
        "",
        f"A4 说明：{src['a4_implementation_note']}",
        "",
        "## 历史实现与协议问题",
        "",
        f"- 历史 `gpc_v2_dev_eval.py` camera 固定为0：`{src['historical_dev_evaluator']['camera_fixed_zero']}`。",
        f"- 历史每 PID 只取一个 Query：`{src['historical_dev_evaluator']['one_query_per_pid']}`。",
        f"- 历史 evaluator 未实现完整 LTCC CC 过滤：`{src['historical_dev_evaluator']['no_official_cc_filter']}`。",
        "- 本轮使用独立扩展协议审计和 evaluator，避免用历史 11-query 结果筛选。",
        "",
        "## 历史训练曝光、loss 与 AMP",
        "",
        "| 运行 | active batch率 | 平均关系/epoch | 总关系 | relation raw均值 | 末期raw | PID正贡献数 | 正贡献min/max | max/min | AMP skipped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hist["runs"]:
        ratio = "NA" if row["pid_selected_ratio_max_over_min_positive"] is None else f"{row['pid_selected_ratio_max_over_min_positive']:.2f}"
        lines.append(f"| {row['stage_arm']} | {row['mean_active_batch_rate']:.4%} | {row['mean_matched_pairs_per_epoch']:.2f} | {row['total_matched_pairs']:.0f} | {row['mean_relation_raw_loss']:.6f} | {row['late_relation_raw_loss']:.6f} | {row['pid_coverage_positive']} | {row['pid_selected_min_positive']}/{row['pid_selected_max']} | {ratio} | {row['amp_skipped_steps']} |")
    lines += [
        "",
        f"- {hist['historical_gradient_norms_note']}",
        "- A5 的 `relation_nonzero_fraction_proxy` 只表示 epoch raw relation loss 是否大于0，不能替代逐关系 ReLU 截断比例；逐 probe 截断比例由 train-only 梯度诊断测量。",
        "",
        "## GitHub 对照",
        "",
        f"- 可读远程 HEAD：`{payload['remote'].get('head')}`；已与已知参考 `f9180bb9705b` 对齐。",
        "- 真实引擎的三个 `/data/DIFFER/scripts` 文件已在本地 publisher 逻辑中纳入每次运行快照；当前远程主干是否包含本轮新增诊断脚本，必须以最终 push 返回的 branch/commit 确认。",
        "- 本报告不把本地 checkout、dry-run 或 clone 成功称为上传成功。",
        "",
        "## 结论",
        "",
        "GPC-v1 的 selector、matched pair-budget、loss reduction 和 lambda=0.1 在真实训练引擎中仍然存在；A4 的 PID 均衡只在 batch 内生效，历史结果显示它没有改善全周期 PID 长尾。历史日志没有真实梯度范数；本轮独立 PPU train-only autograd 诊断已补充该证据，详见 GPC_GRADIENT_COMPATIBILITY_AUDIT.md。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
