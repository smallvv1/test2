#!/usr/local/bin/python3.10
"""GPC-v2 扩展开发验证的 PID 配对 Bootstrap 与报告生成。

只读取已完成的 PPU checkpoint 评估 JSON；不训练、不读取官方 query/gallery，
也不修改历史结果。Bootstrap 的重采样单位是 PID，比较两模型时复用同一组索引。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/DIFFER")
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20261016


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pid_macro(result: dict[str, Any]) -> dict[str, float]:
    rows = result["pid_metrics"]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in ("rank1", "rank5", "rank10", "mAP")}


def result_row(label: str, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    macro = pid_macro(result)
    return {
        "label": label,
        "stage": result["stage"],
        "arm": result["arm"],
        "seed": result["seed"],
        "query_count": metrics["query_count"],
        "gallery_count": metrics["gallery_count"],
        "effective_pid_count": metrics["effective_pid_count"],
        "rank1": float(metrics["rank1"]),
        "rank5": float(metrics["rank5"]),
        "rank10": float(metrics["rank10"]),
        "mAP": float(metrics["mAP"]),
        "pid_macro_rank1": macro["rank1"],
        "pid_macro_rank5": macro["rank5"],
        "pid_macro_rank10": macro["rank10"],
        "pid_macro_mAP": macro["mAP"],
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
    }


def paired(baseline: dict[str, Any], candidate: dict[str, Any], pair_name: str, pair_index: int) -> dict[str, Any]:
    base_pid = {int(row["raw_pid"]): row for row in baseline["pid_metrics"]}
    cand_pid = {int(row["raw_pid"]): row for row in candidate["pid_metrics"]}
    pids = sorted(set(base_pid) & set(cand_pid))
    if not pids:
        raise RuntimeError(f"{pair_name} 没有共同 PID")
    base_q = {row["image_path"]: row for row in baseline["query_metrics"]}
    cand_q = {row["image_path"]: row for row in candidate["query_metrics"]}
    paths = sorted(set(base_q) & set(cand_q))

    base_r1 = np.asarray([float(base_pid[pid]["rank1"]) for pid in pids], dtype=np.float64)
    cand_r1 = np.asarray([float(cand_pid[pid]["rank1"]) for pid in pids], dtype=np.float64)
    base_map = np.asarray([float(base_pid[pid]["mAP"]) for pid in pids], dtype=np.float64)
    cand_map = np.asarray([float(cand_pid[pid]["mAP"]) for pid in pids], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + pair_index)
    indices = rng.integers(0, len(pids), size=(BOOTSTRAP_REPEATS, len(pids)))
    delta_r1 = (cand_r1[indices] - base_r1[indices]).mean(axis=1)
    delta_map = (cand_map[indices] - base_map[indices]).mean(axis=1)

    pid_rows = []
    for pid in pids:
        pid_rows.append({
            "pair": pair_name,
            "raw_pid": pid,
            "baseline_rank1": float(base_pid[pid]["rank1"]),
            "candidate_rank1": float(cand_pid[pid]["rank1"]),
            "delta_rank1": float(cand_pid[pid]["rank1"] - base_pid[pid]["rank1"]),
            "baseline_mAP": float(base_pid[pid]["mAP"]),
            "candidate_mAP": float(cand_pid[pid]["mAP"]),
            "delta_mAP": float(cand_pid[pid]["mAP"] - base_pid[pid]["mAP"]),
            "query_count": int(cand_pid[pid]["query_count"]),
        })

    query_rows = []
    for path in paths:
        query_rows.append({
            "pair": pair_name,
            "image_path": path,
            "raw_pid": int(cand_q[path]["raw_pid"]),
            "baseline_rank1": int(base_q[path]["rank1"]),
            "candidate_rank1": int(cand_q[path]["rank1"]),
            "delta_rank1": int(cand_q[path]["rank1"] - base_q[path]["rank1"]),
            "baseline_AP": float(base_q[path]["AP"]),
            "candidate_AP": float(cand_q[path]["AP"]),
            "delta_AP": float(cand_q[path]["AP"] - base_q[path]["AP"]),
        })

    return {
        "pair": pair_name,
        "baseline": baseline["arm"],
        "candidate": candidate["arm"],
        "seed": int(candidate["seed"]),
        "common_pid_count": len(pids),
        "common_query_count": len(paths),
        "statistical_evidence": "LIMITED_11_PID" if len(pids) < 20 else "ADEQUATE_FOR_THIS_EXPLORATORY_AUDIT",
        "pid_bootstrap": {
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED + pair_index,
            "same_resample_indices_for_baseline_and_candidate": True,
            "delta_rank1_mean": float(np.mean(cand_r1 - base_r1)),
            "delta_rank1_ci95": [float(np.quantile(delta_r1, 0.025)), float(np.quantile(delta_r1, 0.975))],
            "delta_mAP_mean": float(np.mean(cand_map - base_map)),
            "delta_mAP_ci95": [float(np.quantile(delta_map, 0.025)), float(np.quantile(delta_map, 0.975))],
        },
        "query_pair_counts": {
            "rank1_wins": int(sum(row["delta_rank1"] > 0 for row in query_rows)),
            "rank1_ties": int(sum(row["delta_rank1"] == 0 for row in query_rows)),
            "rank1_losses": int(sum(row["delta_rank1"] < 0 for row in query_rows)),
            "AP_improved": int(sum(row["delta_AP"] > 0 for row in query_rows)),
            "AP_tied": int(sum(row["delta_AP"] == 0 for row in query_rows)),
            "AP_worsened": int(sum(row["delta_AP"] < 0 for row in query_rows)),
        },
        "pid_rows": pid_rows,
        "query_rows": query_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    evaluation = load_json(args.evaluation)
    protocol = load_json(args.protocol)
    results = evaluation["results"]
    rows = [result_row(label, result) for label, result in results.items()]
    pairs = []
    for index, arm in enumerate(("A1", "A2", "A3", "A4", "A5")):
        pairs.append(paired(results["screen_B0_seed1234"], results[f"screen_{arm}_seed1234"], f"E30_B0_vs_{arm}_seed1234", index))
    pairs.append(paired(results["two_seed_B0_seed1234"], results["two_seed_A5_seed1234"], "E60_B0_vs_A5_seed1234", 10))
    pairs.append(paired(results["two_seed_B0_seed4321"], results["two_seed_A5_seed4321"], "E60_B0_vs_A5_seed4321", 11))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "model_metrics.csv", rows)
    pid_rows = [row for item in pairs for row in item["pid_rows"]]
    query_rows = [row for item in pairs for row in item["query_rows"]]
    write_csv(args.output / "paired_pid_deltas.csv", pid_rows)
    write_csv(args.output / "paired_query_deltas.csv", query_rows)
    payload = {
        "evaluation": str(args.evaluation),
        "evaluation_sha256": sha256(args.evaluation),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "protocol_status": protocol["status"],
        "query_count": evaluation["query_count"],
        "gallery_count": evaluation["gallery_count"],
        "effective_pid_count": evaluation["effective_pid_count"],
        "model_metrics": rows,
        "paired_comparisons": pairs,
        "historical_old_results_preserved": str(ROOT / "outputs/gpc_v2_ltcc_20261015_155822/dev_evaluation_screen.json"),
        "official_query_gallery_used": False,
        "cpu_model_fallback": False,
    }
    write_json(args.output / "paired_pid_bootstrap.json", payload)

    lines = [
        "# GPC-v2 配对 PID Bootstrap 复核",
        "",
        "本报告只使用冻结开发验证身份和既有 E30/E60 checkpoint；旧 11-query 结果保留在历史目录，未被覆盖。",
        "",
        f"- 协议：`{args.protocol}`；SHA256：`{sha256(args.protocol)}`。",
        f"- Query：`{evaluation['query_count']}`；Gallery：`{evaluation['gallery_count']}`；有效 PID：`{evaluation['effective_pid_count']}`。",
        f"- PID Bootstrap：`{BOOTSTRAP_REPEATS}` 次；以 PID 为重采样单位；同一比较中两模型复用相同抽样索引。",
        "- 统计含义：图像微平均指标来自所有 Query；PID 宏平均指标先对每个 PID 的 Query 指标取均值，再对 PID 取均值。",
        "",
        "## 模型结果",
        "",
        "| 模型 | seed | Query | Gallery | PID | 图像 R1 | 图像 R5 | 图像 R10 | 图像 mAP | PID宏R1 | PID宏mAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {label} | {seed} | {query_count} | {gallery_count} | {effective_pid_count} | {rank1:.4%} | {rank5:.4%} | {rank10:.4%} | {mAP:.4%} | {pid_macro_rank1:.4%} | {pid_macro_mAP:.4%} |".format(**row))
    lines += ["", "## 配对比较", "", "| 比较 | PID | Query | ΔRank-1均值 | ΔRank-1 CI95 | ΔmAP均值 | ΔmAP CI95 | R1 胜/平/负 | AP 改善/平/恶化 |", "|---|---:|---:|---:|---|---:|---|---|---|"]
    for item in pairs:
        boot = item["pid_bootstrap"]
        qc = item["query_pair_counts"]
        lines.append(f"| {item['pair']} | {item['common_pid_count']} | {item['common_query_count']} | {boot['delta_rank1_mean']:+.4%} | [{boot['delta_rank1_ci95'][0]:+.4%}, {boot['delta_rank1_ci95'][1]:+.4%}] | {boot['delta_mAP_mean']:+.4%} | [{boot['delta_mAP_ci95'][0]:+.4%}, {boot['delta_mAP_ci95'][1]:+.4%}] | {qc['rank1_wins']}/{qc['rank1_ties']}/{qc['rank1_losses']} | {qc['AP_improved']}/{qc['AP_tied']}/{qc['AP_worsened']} |")
    lines += [
        "",
        "## 逐 PID 与逐 Query 明细",
        "",
        "- 逐 PID 改善/恶化：`paired_pid_deltas.csv`。",
        "- 逐 Query 改善/恶化：`paired_query_deltas.csv`。",
        "- 完整 JSON：`paired_pid_bootstrap.json`。",
        "- 当前有效 PID 只有 11 个，因此所有配对结论均标记为探索性、统计证据有限，不能替代独立确认实验。",
        "",
        "## 防泄漏与运行约束",
        "",
        "- 没有使用官方 LTCC query/gallery；模型 forward 使用 PPU，CPU 仅用于离线指标与 Bootstrap。",
        "- `cpu_model_fallback=False`；历史 11-query 文件未修改。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "report": str(args.report), "query_count": evaluation["query_count"], "effective_pid_count": evaluation["effective_pid_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
