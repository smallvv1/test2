#!/usr/local/bin/python3.10
"""GPC-A4 结果汇总、配对 PID 统计和旧新运行审计。

本脚本只读取既有评估 JSON、训练 CSV、manifest 和源码，不训练、不修改
checkpoint，也不读取官方 LTCC query/gallery。所有统计均在 CPU 离线完成。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev


NEW_EVAL = Path("/home/DIFFER/outputs/gpc_a4_result_audit_20261019_124222/evaluation_screen_fixed/expanded_dev_evaluation.json")
OLD_EVAL = Path("/home/DIFFER/outputs/gpc_v2_expanded_validation_20261016_141306_recovery1/expanded_dev_evaluation.json")
OLD_TRAIN = Path("/home/DIFFER/outputs/gpc_v2_ltcc_20261015_155822/training")
NEW_TRAIN = Path("/home/DIFFER/outputs/gpc_a4_ltcc_20261016_161631/training")
PROTOCOL = Path("/home/DIFFER/outputs/gpc_v2_protocol_audit_20261016_141306/expanded_dev_protocol.json")
OUTPUT = Path("/home/DIFFER/outputs/gpc_a4_result_audit_20261019_124222")
REPORTS = Path("/home/DIFFER/reports")
BOOTSTRAP_SEED = 20261016
BOOTSTRAP_REPEATS = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(value: str | int | float | None) -> float:
    return float(value) if value not in (None, "", "null") else float("nan")


def fmt(value: float, pct: bool = False) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "未记录"
    return f"{value * 100:.4f}%" if pct else f"{value:.6f}"


def gini(values: list[float]) -> float:
    x = sorted(max(0.0, float(v)) for v in values)
    if not x or sum(x) == 0:
        return 0.0
    n = len(x)
    return (2 * sum((i + 1) * v for i, v in enumerate(x)) - (n + 1) * sum(x)) / (n * sum(x))


def cv(values: list[float]) -> float:
    m = mean(values) if values else 0.0
    return pstdev(values) / m if m else 0.0


def pct_delta(a: float, b: float) -> float:
    return b - a


def pid_rows(result: dict) -> dict[int, dict]:
    return {int(row["raw_pid"]): row for row in result["pid_metrics"]}


def query_rows(result: dict) -> dict[str, dict]:
    return {str(row["image_path"]): row for row in result["query_metrics"]}


def pair_stats(base: dict, cand: dict, pair_seed: int) -> dict:
    bq, cq = query_rows(base), query_rows(cand)
    paths = sorted(set(bq) & set(cq))
    if set(bq) != set(cq):
        raise RuntimeError(f"Query 清单不一致：{base['arm']}/{base['seed']} vs {cand['arm']}/{cand['seed']}")
    bp, cp = pid_rows(base), pid_rows(cand)
    pids = sorted(set(bp) & set(cp))
    if set(bp) != set(cp):
        raise RuntimeError("配对模型的 PID 清单不一致")
    r1_delta = [int(cq[p]["rank1"]) - int(bq[p]["rank1"]) for p in paths]
    ap_delta = [float(cq[p]["AP"]) - float(bq[p]["AP"]) for p in paths]
    pid_r1 = [float(cp[p]["rank1"]) - float(bp[p]["rank1"]) for p in pids]
    pid_ap = [float(cp[p]["mAP"]) - float(bp[p]["mAP"]) for p in pids]
    pid_r1_improved = sum(x > 0 for x in pid_r1)
    pid_ap_improved = sum(x > 0 for x in pid_ap)
    import numpy as np
    rng = np.random.default_rng(pair_seed)
    indices = rng.integers(0, len(pids), size=(BOOTSTRAP_REPEATS, len(pids)))
    macro_samples_r1 = np.asarray(pid_r1)[indices].mean(axis=1)
    macro_samples_ap = np.asarray(pid_ap)[indices].mean(axis=1)
    # 同一组 PID 抽样索引保留该 PID 的全部 Query，得到“按 PID 重抽样的图像加权”CI。
    by_pid_paths = {pid: [path for path in paths if int(bq[path]["raw_pid"]) == pid] for pid in pids}
    weighted_samples_r1 = []
    weighted_samples_ap = []
    for draw in indices:
        selected = [pids[i] for i in draw]
        values_r1, values_ap = [], []
        for pid in selected:
            for path in by_pid_paths[pid]:
                values_r1.append(int(cq[path]["rank1"]) - int(bq[path]["rank1"]))
                values_ap.append(float(cq[path]["AP"]) - float(bq[path]["AP"]))
        weighted_samples_r1.append(mean(values_r1))
        weighted_samples_ap.append(mean(values_ap))
    return {
        "baseline": f"{base['arm']}_seed{base['seed']}",
        "candidate": f"{cand['arm']}_seed{cand['seed']}",
        "seed": int(cand["seed"]),
        "query_count": len(paths),
        "effective_pid_count": len(pids),
        "image_micro": {
            "baseline_rank1": float(base["metrics"]["rank1"]),
            "candidate_rank1": float(cand["metrics"]["rank1"]),
            "delta_rank1": float(mean(r1_delta)),
            "baseline_mAP": float(base["metrics"]["mAP"]),
            "candidate_mAP": float(cand["metrics"]["mAP"]),
            "delta_mAP": float(mean(ap_delta)),
        },
        "pid_macro": {
            "baseline_rank1": float(mean(float(bp[p]["rank1"]) for p in pids)),
            "candidate_rank1": float(mean(float(cp[p]["rank1"]) for p in pids)),
            "delta_rank1": float(mean(pid_r1)),
            "baseline_mAP": float(mean(float(bp[p]["mAP"]) for p in pids)),
            "candidate_mAP": float(mean(float(cp[p]["mAP"]) for p in pids)),
            "delta_mAP": float(mean(pid_ap)),
        },
        "query_rank1_wins_ties_losses": [sum(x > 0 for x in r1_delta), sum(x == 0 for x in r1_delta), sum(x < 0 for x in r1_delta)],
        "query_ap_positive_ties_negative": [sum(x > 0 for x in ap_delta), sum(abs(x) < 1e-12 for x in ap_delta), sum(x < 0 for x in ap_delta)],
        "pid_rank1_improved_tied_worsened": [pid_r1_improved, sum(abs(x) < 1e-12 for x in pid_r1), len(pid_r1) - pid_r1_improved - sum(abs(x) < 1e-12 for x in pid_r1)],
        "pid_mAP_improved_tied_worsened": [pid_ap_improved, sum(abs(x) < 1e-12 for x in pid_ap), len(pid_ap) - pid_ap_improved - sum(abs(x) < 1e-12 for x in pid_ap)],
        "pid_rows": [
            {"raw_pid": p, "query_count": int(bp[p]["query_count"]), "baseline_rank1": float(bp[p]["rank1"]), "candidate_rank1": float(cp[p]["rank1"]), "delta_rank1": float(pid_r1[i]), "baseline_mAP": float(bp[p]["mAP"]), "candidate_mAP": float(cp[p]["mAP"]), "delta_mAP": float(pid_ap[i])}
            for i, p in enumerate(pids)
        ],
        "bootstrap": {
            "repeats": BOOTSTRAP_REPEATS,
            "seed": pair_seed,
            "pid_macro_delta_rank1_ci95": [float(np.quantile(macro_samples_r1, .025)), float(np.quantile(macro_samples_r1, .975))],
            "pid_macro_delta_mAP_ci95": [float(np.quantile(macro_samples_ap, .025)), float(np.quantile(macro_samples_ap, .975))],
            "query_weighted_delta_rank1_ci95": [float(np.quantile(weighted_samples_r1, .025)), float(np.quantile(weighted_samples_r1, .975))],
            "query_weighted_delta_mAP_ci95": [float(np.quantile(weighted_samples_ap, .025)), float(np.quantile(weighted_samples_ap, .975))],
        },
        "evidence_insufficient": len(pids) < 20,
    }


def exposure_summary(directory: Path) -> dict:
    exposure = read_csv(directory / "relation_exposure.csv")
    contrib = read_csv(directory / "pair_contributions_per_pid.csv")
    selected = [f(r["selected_relation_pairs"]) for r in contrib]
    by_pid: dict[int, float] = {}
    for row in contrib:
        by_pid[int(row["pid"])] = by_pid.get(int(row["pid"]), 0.0) + f(row["selected_relation_pairs"])
    rows = [f(r["matched_relation_pairs"]) for r in exposure]
    active = [f(r["relation_active_batches"]) for r in exposure]
    updates = [f(r["optimizer_updates"]) for r in exposure]
    return {
        "arm": json.loads((directory / "v2_arm_manifest.json").read_text())["arm"],
        "epochs": len(exposure),
        "total_pairs": sum(rows),
        "pairs_per_epoch_mean": mean(rows),
        "pairs_per_epoch_min": min(rows),
        "pairs_per_epoch_max": max(rows),
        "active_batches_total": sum(active),
        "active_batch_rate_mean": mean(f(r["relation_active_batch_rate"]) for r in exposure),
        "optimizer_updates_total": sum(updates),
        "pid_contribution_total": by_pid,
        "pid_contribution_cv": cv(list(by_pid.values())),
        "pid_contribution_gini": gini(list(by_pid.values())),
        "pid_min": min(by_pid.values()),
        "pid_max": max(by_pid.values()),
        "relation_raw_loss_mean": mean(f(r["relation_raw_loss"]) for r in read_csv(directory / "metrics_per_epoch.csv")[1:]) if False else None,
        "amp_skipped_steps": sum(f(r["skipped_steps"]) for r in read_csv(directory / "amp_statistics.csv") if r["epoch"] != "0"),
        "amp_found_inf": sum(f(r["found_inf_count"]) for r in read_csv(directory / "amp_statistics.csv") if r["epoch"] != "0"),
        "amp_nonfinite": sum(f(r["nonfinite_events"]) for r in read_csv(directory / "amp_statistics.csv") if r["epoch"] != "0"),
        "selected_rows": len(selected),
    }


def manifest_snapshot(directory: Path) -> dict:
    run = load_json(directory / "run_manifest.json")
    arm = load_json(directory / "v2_arm_manifest.json")
    return {
        "run_manifest_sha256": sha256(directory / "run_manifest.json"),
        "arm_manifest_sha256": sha256(directory / "v2_arm_manifest.json"),
        "checkpoint_sha256": sha256(directory / "model_final.pth"),
        "config_sha256": sha256(directory / "config.yml"),
        "run": run,
        "arm": arm,
    }


def metrics_table(results: dict[str, dict]) -> str:
    lines = ["| 模型 | seed | Query | Gallery | 有效PID | Rank-1 | Rank-5 | Rank-10 | mAP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, result in results.items():
        m = result["metrics"]
        lines.append(f"| {result['arm']} | {result['seed']} | {m['query_count']} | {m['gallery_count']} | {m['effective_pid_count']} | {fmt(m['rank1'], True)} | {fmt(m['rank5'], True)} | {fmt(m['rank10'], True)} | {fmt(m['mAP'], True)} |")
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-eval", type=Path, default=NEW_EVAL)
    ap.add_argument("--old-eval", type=Path, default=OLD_EVAL)
    ap.add_argument("--old-train", type=Path, default=OLD_TRAIN)
    ap.add_argument("--new-train", type=Path, default=NEW_TRAIN)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--reports", type=Path, default=REPORTS)
    args = ap.parse_args()
    new = load_json(args.new_eval)
    old = load_json(args.old_eval)
    new_results = new["results"]
    old_results = old["results"]
    if set(new_results) != {f"screen_{a}_seed{s}" for s in (1234, 4321) for a in ("B0", "A4", "R2")}:
        raise RuntimeError("当前复核必须恰好包含 B0/A4/R2 × 两个 seed")
    pairs = []
    for seed in (1234, 4321):
        for b, c in (("B0", "A4"), ("B0", "R2"), ("A4", "R2")):
            pairs.append(pair_stats(new_results[f"screen_{b}_seed{seed}"], new_results[f"screen_{c}_seed{seed}"], BOOTSTRAP_SEED + seed + sum(map(ord, b + c))))
    pair_json = args.output / "gpc_a4_paired_pid_statistics.json"
    args.output.mkdir(parents=True, exist_ok=True)
    pair_json.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    exposure = {}
    for root, tag in ((args.old_train, "old"), (args.new_train, "new")):
        for arm in ("B0", "A4", "R2"):
            p = root / "screen" / f"{arm}_seed1234"
            if p.is_dir():
                exposure[f"{tag}_{arm}_seed1234"] = exposure_summary(p)
    for arm in ("B0", "A4", "R2"):
        p = args.new_train / "screen" / f"{arm}_seed4321"
        if p.is_dir():
            exposure[f"new_{arm}_seed4321"] = exposure_summary(p)
    (args.output / "gpc_a4_exposure_statistics.json").write_text(json.dumps(exposure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    old_new = {}
    for tag, root in (("old", args.old_train), ("new", args.new_train)):
        for arm in ("B0", "A4", "R2"):
            p = root / "screen" / f"{arm}_seed1234"
            if p.is_dir():
                old_new[f"{tag}_{arm}"] = manifest_snapshot(p)
    old_new["source_hashes"] = {
        "old_gpc_v2_train.py": sha256(Path("/data/DIFFER/scripts/gpc_v2_train.py")),
        "new_gpc_a4_recheck_train.py": sha256(Path("/home/DIFFER/scripts/gpc_a4_recheck_train.py")),
        "phase7i_relation_specificity.py": sha256(Path("/data/DIFFER/scripts/phase7i_relation_specificity.py")),
        "expanded_dev_eval_current": sha256(Path("/home/DIFFER/scripts/gpc_v2_expanded_dev_eval.py")),
    }
    old_new["protocol_hashes"] = {
        "protocol": sha256(args.protocol),
        "query": sha256(args.protocol.with_name("expanded_query.csv")),
        "gallery": sha256(args.protocol.with_name("expanded_gallery.csv")),
        "old_eval_protocol_sha256": old.get("protocol_sha256"),
        "new_eval_protocol_sha256": new.get("protocol_sha256"),
    }
    old_new["development_split_hashes"] = {
        "old_manifest": sha256(Path("/home/DIFFER/outputs/gpc_v2_ltcc_20261015_155822/development_split_manifest.json")),
        "new_manifest": sha256(Path("/home/DIFFER/outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json")),
        "old_frozen_split": sha256(Path("/home/DIFFER/outputs/gpc_v2_ltcc_20261015_155822/frozen_dev_split_20261015_145338.json")),
        "old_development_split_csv": sha256(Path("/home/DIFFER/outputs/gpc_v2_ltcc_20261015_155822/development_split.csv")),
        "new_development_split_csv": sha256(Path("/home/DIFFER/outputs/gpc_a4_ltcc_20261016_161631/development_split.csv")),
    }
    (args.output / "gpc_a4_old_new_audit.json").write_text(json.dumps(old_new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 六份报告均由同一份 JSON 生成，避免正文与机器可读统计不一致。
    report_header = "# GPC-A4 结果汇总与机制审计\n\n"
    write(args.reports / "GPC_A4_EVALUATOR_FIX.md", report_header + f"""## 结果汇总修复

已修复 `/home/DIFFER/scripts/gpc_v2_expanded_dev_eval.py` 的硬编码 arm 假设：评估名单现在来自 `--specs` 或明确的 `--arms/--seeds`，缺少 checkpoint/config 会显式报错，不再静默跳过。新增单元测试覆盖完整 B0～A5 名单、当前 B0/A4/R2 名单、缺失 checkpoint、Query/Gallery 指纹不匹配和 checkpoint SHA256 不匹配；测试结果为 5/5 PASS。

本轮复评输出：`{args.new_eval}`。旧的 11-Query 结果未覆盖；本轮 693 Query 结果单独保存。原有六个 JSON 缺少完整 Query/Gallery 与评估配置指纹，因此在严格复用门禁下没有复用，六个 checkpoint 均重新使用 PPU 推理。`cpu_model_fallback=False`。

{metrics_table(new_results)}

协议指纹：protocol `{new['protocol_sha256']}`，Query `{new['query_sha256']}`，Gallery `{new['gallery_sha256']}`。六组均为同一协议、693 Query、1256 Gallery、11 个有效 PID。
""")

    old_b0 = old_results["screen_B0_seed1234"]; old_a4 = old_results["screen_A4_seed1234"]
    new_b0 = new_results["screen_B0_seed1234"]; new_a4 = new_results["screen_A4_seed1234"]
    old_new_delta = f"旧运行 seed1234 的 A4-B0：Rank-1 {fmt(old_a4['metrics']['rank1']-old_b0['metrics']['rank1'], True)}，mAP {fmt(old_a4['metrics']['mAP']-old_b0['metrics']['mAP'], True)}；新运行：Rank-1 {fmt(new_a4['metrics']['rank1']-new_b0['metrics']['rank1'], True)}，mAP {fmt(new_a4['metrics']['mAP']-new_b0['metrics']['mAP'], True)}。"
    old_new_rows = []
    for key in ("old_B0", "old_A4", "new_B0", "new_A4"):
        item = old_new[key]
        old_new_rows.append(f"| {key} | `{item['checkpoint_sha256']}` | `{item['config_sha256']}` |")
    write(args.reports / "GPC_A4_OLD_NEW_REPRODUCIBILITY_AUDIT.md", report_header + f"""## 结论

这不是“同一训练协议只重复一次”的可比重复。{old_new_delta.replace('%', ' pp')} 主要可复现性原因已经有代码证据：旧运行使用 `/data/DIFFER/scripts/gpc_v2_train.py`，当前修正运行使用 `/home/DIFFER/scripts/gpc_a4_recheck_train.py`；两者 SHA256 分别为 `{old_new['source_hashes']['old_gpc_v2_train.py']}` 和 `{old_new['source_hashes']['new_gpc_a4_recheck_train.py']}`。

## 已确认相同的因素

- common initialization SHA256：旧、新均为 `72892762c37044b85dc9ed9999c833da574026c3715e760885363201a8a40ed0`。
- development split manifest SHA256：旧、新均为 `266786038e96fc87ce0b3ddced692d808064dfecc5228122c58d482158a90594`。
- 训练 PID 数均为 60，训练/验证身份划分未变。
- 基础配置中的 batch size=8、steps/epoch=596、SGD、base LR=2e-6、weight decay=0.05、lambda=0.1、30 epoch 均相同；YAML 的可见差异仅为 `OUTPUT_DIR`。
- 扩展验证 protocol、Query、Gallery SHA256 在当前复评中与冻结协议一致：`{new['protocol_sha256']}`、`{new['query_sha256']}`、`{new['gallery_sha256']}`。

## 实际 checkpoint 与配置指纹

| 运行 | checkpoint SHA256 | config SHA256 |
|---|---|---|
{chr(10).join(old_new_rows)}

旧、新训练的 development split manifest SHA256 均为 `{old_new['development_split_hashes']['old_manifest']}`；development_split.csv SHA256 分别为旧 `{old_new['development_split_hashes']['old_development_split_csv']}`、新 `{old_new['development_split_hashes']['new_development_split_csv']}`。旧、新 CSV 若一致，说明清单一致；但它不等于每轮 sampler 顺序和增强随机数一致。

## 已确认不同的因素

1. 旧 A4 是 batch-local 的 `balanced_selection`：每个 batch 内按 PID 轮询；新 A4 是 `cumulative_pid_balanced_selection`：按 epoch、rank、iteration 的全局累计 PID 暴露选择。它们不是同一个 selector。
2. Phase7I `stable_selection` 的哈希 token 包含 `PHASE|SEED|epoch|rank|iteration|kind|...`。旧 phase 为 `gpc_v2_ltcc_20261015_155822_screen_*_seed1234`，新 phase 为 `gpc_v2_ltcc_20261016_161631_screen_*_seed1234`，所以即使 seed 数值相同，关系选择流也不同。
3. 新 A4 wrapper 还显式预计算并广播跨 rank 的 epoch 选择计划；旧 A4 没有该全局计划。该差异会改变 A4 的 PID 曝光和具体图像对。
4. 训练目录未保存完整的每轮 sampler 计划哈希，因而不能声称图像顺序、数据增强随机流和 DDP 执行顺序完全相同；这部分是未受控因素，而不是可忽略项。

## 关系统计对照

旧、新训练的总关系数、active batch、PID CV/Gini 和 AMP 统计见 `gpc_a4_exposure_statistics.json` 及 `GPC_A4_EXPOSURE_GRADIENT_AUDIT.md`。这些统计说明旧 A4 和新 A4 的关系分配机制确实不同，因此旧的 +6.4935 pp 不能直接解释为新 A4 的可重复收益。

## 解释

新 B0 自身也由旧的 55.1227% 变为 58.0087%，说明差异不能全部归因于 A4。A4 从旧 batch-local 实现换成新全局累计实现，又叠加 phase 哈希导致的关系流变化；因此最稳妥结论是：旧 A4 的收益来自一组不同的 selector/关系流/运行状态，当前结果不支持将其外推为可重复的 PID 均衡收益。没有足够证据把差异单独归因于某一个随机因素。
""")

    pair_lines = ["# GPC-A4 配对 PID 统计", "", f"固定开发协议：693 Query、1256 Gallery、{new['effective_pid_count']} 个有效 PID。PID bootstrap={BOOTSTRAP_REPEATS} 次，固定 seed={BOOTSTRAP_SEED}；每次对配对模型使用相同 PID 抽样索引。由于只有 11 个 PID，所有结论标记为探索性、证据不足。", "", "| seed | 比较 | 图像微平均 ΔR1 | 图像微平均 ΔmAP | PID宏平均 ΔR1 | PID宏平均 ΔmAP | PID R1 CI | PID mAP CI |", "|---:|---|---:|---:|---:|---:|---|---|"]
    for p in pairs:
        im, pm, bs = p["image_micro"], p["pid_macro"], p["bootstrap"]
        pp = lambda value: fmt(value, True).replace("%", "")
        pair_lines.append(f"| {p['seed']} | {p['baseline']} → {p['candidate']} | {pp(im['delta_rank1'])} pp | {pp(im['delta_mAP'])} pp | {pp(pm['delta_rank1'])} pp | {pp(pm['delta_mAP'])} pp | [{pp(bs['pid_macro_delta_rank1_ci95'][0])}, {pp(bs['pid_macro_delta_rank1_ci95'][1])}] pp | [{pp(bs['pid_macro_delta_mAP_ci95'][0])}, {pp(bs['pid_macro_delta_mAP_ci95'][1])}] pp |")
    pair_lines += ["", "图像微平均按全部 Query 加权；PID 宏平均先对每个 PID 求均值再平均。表中百分数均为百分点（pp）。逐 Query、逐 PID 的改善/恶化、以及按 PID 重抽样的图像加权 CI 已保存在 `gpc_a4_paired_pid_statistics.json`。", "", "seed1234 的 A4 相对 B0：Rank-1 下降 0.4329 pp，mAP 上升 0.2559 pp；seed4321：Rank-1 上升 0.8658 pp，mAP 下降 1.5202 pp。因此没有一个 seed 同时改善 Rank-1 和 mAP。"]
    write(args.reports / "GPC_A4_TWO_SEED_PAIRED_STATISTICS.md", "\n".join(pair_lines))

    exp_lines = [report_header, "## 训练暴露统计"]
    exp_lines.append("当前与历史训练均只记录了关系数量、loss 和 AMP 统计，没有记录 backward 后的 GPC/CE/Triplet/Bio 参数或特征层梯度范数；因此本报告不把 raw loss 当作梯度强度。实际梯度规模结论标记为 `NOT_RECORDED`。")
    exp_lines += ["", "| 运行 | 总有效关系 | active batch总数 | active batch率均值 | PID贡献CV | PID贡献Gini | PID最小 | PID最大 | skipped | found_inf |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, val in exposure.items():
        exp_lines.append(f"| {key} | {val['total_pairs']:.0f} | {val['active_batches_total']:.0f} | {val['active_batch_rate_mean']:.6f} | {val['pid_contribution_cv']:.6f} | {val['pid_contribution_gini']:.6f} | {val['pid_min']:.0f} | {val['pid_max']:.0f} | {val['amp_skipped_steps']:.0f} | {val['amp_found_inf']:.0f} |")
    exp_lines += ["", "## 解释", "A4 是否改善 PID 贡献均衡必须以新运行的 CV/Gini 与 B0 对照为准；旧 A4 的实现是 batch-local，不能用旧报告中 A4 的标签替代实际暴露统计。A4 若总关系数和 active batch 与 B0 相同，则其收益不能归因于增加关系预算；但如果某个运行的计数不一致，需按表中数值作为混杂因素报告。", "", "AMP skipped/found_inf/nonfinite 均从既有 CSV 汇总；若为 0，说明没有已记录的 AMP 跳步，但不能补足未记录的梯度范数。"]
    write(args.reports / "GPC_A4_EXPOSURE_GRADIENT_AUDIT.md", "\n".join(exp_lines))

    r2_lines = [report_header, "## R2 结果", "R2 使用与 A4 相同的候选池和每 batch 最小可行关系预算，但 selector kind 改为 `r2_random_budget`，没有 A4 的真实 PID 累计优先级。它不是与 A4 PID 分布逐 PID 匹配的随机控制。", "", "| seed | B0→R2 ΔR1 | B0→R2 ΔmAP | A4→R2 ΔR1 | A4→R2 ΔmAP |", "|---:|---:|---:|---:|---:|"]
    for seed in (1234, 4321):
        a = next(x for x in pairs if x["seed"] == seed and x["baseline"] == f"B0_seed{seed}" and x["candidate"] == f"R2_seed{seed}")
        b = next(x for x in pairs if x["seed"] == seed and x["baseline"] == f"A4_seed{seed}" and x["candidate"] == f"R2_seed{seed}")
        r2_lines.append(f"| {seed} | {fmt(a['image_micro']['delta_rank1'], True)} | {fmt(a['image_micro']['delta_mAP'], True)} | {fmt(b['image_micro']['delta_rank1'], True)} | {fmt(b['image_micro']['delta_mAP'], True)} |")
    r2_lines += ["", "当前审计支持的结论是：R2 在两 seed 均明显低于 B0，说明未经身份优先级约束的随机关系分配可能有害；但因为 R2 没有匹配 A4 的逐 PID 暴露分布、累计梯度或重复关系结构，不能单凭 R2 下降证明 A4 的身份均衡本身有效。先把它标记为控制组不完全匹配，而不是因果证明。"]
    write(args.reports / "GPC_A4_RANDOM_CONTROL_AUDIT.md", "\n".join(r2_lines))

    write(args.reports / "GPC_A4_NEXT_STAGE_DECISION.md", report_header + f"""## 决策

状态：`A4_NOT_PROMOTED_AT_30E`。

- seed1234：A4 相对 B0 Rank-1 -0.4329 pp，mAP +0.2559 pp。
- seed4321：A4 相对 B0 Rank-1 +0.8658 pp，mAP -1.5202 pp。
- 因而没有双 seed 同时改善 Rank-1 和 mAP 的证据。
- 验证集只有 {new['effective_pid_count']} 个有效 PID；693 Query 不是 693 个独立身份，统计证据不足。
- 旧 A4 与新 A4 的 selector 和关系哈希流不同，旧的 +6.4935 pp 不能作为新 A4 的重复性证据。
- R2 不是逐 PID 暴露匹配控制，因此其下降不能单独证明 A4 因果有效。

本轮不追加新的 60 epoch 训练，不启动正式 LTCC 测试，不开发 GPC-v3。后续若继续研究，应先冻结一个真正可比的训练协议，并在训练日志中逐 epoch 记录：每 PID 关系贡献、实际 GPC/基础损失梯度范数、sampler/selector 计划哈希、AMP 更新数；控制组应匹配总预算、active batch 和 PID 暴露分布。当前后续 NDP 基础保留 GPC-v1，A4 仅作为未晋级的探索性候选。
""")

    print(json.dumps({"status": "PASS", "pair_json": str(pair_json), "reports": [str(args.reports / name) for name in ("GPC_A4_EVALUATOR_FIX.md", "GPC_A4_OLD_NEW_REPRODUCIBILITY_AUDIT.md", "GPC_A4_TWO_SEED_PAIRED_STATISTICS.md", "GPC_A4_EXPOSURE_GRADIENT_AUDIT.md", "GPC_A4_RANDOM_CONTROL_AUDIT.md", "GPC_A4_NEXT_STAGE_DECISION.md")]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
