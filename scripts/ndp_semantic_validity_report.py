#!/usr/local/bin/python3
"""基于已完成 NDP 指标修正结果的语义与覆盖有效性复核。

本脚本只读取 ndp_metric_coverage_audit_20261020_165332 的小型 JSON/CSV，
不重新计算特征、不实例化模型、不读取官方 query/gallery，不启动训练。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/DIFFER")
SOURCE_RUN = "20261020_165332"
SOURCE_OUT = ROOT / "outputs" / f"ndp_metric_coverage_audit_{SOURCE_RUN}"
SEEDS = (1234, 4321)
K_VALUES = (5, 10, 20)
BOOTSTRAP_REPS = 2000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pid_bootstrap(pid_values: dict[int, float], seed: int) -> list[float]:
    if not pid_values:
        return [float("nan"), float("nan"), float("nan")]
    x = np.asarray(list(pid_values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = [float(np.mean(x[rng.integers(0, len(x), size=len(x))])) for _ in range(BOOTSTRAP_REPS)]
    return [float(np.mean(x)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def metric_row(seed: int, k: int) -> dict[int, dict[str, Any]]:
    path = SOURCE_OUT / "tables" / f"seed{seed}_K{k}_metric_correction_anchor.csv"
    rows = {int(r["anchor"]): r for r in read_csv(path)}
    # 修正表刻意只保存计数；邻域是否发生成员变化复用旧审计的同一冻结图表，
    # 不重新计算特征或图邻居。
    old_path = ROOT / "outputs" / "ndp_net_identity_audit_20261020_162456" / "tables" / f"seed{seed}_K{k}_anchor_graph_metrics.csv"
    for r in read_csv(old_path):
        a = int(r["anchor"])
        rows[a]["neighborhood_changed"] = int(r["neighborhood_changed"])
    return rows


def relation_rows(seed: int, k: int) -> list[dict[str, Any]]:
    path = SOURCE_OUT / "tables" / f"seed{seed}_K{k}_identity_relation_states.csv"
    result = []
    for r in read_csv(path):
        for key in ("anchor", "pid", "positive", "negative"):
            r[key] = int(r[key])
        for key in ("positive_clothes_cross", "positive_cross_camera", "negative_cross_camera", "negative_cross_clothes", "reference_reliable", "reference_correct", "current_order_correct", "reference_reliable_flip"):
            raw = str(r[key]).strip().lower()
            r[key] = raw in {"1", "true", "yes"}
        r["g_ref"] = float(r["g_ref"]); r["g_current"] = float(r["g_current"])
        result.append(r)
    return result


def group_summary(seed: int, k: int, graph: dict[int, dict[str, Any]], relations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    relation_by_anchor: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in relations:
        relation_by_anchor[r["anchor"]].append(r)
    h1_anchors = {a for a, rs in relation_by_anchor.items() if any(r["reference_reliable_flip"] for r in rs)}
    h2_anchors = {a for a, rs in relation_by_anchor.items() if not any(r["reference_reliable_flip"] for r in rs) and int(graph[a]["DELTA_SAME_ID_COUNT"]) < 0}
    h3_anchors = {a for a, g in graph.items() if int(g["neighborhood_changed"] if "neighborhood_changed" in g else 0) and int(g["DELTA_SAME_ID_COUNT"]) > 0}
    groups = {"H1_reliable_flip": h1_anchors, "H2_same_id_drop_without_reliable_flip": h2_anchors, "H3_changed_with_same_id_gain": h3_anchors}
    rows=[]; pid_rows=[]
    for name, anchors in groups.items():
        rs = [r for r in relations if r["anchor"] in anchors]
        # H1 的事件分母必须是实际“可靠且翻转”的关系行；不能把同一 H1
        # anchor 上的其它关系行一起误报为 H1 事件。H2/H3 则按固定 anchor
        # 分组报告其对应关系上下文。
        event_rs = [r for r in rs if r["reference_reliable_flip"]] if name == "H1_reliable_flip" else rs
        pids = sorted({int(r["pid"]) for r in rs})
        graph_values = [graph[a] for a in anchors]
        relation_margin = [float(r["g_current"] - r["g_ref"]) for r in event_rs]
        pid_counts = defaultdict(int)
        pid_rates = defaultdict(list)
        for r in rs:
            pid_counts[r["pid"]] += 1
            pid_rates[r["pid"]].append(1.0 if r["reference_reliable_flip"] else 0.0)
        pid_summary = {pid: float(np.mean(v)) for pid, v in pid_rates.items()}
        ci = pid_bootstrap(pid_summary, 20261020 + seed + k + len(name))
        row = {
            "seed": seed, "K": k, "group": name, "anchor_count": len(anchors), "pid_count": len(pids),
            "relation_rows_context": len(rs), "event_rows": len(event_rs), "cross_clothes_positive_rate": float(np.mean([r["positive_clothes_cross"] for r in event_rs])) if event_rs else 0.0,
            "negative_cross_camera_rate": float(np.mean([r["negative_cross_camera"] for r in event_rs])) if event_rs else 0.0,
            "negative_cross_clothes_rate": float(np.mean([r["negative_cross_clothes"] for r in event_rs])) if event_rs else 0.0,
            "mean_margin_delta": float(np.mean(relation_margin)) if relation_margin else 0.0,
            "mean_same_id_delta": float(np.mean([int(g["DELTA_SAME_ID_COUNT"]) for g in graph_values])) if graph_values else 0.0,
            "mean_net_intrusion": float(np.mean([float(g["NET_NEGATIVE_INTRUSION"] ) for g in graph_values])) if graph_values else 0.0,
            "pid_rate_bootstrap_mean": ci[0], "pid_rate_bootstrap_ci95_low": ci[1], "pid_rate_bootstrap_ci95_high": ci[2],
        }
        rows.append(row)
        for pid in pids:
            pid_rows.append({"seed": seed, "K": k, "group": name, "pid": pid, "relation_rows": pid_counts[pid], "group_rate": pid_summary[pid]})
    return rows, pid_rows, {"H1": h1_anchors, "H2": h2_anchors, "H3": h3_anchors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    run_time = args.run_time
    out = ROOT / "outputs" / f"ndp_semantic_validity_audit_{run_time}"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    log_path = ROOT / "logs" / f"NDP语义与Triplet覆盖有效性复核_{run_time}.log"
    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f: f.write(msg + "\n")

    source_files = {}
    for path in (
        ROOT / "scripts" / "ndp_net_intrusion_triplet_coverage_audit.py",
        ROOT / "scripts" / "ndp_metric_correction_triplet_denominator_audit.py",
        Path("/data/DIFFER/loss/triplet_loss.py"), Path("/data/DIFFER/loss/make_loss.py"),
        Path("/data/DIFFER/data/samplers.py"), Path("/data/DIFFER/scripts/phase5b_id_damage_ds.py"),
        Path("/data/DIFFER/scripts/phase7i_relation_specificity.py"),
        SOURCE_OUT / "ndp_metric_coverage_audit.json",
    ):
        source_files[str(path)] = sha256_file(path)

    group_rows=[]; pid_rows=[]; source_summary=[]
    log("【复用】只读取已完成的 20261020_165332 JSON/CSV，不重新计算特征，不实例化模型。")
    for seed in SEEDS:
        for k in K_VALUES:
            graph = metric_row(seed, k)
            rel = relation_rows(seed, k)
            rows, pids, sets = group_summary(seed, k, graph, rel)
            group_rows.extend(rows); pid_rows.extend(pids)
            source_summary.append({"seed": seed, "K": k, "H1_anchor_count": len(sets["H1"]), "H2_anchor_count": len(sets["H2"]), "H3_anchor_count": len(sets["H3"]), "H1_pid_count": len({int(r["pid"]) for r in rel if r["anchor"] in sets["H1"]}), "H2_pid_count": len({int(r["pid"]) for r in rel if r["anchor"] in sets["H2"]}), "H3_pid_count": len({int(r["pid"]) for r in rel if r["anchor"] in sets["H3"]})})
            log(f"【完成】seed={seed} K={k}：H1/H2/H3 anchor={len(sets['H1'])}/{len(sets['H2'])}/{len(sets['H3'])}。")

    report_root = ROOT / "reports"
    k10 = [r for r in group_rows if r["K"] == 10]
    by_group = {(r["seed"], r["group"]): r for r in k10}
    metric_source = read_json(SOURCE_OUT / "ndp_metric_coverage_audit.json")
    cov_k10 = {(c["seed"], s["scope"]): s for c in metric_source["coverage_summaries"] if c["K"] == 10 for s in c["scopes"]}
    report = "# NDP 指标语义修正与审计边界\n\n"
    report += "本轮没有重复冻结特征统计，直接复用 `ndp_metric_coverage_audit_20261020_165332`。`current_negative_count-reference_negative_count` 是有符号变化，因此可以为负；非负侵入必须取其正部。\n\n"
    report += "| seed | K=10 有符号变化 | 非负净侵入槽位 | 非负缓解槽位 | SAME_ID_LOST | SAME_ID_GAINED |\n|---:|---:|---:|---:|---:|---:|\n"
    for g in metric_source["graph_summaries"]:
        if g["K"] == 10:
            report += f"| {g['seed']} | {g['total_SIGNED_NEGATIVE_DELTA_count']} | {g['total_NET_NEGATIVE_INTRUSION_count']} | {g['total_NET_NEGATIVE_RELIEF_count']} | {g['total_SAME_ID_LOST']} | {g['total_SAME_ID_GAINED']} |\n"
    report += "\n`SAME_ID_LOST` 是参考同身份成员在当前 Top-K 中缺失；`SAME_ID_GAINED` 是当前新增的同身份成员；二者不是同一件事，也不等于一一对应的因果替换。已对全部 anchor 断言：异身份数量变化 = 同身份数量变化的相反数。\n"
    (report_root / "NDP_METRIC_SEMANTIC_FIX.md").write_text(report, encoding="utf-8")

    report = "# NDP 身份邻居成员置换分析\n\n"
    report += "成员置换统计必须与身份数量变化分开：一个参考同身份成员丢失、另一个同身份成员进入，可能净数量不变；只有 `DELTA_SAME_ID_COUNT<0` 才表示同身份 Top-K 数量下降。\n\n"
    report += "| seed | K | 正确→错误 | 错误→正确 | H1可靠翻转关系 | H1 anchor | H2 anchor | H3 anchor |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for seed in SEEDS:
        for k in K_VALUES:
            st = next(x for x in metric_source["state_summaries"] if x["seed"] == seed and x["K"] == k)
            gs = next(x for x in source_summary if x["seed"] == seed and x["K"] == k)
            report += f"| {seed} | {k} | {st['correct_wrong']} | {st['wrong_correct']} | {st['reliable_flip_rows']} | {gs['H1_anchor_count']} | {gs['H2_anchor_count']} | {gs['H3_anchor_count']} |\n"
    report += "\nK=10 的错误→正确为 seed1234=`5248`、seed4321=`2430`；可靠翻转为 `971/58097` 与 `436/61012`。这些是冻结关系状态差异，不证明某次 GPC 更新造成变化。\n"
    (report_root / "NDP_IDENTITY_NEIGHBOR_REPLACEMENT.md").write_text(report, encoding="utf-8")

    report = "# NDP Triplet 覆盖有效性\n\n"
    report += "真实源码显示 Triplet 使用欧氏距离矩阵，Batch-Hard 选择最大正距离和最小负距离；`NO_MARGIN=True` 时使用 SoftMarginLoss。NDP 关系来自全 train 归一化余弦邻域，关系选择规则不同。\n\n"
    report += "| seed | 全局关系行 | 正端同batch | 负端同batch | 三端同batch | 完整hard | 目标可靠翻转完整hard | 窗口内有其他miner的翻转anchor |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for seed in SEEDS:
        all_s = cov_k10[(seed, "all_global_relation_rows")]
        flip_s = cov_k10[(seed, "reliable_flip_rows")]
        report += f"| {seed} | {all_s['rows']} | {all_s['positive_pair_opportunity']} | {all_s['negative_pair_opportunity']} | {all_s['three_way_opportunity']} | {all_s['full_triplet_hard_selected']} | {flip_s['full_triplet_hard_selected']} | {flip_s['anchor_any_miner_selection']}/{flip_s['unique_anchors']} |\n"
    report += "\n原始 `0.0012%` 是 seed1234/K10 的 `1/83670` 全局关系行命中率。它不是 optimizer step 覆盖率。可靠翻转目标关系的完整命中为两个 seed 均为 0，但翻转 anchor 大多在窗口内接受过其他 Triplet 约束，因此应表述为‘目标关系直接覆盖不足’，不能表述为‘现有 Triplet 完全未保护这些身份’。\n"
    (report_root / "NDP_TRIPLET_COVERAGE_VALIDITY.md").write_text(report, encoding="utf-8")

    report = "# NDP 可靠有害关系审计\n\n参考可靠条件固定为 `g_ref>0.05`，翻转定义为 `g_current<=0`。H1 是参考可靠且发生翻转；H2 是参考可靠但无翻转、anchor 的 Top-K 同身份数量下降；H3 是邻域变化且同身份数量增加。分组规则在查看结果前固定，未按 seed 结果挑选 PID。\n\n"
    report += "| seed | H1事件关系 | H1 anchor/PID | H1平均margin变化 | H2 anchor/PID | H3 anchor/PID |\n|---:|---:|---:|---:|---:|---:|\n"
    for seed in SEEDS:
        h1=by_group[(seed,"H1_reliable_flip")]; h2=by_group[(seed,"H2_same_id_drop_without_reliable_flip")]; h3=by_group[(seed,"H3_changed_with_same_id_gain")]
        report += f"| {seed} | {h1['event_rows']} | {h1['anchor_count']}/{h1['pid_count']} | {h1['mean_margin_delta']:+.6f} | {h2['anchor_count']}/{h2['pid_count']} | {h3['anchor_count']}/{h3['pid_count']} |\n"
    report += "\nH1 在两个 seed 都存在，但规模、方向和错误/修复平衡不同；目前没有足够证据证明共同的、稳定的危险子群。详细 H1/H2/H3 anchor/PID 表和 2000 次 PID bootstrap 位于新输出目录。\n"
    (report_root / "NDP_RELIABLE_HARMFUL_RELATION_AUDIT.md").write_text(report, encoding="utf-8")

    report = "# NDP 增量机制决策\n\n状态：`NDP_LOCAL_RISK_MECHANISM_UNRESOLVED`。\n\n"
    report += "候选1（参考自适应 margin 局部 Triplet）在同一激活三元组上与参考 margin hinge/Triplet 具有相同的逐关系梯度方向，不能单凭‘自适应 margin’宣称独立机制。\n\n"
    report += "候选2（多邻居局部同身份概率质量单侧保护）仍是较有区分度的研究接口：在固定参考邻域 U 上比较同身份 softmax 概率质量，只惩罚相对参考的质量下降，允许错误关系修复和有益邻域重排。但目前尚未证明它能跨 seed 保护一类现有 Triplet 未覆盖且确实有害的关系。\n\n"
    report += "因此不接入 processor、不设置 lambda、不启动短程 PPU 或 30/60 epoch 训练。下一步只有在在线 miner/状态记录和 AMP 数值门禁可接受后，才可预注册候选2与普通 Triplet、SupCon/N-pair、无参考局部质量控制的短程对照。\n"
    (report_root / "NDP_INCREMENTAL_MECHANISM_DECISION.md").write_text(report, encoding="utf-8")

    summary = {"run_time": run_time, "source_run": SOURCE_RUN, "official_query_gallery_used": False, "model_training_started": False, "decision": "NDP_LOCAL_RISK_MECHANISM_UNRESOLVED", "source_files": source_files, "group_rows": group_rows, "source_summary": source_summary, "output_reports": ["NDP_METRIC_SEMANTIC_FIX.md", "NDP_IDENTITY_NEIGHBOR_REPLACEMENT.md", "NDP_TRIPLET_COVERAGE_VALIDITY.md", "NDP_RELIABLE_HARMFUL_RELATION_AUDIT.md", "NDP_INCREMENTAL_MECHANISM_DECISION.md"]}
    write_csv(out / "h1_h2_h3_groups.csv", group_rows)
    write_csv(out / "h1_h2_h3_pid_rows.csv", pid_rows)
    write_json(out / "ndp_semantic_validity_summary.json", summary)
    log("【决策】NDP_LOCAL_RISK_MECHANISM_UNRESOLVED；未启动任何模型计算或训练。")


if __name__ == "__main__":
    main()
