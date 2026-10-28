#!/usr/local/bin/python3.10
"""生成本轮 GPC-v2 扩展验证、梯度可行性和下一阶段决策报告。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/DIFFER")
RUN_TIME = "20261016_141306"
PROTOCOL = ROOT / f"outputs/gpc_v2_protocol_audit_{RUN_TIME}/expanded_dev_protocol.json"
EVAL = ROOT / f"outputs/gpc_v2_expanded_validation_{RUN_TIME}_recovery1/expanded_dev_evaluation.json"
PAIRED = ROOT / f"outputs/gpc_v2_paired_pid_bootstrap_{RUN_TIME}/paired_pid_bootstrap.json"
GRADIENT = ROOT / f"outputs/gpc_gradient_compatibility_{RUN_TIME}_recovery4/gradient_compatibility.json"
SOURCE = ROOT / f"outputs/gpc_v2_source_audit_{RUN_TIME}/source_implementation_audit.json"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / f"outputs/gpc_v2_validation_closed_loop_{RUN_TIME}"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.4f}%"


def pp(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * float(value):+.4f}pp"


def find_pair(pairs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in pairs if row["pair"] == name)


def main() -> None:
    protocol = load(PROTOCOL)
    evaluation = load(EVAL)
    paired = load(PAIRED)
    gradient = load(GRADIENT)
    source = load(SOURCE)
    rows = paired["model_metrics"]
    pairs = paired["paired_comparisons"]
    gradient_rows = gradient["summary"]
    b0_gradient = [row for row in gradient_rows if row["arm"] == "B0" and row["cos_base_gpc"] is not None]
    mean_cos = sum(float(row["cos_base_gpc"]) for row in b0_gradient) / len(b0_gradient)
    min_cos = min(float(row["cos_base_gpc"]) for row in b0_gradient)
    max_ratio = max(float(row["gpc_over_base_grad_ratio"]) for row in b0_gradient)
    all_summary_cos_positive = all(float(row["cos_base_gpc"]) > 0.0 for row in b0_gradient)
    gradient_conflict = bool(min_cos < -0.1 and sum(float(row["cos_base_gpc"]) < 0 for row in b0_gradient) >= 2)

    a5_1234 = find_pair(pairs, "E60_B0_vs_A5_seed1234")
    a5_4321 = find_pair(pairs, "E60_B0_vs_A5_seed4321")
    a4_e30 = find_pair(pairs, "E30_B0_vs_A4_seed1234")
    a2_e30 = find_pair(pairs, "E30_B0_vs_A2_seed1234")
    a5_e30 = find_pair(pairs, "E30_B0_vs_A5_seed1234")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_time": RUN_TIME,
        "protocol": str(PROTOCOL), "protocol_sha256": sha256(PROTOCOL),
        "evaluation": str(EVAL), "evaluation_sha256": sha256(EVAL),
        "paired_bootstrap": str(PAIRED), "paired_bootstrap_sha256": sha256(PAIRED),
        "gradient": str(GRADIENT), "gradient_sha256": sha256(GRADIENT),
        "source_audit": str(SOURCE), "source_audit_sha256": sha256(SOURCE),
        "official_query_gallery_used": False, "cpu_model_fallback": False,
        "new_training_started": False, "historical_results_modified": False,
        "final_status": "GPC-v2 MIXED",
    }
    (OUTPUT_DIR / "validation_closed_loop_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# GPC-v2 LTCC 扩展开发验证闭环报告",
        "",
        "本轮是固定开发协议下的探索性复核，不是官方 LTCC 独立确认测试。没有重新训练，没有读取官方 query/gallery，也没有修改历史 GPC-v1 或 `20261015_155822` 结果。",
        "",
        f"- 本轮运行：`{RUN_TIME}`。",
        f"- 协议状态：`{protocol['status']}`；协议 SHA256：`{sha256(PROTOCOL)}`。",
        f"- Query：`{protocol['expanded_protocol']['query_count']}`；Gallery：`{protocol['expanded_protocol']['gallery_count']}`；有效 PID：`{protocol['expanded_protocol']['effective_pid_count']}`。",
        f"- Query 扩展前候选数：`{protocol['expanded_protocol']['query_count_before_legal_filter']}`；无合法跨衣正样本而剔除：`{protocol['expanded_protocol']['dropped_no_legal_positive_count']}`。",
        "- 运行设备：PPU-ZW810E；CPU 仅用于离线距离、指标和 Bootstrap；`cpu_model_fallback=False`。",
        "",
        "## 协议审计",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for key, value in protocol["checks"].items(): lines.append(f"| {key} | `{value}` |")
    lines += [
        "",
        "### Query 数量变化解释",
        "",
        "旧 evaluator 固定每个 PID 取最小衣物状态的第一张图。冻结验证池有 17 个 PID，其中 6 个 PID 在原摄像头/衣物排除规则下没有合法跨衣正样本，因此历史运行实际留下 11 个 Query；这不是随机丢图，而是旧规则与 LTCC CC 合法性共同造成的。扩展协议保留每个验证 PID 的最小衣物状态全部图像，再逐 Query 过滤无合法正样本，最终得到 693 个 Query。",
        "",
        f"- Query manifest SHA256：`{protocol['expanded_protocol']['query_manifest']['sha256']}`。",
        f"- Gallery manifest SHA256：`{protocol['expanded_protocol']['gallery_manifest']['sha256']}`。",
        "- 旧 11-query JSON 保留在 `/home/DIFFER/outputs/gpc_v2_ltcc_20261015_155822/`，未与新结果混算。",
        "",
        "## 已有 checkpoint 的固定 E30/E60 复核",
        "",
        "| 模型 | seed | Query | Gallery | PID | 图像R1 | 图像R5 | 图像R10 | 图像mAP | PID宏R1 | PID宏mAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['label']} | {row['seed']} | {row['query_count']} | {row['gallery_count']} | {row['effective_pid_count']} | {pct(row['rank1'])} | {pct(row['rank5'])} | {pct(row['rank10'])} | {pct(row['mAP'])} | {pct(row['pid_macro_rank1'])} | {pct(row['pid_macro_mAP'])} |")
    lines += [
        "",
        "## 配对 PID Bootstrap",
        "",
        "Bootstrap 以 PID 为重采样单位，2000 次；每个比较中 baseline/candidate 使用相同抽样索引。当前只有 11 个有效 PID，因此 CI 只能作为探索性证据。逐 PID 与逐 Query 明细见 [GPC_V2_PAIRED_PID_BOOTSTRAP.md](GPC_V2_PAIRED_PID_BOOTSTRAP.md)。",
        "",
        "| 比较 | ΔR1均值 | ΔR1 CI95 | ΔmAP均值 | ΔmAP CI95 | R1 胜/平/负 |",
        "|---|---:|---|---:|---|---|",
    ]
    for item in pairs:
        b = item["pid_bootstrap"]
        q = item["query_pair_counts"]
        lines.append(f"| {item['pair']} | {pp(b['delta_rank1_mean'])} | [{pp(b['delta_rank1_ci95'][0])}, {pp(b['delta_rank1_ci95'][1])}] | {pp(b['delta_mAP_mean'])} | [{pp(b['delta_mAP_ci95'][0])}, {pp(b['delta_mAP_ci95'][1])}] | {q['rank1_wins']}/{q['rank1_ties']}/{q['rank1_losses']} |")
    lines += [
        "",
        "## 候选判断",
        "",
        f"- A1：E30 相对 B0 的 R1/mAP 均为负，不支持。",
        f"- A2：E30 点估计 mAP 改善，但 R1 CI 跨 0，且没有双 seed E60 复核，不支持晋级。",
        f"- A3：E30 R1/mAP 均为负，不支持。",
        f"- A4：E30 点估计最强（ΔR1={pp(a4_e30['pid_bootstrap']['delta_rank1_mean'])}，ΔmAP={pp(a4_e30['pid_bootstrap']['delta_mAP_mean'])}），PID Bootstrap CI 未跨 0；但它只有 seed1234/E30，且源码审计显示所谓 PID balance 仅在单 batch 内 round-robin，历史全周期贡献长尾仍达约 795:1，因此不能称为稳定双 seed 改善。",
        f"- A5：E60 seed1234 点估计 ΔR1={pp(a5_1234['pid_bootstrap']['delta_rank1_mean'])}、ΔmAP={pp(a5_1234['pid_bootstrap']['delta_mAP_mean'])}；seed4321 点估计 ΔR1={pp(a5_4321['pid_bootstrap']['delta_rank1_mean'])}、ΔmAP={pp(a5_4321['pid_bootstrap']['delta_mAP_mean'])}，两 seed 点估计同向，但两项 R1 CI 都跨 0，mAP CI 也跨 0。因此历史“未通过双 seed 晋级”的负结果保留；扩展协议下仍不足以支持正式确认。",
        "",
        "## GPC 实现与训练曝光审计",
        "",
        "- B0/A1/A2/A3/A5 使用 Phase7I I3 ID-only stable selector；matched pair-budget 是每 batch 的 `min(cross, same, id)`，没有额外 pair。A4 使用同一预算，但仅 batch 内 PID round-robin。",
        "- 真实训练引擎通过一次 combined `total` loss backward；本轮没有把历史 GPC-v1 改成 dense all-pair。",
        "- 历史 E30 active batch rate 约 11.22%，E60 约 11.43%–11.51%；历史 logs 没有 CE/Triplet/Bio/GPC 梯度范数，不能从 raw loss 猜梯度。",
        "- 历史训练存在 AMP skipped/nonfinite 事件；本轮 train-only 梯度 probe 的有限性检查全部通过。",
        "- A5 tau=0.9021800160：早期 probe 的 ReLU 激活比例为 0、GPC 梯度为0；中后期才恢复部分有效梯度。这支持“早期约束饱和/失活”现象，但不是梯度冲突证据。",
        "",
        "## 结论",
        "",
        "当前最终状态：**GPC-v2 MIXED**。没有候选满足“两个 seed + 统计证据 + mAP 保持”的可靠晋级条件；不启动新的 60 epoch 或正式 LTCC 官方测试。",
        "",
        "下一步最值得验证的不是新增损失，而是 A4 所代表的 PID 覆盖机制：在保持原始 matched pair-budget 的前提下，预注册一个真正跨 epoch 累积均衡、仍不按困难度选 pair 的 selector，再做双 seed 复核。此举需要新的训练协议，不能重新解释现有 checkpoint。",
        "",
        f"机器可读 manifest：`{OUTPUT_DIR / 'validation_closed_loop_manifest.json'}`。",
    ]
    (REPORT_DIR / "GPC_V2_EXPANDED_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    grad_lines = [
        "# GPC-v3 可行性决策",
        "",
        "## 决策",
        "",
        "**GPC-v3：NOT JUSTIFIED；不启动新训练。**",
        "",
        "## 梯度证据",
        "",
        f"- 诊断空间：共享 `feat[0]` 输出层，不是完整参数梯度；GPC 梯度包含归一化身份特征 z 的链式梯度。",
        f"- B0 跨 seed/阶段汇总 probe 的 `cos(g_base, g_gpc)` 均值：`{mean_cos:.6f}`；最小阶段均值：`{min_cos:.6f}`；最大 `lambda*g_gpc / g_base` 范围上界：`{max_ratio:.6f}`。",
        f"- 所有 B0 汇总方向均为正：`{all_summary_cos_positive}`。个别 probe 出现轻微负值，但没有跨 seed、跨阶段持续明显冲突。",
        "- 因此不标记 `GRADIENT_CONFLICT_OBSERVED`；更严格的结论是 `NO_CLEAR_GRADIENT_CONFLICT`。不能声称梯度冲突导致检索下降。",
        "",
        "| arm | seed | 阶段 | cos(base,GPC) | cos(triplet,GPC) | λGPC/base | 正样本cos | 最近异身份cos | A5激活比例 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gradient_rows:
        def val(key: str) -> str:
            return "NA" if row.get(key) is None else f"{float(row[key]):.6f}"
        grad_lines.append(f"| {row['arm']} | {row['seed']} | {row['phase']} | {val('cos_base_gpc')} | {val('cos_triplet_gpc')} | {val('gpc_over_base_grad_ratio')} | {val('positive_cosine_mean')} | {val('nearest_negative_cosine_mean')} | {val('a5_relu_active_fraction')} |")
    grad_lines += [
        "",
        "## GPC-v3 候选是否提出",
        "",
        "本轮不提出 Gradient-Compatibility-Aware GPC 训练候选。原因是没有观察到足够稳定的基础身份梯度与 GPC 梯度冲突；A5 早期失活更像阈值饱和/训练时机问题，且后期 probe 方向总体为正。若未来要研究动态强度，必须先重新预注册公式、动态预算匹配随机控制和小规模 train-only 预检，不能把它作为本轮结论。",
        "",
        "## 当前后续选择",
        "",
        "- 保留 GPC-v1 作为后续 NDP 基础；不以 A5 替代。",
        "- 暂不继续训练；不执行官方 LTCC query/gallery 确认。",
        "- 若继续研究，优先审计/重做 A4 的跨 epoch PID 覆盖机制，保持 pair-budget、loss reduction、Bio、Triplet、CE 和基础训练配置不变。",
        "- PRCC、NDP 和历史 GPC-v1 均未修改。",
    ]
    (REPORT_DIR / "GPC_V3_FEASIBILITY_DECISION.md").write_text("\n".join(grad_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "expanded_report": str(REPORT_DIR / 'GPC_V2_EXPANDED_VALIDATION_REPORT.md'), "v3_report": str(REPORT_DIR / 'GPC_V3_FEASIBILITY_DECISION.md'), "final_status": manifest['final_status'], "new_training_started": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
