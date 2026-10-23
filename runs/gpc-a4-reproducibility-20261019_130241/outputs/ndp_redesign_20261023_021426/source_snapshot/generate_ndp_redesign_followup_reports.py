#!/usr/bin/env python3.10
"""生成 NDP P5-P9 中文报告；只读取既有离线审计产物。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE = Path("/home/DIFFER")
REMOTE = "d6e24867a9bf33423d3342682c96d053d37ff190"
CHECKPOINTS = {"seed1234": "e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177", "seed4321": "d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7"}
BASELINES = {"gpc_v2_train.py": "d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5", "make_loss.py": "9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043", "eva02_l_bio.yml": "c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97", "triplet_loss.py": "86656d44255172866d1e9854880c16e5d9536c6a756a577876312b663512092a"}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def metadata(scripts: dict[str, str], commands: list[str], data: list[str]) -> str:
    return "\n".join([
        "\n## 可复现元数据\n",
        f"- 远端 main：`{REMOTE}`；本地工作树无 `.git`，不伪造本地 commit。",
        "- 固定 baseline SHA256：" + "; ".join(f"`{k}={v}`" for k, v in BASELINES.items()) + "。",
        "- checkpoint SHA256：" + "; ".join(f"`{k}={v}`" for k, v in CHECKPOINTS.items()) + "。",
        "- 新研究代码 SHA256：" + "; ".join(f"`{k}={v}`" for k, v in scripts.items()) + "。",
        "- 数据来源：" + "; ".join(f"`{x}`" for x in data) + "。",
        "- 全部为 train-only、冻结特征或无更新只读审计；无 query/gallery；无新的 optimizer.step。",
        "\n## 运行命令\n",
        *[f"- `{x}`" for x in commands],
        "\n未完成：真实网络因果效果、官方评估变化、完整历史 Batch-Hard 逐步日志和 A/A 根因仍未证明。",
        "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--audit-output", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--stage3-output", type=Path, required=True)
    args = ap.parse_args(); args.report_dir.mkdir(parents=True, exist_ok=True)
    audit = BASE / "scripts/ndp_reliability_safe_boundary_audit.py"
    stage = BASE / "scripts/ndp_overnight_recovery_stage3.py"
    activation = BASE / "scripts/ndp_real_trajectory_activation.py"
    scripts = {str(p): sha(p) for p in (audit, stage, activation)}
    cmd_audit = f"/usr/local/bin/python3.10 -u {audit} --run-id {args.run_id} --output-root {args.audit_output} --report-dir {args.report_dir} --remote-main {REMOTE}"
    cmd_stage = f"source /usr/local/PPU_SDK/envsetup.sh ppu && /usr/local/bin/python3.10 -u {stage} {args.run_id}"
    data = [str(args.audit_output), "/home/DIFFER/outputs/ndp_triplet_alignment_20261021_134148", str(args.stage3_output)]
    summaries = {s: load(args.audit_output / s / f"redesign_summary_{s}.json") for s in ("seed1234", "seed4321")}
    impacts = {s: load(args.audit_output / s / f"neighborhood_collateral_summary_{s}.json") for s in ("seed1234", "seed4321")}
    miner = {s: load(BASE / "outputs/ndp_triplet_alignment_20261021_134148/miner_coverage" / s / "triplet_miner_coverage_summary.json") for s in ("seed1234", "seed4321")}
    comp = {s: load(BASE / "outputs/ndp_margin_consistency_20261021_220823/complementarity" / s / f"triplet_complementarity_summary_{s}.json") for s in ("seed1234", "seed4321")}

    p5 = ["# NDP 与真实 Triplet 的结构互补性\n", "原始 DIFFER Triplet 固定为未归一化 raw `feat[0]`、欧氏距离、hard positive=max 同 PID、hard negative=min 异 PID、`SoftMarginLoss(dist_an-dist_ap)`。以下复用真实 raw 路径对齐和重建 Batch-Hard 覆盖；NDP 只在固定 difficult-K5 图上做跨 Batch 关系诊断。\n", "| seed | K5关系 | PID级有机会 | PID级hard负选择 | 固定图像级有机会 | 完整关系选择 | active triplet |\n|---|---:|---:|---:|---:|---:|---:|"]
    for s in ("seed1234", "seed4321"):
        m = miner[s]; p5.append(f"| {s} | {m['ndp_relation_count']} | {m['ndp_relations_with_negative_pid_opportunity']} | {m['ndp_relations_with_hard_negative_pid']} | {m['ndp_relations_with_batch_opportunity']} | {m['ndp_relations_with_full_miner_selection']} | {m['ndp_relations_with_active_triplet']} |")
    p5 += ["\nPID级机会约8%，固定图像级完整关系选择为0；因此 NDP 有结构访问增量，但不能说真实训练中这些身份完全没有间接监督（anchor_any_triplet=398）。冻结关系梯度余弦既有结果为 seed1234 0.0486、seed4321 0.0230；这不是有效性证明。", "结论：`STRUCTURAL_COMPLEMENTARITY = DESCRIPTIVE_ONLY`，没有真实参数增量或有益冲突的因果证据。", metadata(scripts, [cmd_audit, "/usr/local/bin/python3.10 -u scripts/ndp_real_triplet_miner_coverage.py ... --epochs 60 61", "/usr/local/bin/python3.10 -u scripts/ndp_triplet_complementarity_audit.py ... --no-optimizer-step"], data)]
    (args.report_dir / "NDP_TRIPLET_STRUCTURAL_COMPLEMENTARITY.md").write_text("\n".join(p5), encoding="utf-8")

    aa = load(BASE / "outputs/ndp_margin_consistency_20261021_220823/aa_probe/aa_repeat_comparison_float64.json")
    p6 = ["# EVA02 完整 A/A 反向首个可观测差异\n", "本轮不重复孤立 LayerNorm probe，复用固定输入、上游梯度、E60 checkpoint、AMP、dtype/stride 和独立进程的完整 A/A。\n", "| seed | 最大已观测参数张量 | relative L2 | max abs | cosine(float64) |\n|---|---|---:|---:|---:|"]
    for s in ("1234", "4321"):
        best_name, best = max(aa[s]["tensor_comparisons_float64"].items(), key=lambda kv: float(kv[1]["relative_l2_to_repeat1"]))
        p6.append(f"| {s} | {best_name} | {best['relative_l2_to_repeat1']:.8g} | {best['max_abs']} | {best['cosine_float64']:.12f} |")
    p6 += ["\n最大差异仍位于早期 `blocks.0.attn.proj.weight`（约0.00412/0.00413）；block22 q_proj 约8.6e-6，输出投影为零差异。独立 LayerNorm/norm1 probe 稳定，不能把 blocks.22.norm1 hash 变化认定为根因。", "当前只定位到完整组合 backward/传播路径，尚未对 block22 Attention、Q/K/V、softmax、matmul、输出投影、残差、跨 Block、AMP kernel、DDP all-reduce 给出逐级首差异。因此 `A/A_REPRODUCIBILITY=FAIL`，不改变 A/A 决策。", metadata(scripts, ["/usr/local/bin/python3.10 -u scripts/ndp_aa_operator_probe.py --mode amp --no-optimizer-step ...", "/usr/local/bin/python3.10 -u scripts/ndp_aa_operator_analysis.py --root ... --seed {1234,4321} ..."], ["outputs/ndp_margin_consistency_20261021_220823/aa_probe/aa_repeat_comparison_float64.json"])]
    (args.report_dir / "NDP_EVA02_BACKWARD_FIRST_DIVERGENCE.md").write_text("\n".join(p6), encoding="utf-8")

    p7 = ["# 夜间阶段3只读恢复\n", "原失败确为 `NameError: name 'protocol' is not defined`。本轮只修复入口变量：`runtime_protocol = install_protocol(...)` 后显式传给 `choose_groups()`；并在 choose_groups 入口读取一次原始 loader，避免重复包装。没有重跑400步 continuation，阶段4/5未启动。\n", "| seed | groups | 节点激活（update:count） | 自然激活 |\n|---|---:|---|---|"]
    for s in ("1234", "4321"):
        d = load(args.stage3_output / f"trajectory_seed{s}/activation_audit/activation_summary.json")
        node_text = ", ".join(str(x["update"]) + ":" + str(x["active_anchor_count"]) for x in d["nodes"])
        p7.append(f"| {s} | {d['groups']} | {node_text} | {d['real_trajectory_activation']} |")
    st = load(args.stage3_output / "FINAL_DECISION.json")
    p7 += [f"\n阶段3=`{st['stage3']}`；阶段4=`{st['stage4']}`；阶段5=`{st['stage5']}`。自然激活只是保存节点上的只读关系诊断，不是训练效果。", "单测检查协议初始化顺序、choose_groups 不调用 install_protocol、只读激活脚本无 optimizer.step。", metadata(scripts, [cmd_stage, "PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER /usr/local/bin/python3.10 -m pytest -q tests/test_ndp_stage3_readonly_recovery.py"], [str(args.stage3_output), "outputs/ndp_overnight_20261020_201841"])]
    (args.report_dir / "NDP_STAGE3_READONLY_RECOVERY.md").write_text("\n".join(p7), encoding="utf-8")

    p8 = ["# NDP 最终下一阶段决策\n", "`NDP_MECHANISM = INCONCLUSIVE`；`A/A_REPRODUCIBILITY = FAIL`；`CONTROLLED_SHORT_PILOT = NO_GO`。本轮没有修改 DIFFER+GPC、EVA02、CE、原始 Triplet、clipBio、GPC 权重/正样本、优化器或 sampler。\n", "| 问题 | 证据结论 | 门禁 |\n|---|---|---|", "| 1 | 有真实 R∩V：637/618；翻转严格复现28/18。 | 描述性 PASS，不是预测 PASS |", "| 2 | C激活16/8，A=38/9，B=17/0，D=12/6；有集合差异，但仍是历史动态阈值变换。 | 新机制未证明 |", "| 3 | 解耦后新纳入12/6；未覆盖未来翻转，不能回顾性声称预测。 | NO causal claim |", "| 4 | safe target 的其他关系负影响15/9；双侧 safe 32/26。 | collateral 风险存在 |", "| 5 | PID级 Triplet 机会约8%，固定完整关系0；结构访问有增量，真实参数增量未证。 | INCONCLUSIVE |", "| 6 | Hard 回顾性覆盖28/18但混入大量当前错误；R类0翻转；预算样本小且无因果。 | 不足以 GO |", "| 7 | 完整 A/A 最大差异约0.00412，根因未唯一定位。 | FAIL |", "| 8 | 可保留冻结诊断接口；训练损失当前不值得接入。 | NO_GO |", "\n建议停止新增复杂门控、双侧/多 PID 组合和正式训练分支。若继续，只先记录真实 raw Batch-Hard 逐步覆盖并完成 A/A 首差异定位，再申请预注册预算匹配实验。", metadata(scripts, [cmd_audit, cmd_stage, "PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER /usr/local/bin/python3.10 -m pytest -q tests scripts/tests"], data)]
    (args.report_dir / "NDP_FINAL_NEXT_STAGE_DECISION.md").write_text("\n".join(p8), encoding="utf-8")
    manifest = {"run_id": args.run_id, "remote_main": REMOTE, "reports": [str(x) for x in sorted(args.report_dir.glob("NDP_*.md"))], "script_sha256": scripts, "checkpoint_sha256": CHECKPOINTS, "baseline_sha256": BASELINES, "optimizer_step_executed": False, "stage4_stage5_run": False}
    (args.report_dir / "FOLLOWUP_REPORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
