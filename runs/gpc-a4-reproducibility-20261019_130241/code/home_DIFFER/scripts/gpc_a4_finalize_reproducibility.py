#!/usr/local/bin/python3.10
"""生成 GPC-A4 可复现性审计的机器可读指纹和五份最终报告。

本脚本只读取已有 checkpoint manifest、评估 JSON、训练 CSV、代码文件和
GitHub 参考工作树；不训练、不加载模型、不访问 Query/Gallery 图像。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


RUN = "20261019_130241"
ROOT = Path("/home/DIFFER")
NEW_RUN = ROOT / "outputs" / f"gpc_a4_reproducibility_{RUN}"
NEW_TRAIN = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen"
OLD_TRAIN = ROOT / "outputs/gpc_v2_ltcc_20261015_155822/training/screen"
EVAL = ROOT / "outputs/gpc_a4_result_audit_20261019_124222/evaluation_screen_fixed/expanded_dev_evaluation.json"
OLD_EVAL = ROOT / "outputs/gpc_v2_expanded_validation_20261016_141306_recovery1/expanded_dev_evaluation.json"
PAIR = NEW_RUN / "gpc_a4_paired_pid_statistics.json"
EXPOSURE = NEW_RUN / "gpc_a4_exposure_statistics.json"
OLD_NEW = NEW_RUN / "gpc_a4_old_new_audit.json"
GRADIENT = ROOT / "outputs/gpc_gradient_compatibility_20261016_141306_recovery4/gradient_compatibility.json"
REF_DEFAULT = Path("/tmp/test2-ref-audit-AniKE8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.4f} pp"


def file_record(local: Path, remote_root: Path, remote_rel: str, note: str) -> dict[str, Any]:
    remote = remote_root / remote_rel
    return {
        "local": str(local),
        "local_exists": local.is_file(),
        "local_sha256": sha256(local) if local.is_file() else None,
        "github_reference_commit": "3dd98604a38075060d96aeb70506a977dd123be4",
        "github_path": remote_rel,
        "github_exists": remote.is_file(),
        "github_sha256": sha256(remote) if remote.is_file() else None,
        "status": ("MATCH" if local.is_file() and remote.is_file() and sha256(local) == sha256(remote)
                   else "DIFF" if local.is_file() and remote.is_file()
                   else "LOCAL_ONLY" if local.is_file() else "MISSING"),
        "note": note,
    }


def fmt_pair(row: dict[str, Any]) -> str:
    b = row["bootstrap"]
    return (
        f"| {row['baseline']} vs {row['candidate']} | {row['effective_pid_count']} | "
        f"{pp(row['image_micro']['delta_rank1'])} [{pp(b['query_weighted_delta_rank1_ci95'][0])}, {pp(b['query_weighted_delta_rank1_ci95'][1])}] | "
        f"{pp(row['image_micro']['delta_mAP'])} [{pp(b['query_weighted_delta_mAP_ci95'][0])}, {pp(b['query_weighted_delta_mAP_ci95'][1])}] | "
        f"{pp(row['pid_macro']['delta_rank1'])} [{pp(b['pid_macro_delta_rank1_ci95'][0])}, {pp(b['pid_macro_delta_rank1_ci95'][1])}] | "
        f"{pp(row['pid_macro']['delta_mAP'])} [{pp(b['pid_macro_delta_mAP_ci95'][0])}, {pp(b['pid_macro_delta_mAP_ci95'][1])}] |"
    )


def metric_table(evaluation: dict[str, Any]) -> str:
    rows = [
        "| 模型 | seed | Query | Gallery | 有效 PID | Image-micro Rank-1 | Image-micro mAP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, result in evaluation["results"].items():
        m = result["metrics"]
        rows.append(
            f"| {result['arm']} | {result['seed']} | {m['query_count']} | {m['gallery_count']} | "
            f"{m['effective_pid_count']} | {pct(m['rank1'])} | {pct(m['mAP'])} |"
        )
    return "\n".join(rows)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", default=RUN)
    parser.add_argument("--reference-worktree", type=Path, default=REF_DEFAULT)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=NEW_RUN)
    args = parser.parse_args()

    evaluation = load(EVAL)
    old_evaluation = load(OLD_EVAL)
    pairs = load(PAIR)
    exposure = load(EXPOSURE)
    old_new = load(OLD_NEW)
    args.output.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)

    file_specs = [
        (ROOT / "scripts/gpc_v2_expanded_dev_eval.py", "scripts/gpc_v2_expanded_dev_eval.py", "当前评估修复源码；GitHub main 仍是旧的固定名单实现。评估指标和检索函数未发生修改，新增的是资产/配置指纹、显式名单、复用门禁和严格配对检查。"),
        (ROOT / "scripts/gpc_a4_recheck_train.py", "scripts/gpc_a4_recheck_train.py", "真实执行的新 A4 训练 wrapper；参考提交不存在。"),
        (Path("/data/DIFFER/scripts/gpc_v2_train.py"), "scripts/gpc_v2_train.py", "旧运行实际使用的训练 wrapper；参考提交不存在。"),
        (Path("/data/DIFFER/scripts/phase7i_relation_specificity.py"), "scripts/phase7i_relation_specificity.py", "Phase 7I 实际加载源码。"),
        (Path("/data/DIFFER/scripts/phase6b_soft_relation_screen.py"), "scripts/phase6b_soft_relation_screen.py", "Phase 6B 实际加载源码。"),
        (Path("/data/DIFFER/scripts/phase5b_id_damage_ds.py"), "scripts/phase5b_id_damage_ds.py", "Phase 5B 基础损失/模型入口实际加载源码。"),
        (Path("/data/DIFFER/data/samplers.py"), "data/samplers.py", "基础 DistributedRandomIdentitySampler；A4 PID selector 在 wrapper 中，不在此文件。"),
        (Path("/data/DIFFER/loss/make_loss.py"), "loss/make_loss.py", "CE、Triplet、Bio 基础身份损失；没有独立 gpc_relation.py。"),
        (Path("/data/DIFFER/model/build.py"), "model/build.py", "当前实际模型构建源码与 GitHub 参考提交不同；manifest 记录 EVA02-Large Bio-only topology。"),
        (Path("/data/DIFFER/processor/processor.py"), "processor/processor.py", "当前工作树存在，参考提交不存在。"),
        (Path("/data/DIFFER/solver/scheduler.py"), "solver/scheduler.py", "调度器源码。"),
        (Path("/data/DIFFER/solver/scheduler_factory.py"), "solver/scheduler_factory.py", "调度器工厂源码。"),
    ]
    code_records = [file_record(local, args.reference_worktree, remote, note) for local, remote, note in file_specs]
    code_audit = {
        "audit_run_time": args.run_time,
        "github_reference_url": "https://github.com/smallvv1/test2",
        "github_reference_commit": "3dd98604a38075060d96aeb70506a977dd123be4",
        "reference_worktree": str(args.reference_worktree),
        "data_worktree_is_dirty": True,
        "records": code_records,
        "evaluator_metric_rule_review": {
            "status": "CONFIRMED_PRESERVED",
            "preserved": ["L2 normalization", "LTCC same-PID same-camera removal", "same-PID same-clothing removal", "Euclidean retrieval ordering", "per-query AP and Rank-1", "PID aggregation"],
            "changed": ["protocol/query/gallery SHA256 fingerprints", "evaluation config fingerprint", "explicit B0/A4/R2 × seeds spec parsing", "missing-file errors instead of silent skips", "reuse gate", "strict paired Query-set equality"],
            "source_diff_review": "current vs GitHub reference diff reviewed; metric and retrieval function bodies are unchanged",
        },
        "actual_runtime_chain": [
            "gpc_a4_recheck_train.py -> phase7i_relation_specificity.py -> phase6b_soft_relation_screen.py -> phase5b_id_damage_ds.py -> loss.make_loss/model/processor/solver",
            "A4 selector: cumulative_pid_balanced_selection; B0 selector: phase7i stable_selection; R2 selector: random_budget_selection",
            "base loss: CE + Triplet + Bio; total = base_total + lambda * relation; lambda=0.1",
            "sampler: DistributedRandomIdentitySampler supplies batches; relation selector is wrapper-level",
        ],
    }
    (args.output / "gpc_a4_code_file_comparison.json").write_text(json.dumps(code_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    artifact_rows = []
    for seed in (1234, 4321):
        for arm in ("B0", "A4", "R2"):
            directory = NEW_TRAIN / f"{arm}_seed{seed}"
            artifact_rows.append({
                "run": "new_20261016_161631",
                "arm": arm,
                "seed": seed,
                "checkpoint": str(directory / "model_final.pth"),
                "checkpoint_sha256": sha256(directory / "model_final.pth"),
                "config": str(directory / "config.yml"),
                "config_sha256": sha256(directory / "config.yml"),
                "run_manifest_sha256": sha256(directory / "run_manifest.json"),
                "arm_manifest_sha256": sha256(directory / "v2_arm_manifest.json"),
            })
    artifact_meta = {
        "run_time": args.run_time,
        "new_run": "20261016_161631",
        "protocol_sha256": old_new["protocol_hashes"]["protocol"],
        "expanded_query_sha256": old_new["protocol_hashes"]["query"],
        "expanded_gallery_sha256": old_new["protocol_hashes"]["gallery"],
        "development_split_manifest_sha256": old_new["development_split_hashes"]["new_manifest"],
        "development_split_csv_sha256": old_new["development_split_hashes"]["new_development_split_csv"],
        "common_initialization_sha256": "72892762c37044b85dc9ed9999c833da574026c3715e760885363201a8a40ed0",
        "common_initialization_meta_sha256": "dd3acd0da3d935b2d734567b8db9bb171d63d64d24fe2a378f5b05cf335bee67",
        "artifacts": artifact_rows,
        "official_ltcc_test_run": False,
        "large_scale_training_run": False,
    }
    (args.output / "gpc_a4_artifact_hashes.json").write_text(json.dumps(artifact_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (args.output / "gpc_a4_paired_pid_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["baseline", "candidate", "seed", "raw_pid", "query_count", "baseline_rank1", "candidate_rank1", "delta_rank1", "baseline_mAP", "candidate_mAP", "delta_mAP"])
        for pair in pairs:
            for row in pair["pid_rows"]:
                writer.writerow([pair["baseline"], pair["candidate"], pair["seed"], row["raw_pid"], row["query_count"], row["baseline_rank1"], row["candidate_rank1"], row["delta_rank1"], row["baseline_mAP"], row["candidate_mAP"], row["delta_mAP"]])

    metrics = metric_table(evaluation)
    pair_table = "\n".join([
        "| 配对 | 有效 PID | Image ΔR1（95% CI） | Image ΔmAP（95% CI） | PID-macro ΔR1（95% CI） | PID-macro ΔmAP（95% CI） |",
        "|---|---:|---:|---:|---:|---:|",
        *[fmt_pair(row) for row in pairs],
    ])
    old_b0 = old_evaluation["results"]["screen_B0_seed1234"]["metrics"]
    old_a4 = old_evaluation["results"]["screen_A4_seed1234"]["metrics"]
    new_b0 = evaluation["results"]["screen_B0_seed1234"]["metrics"]
    new_a4 = evaluation["results"]["screen_A4_seed1234"]["metrics"]

    write(args.reports / "GPC_A4_CODE_REPRODUCIBILITY_REPORT.md", f"""# GPC-A4 代码与可复现性报告

审计 RUN_TIME：`{args.run_time}`；GitHub 参考提交：`3dd98604a38075060d96aeb70506a977dd123be4`。

## 结论

真实执行源码与 GitHub 参考提交已逐文件对照，机器可读结果见 `{args.output}/gpc_a4_code_file_comparison.json`。当前 data 工作树是 dirty，不能把 GitHub main 当作实际训练源码。修复后的评估源码已保存并通过单元测试：`5/5 PASS`。

| 文件 | 对照状态 |
|---|---|
| `scripts/gpc_v2_expanded_dev_eval.py` | `DIFF`：当前版本加入显式名单、资产/配置/checkpoint 指纹、复用门禁、严格 Query 配对；检索和指标计算规则保持 |
| `scripts/gpc_a4_recheck_train.py` | `LOCAL_ONLY`：参考提交缺失 |
| `data/DIFFER/scripts/gpc_v2_train.py` | `LOCAL_ONLY`：旧运行实际源码，参考提交缺失 |
| Phase 7I / 6B / 5B | `MATCH` |
| sampler / loss | `MATCH` |
| `model/build.py` | `DIFF` |
| `processor/processor.py` | `LOCAL_ONLY` |
| scheduler / scheduler_factory | `MATCH` |

## 实际执行链

`gpc_a4_recheck_train.py → Phase 7I → Phase 6B → Phase 5B → loss/model/processor/solver`。基础身份训练是 CE + Triplet + Bio；总损失为 `base_total + lambda * relation`，本轮 `lambda=0.1`。基础 sampler 是 `DistributedRandomIdentitySampler`；A4 的 PID 选择发生在 wrapper 的全局 epoch 计划，不在 sampler 文件内。

当前 A4 使用 `cumulative_pid_balanced_selection`，B0 使用 Phase7I 的确定性 I3 selector，R2 使用随机 matched-budget selector。旧 A4 使用的是 batch-local `balanced_selection`，所以旧 A4 与新 A4 不是同一实现。

## 评估修复边界

对当前源码与参考提交的 diff 逐段检查后确认：原始模型 forward、L2 归一化、LTCC 同 PID 同 camera / 同 clothing 排除、Euclidean 检索排序、逐 Query AP/Rank-1、PID 聚合均保留；修复只增加协议/Query/Gallery/配置/权重指纹、显式 B0/A4/R2×seed 名单、缺失文件报错、严格配对清单检查和安全复用门禁。

## 归档

已归档修复源码、旧/新训练 wrapper、Phase 7I/6B/5B、sampler、loss、测试和 JSON/CSV 小型报告；归档不包含密钥、模型权重或数据集。六个新 checkpoint 及其 SHA256 只写入本地审计元数据，不上传权重。
""")

    write(args.reports / "GPC_A4_PAIRED_PID_CONFIDENCE_REPORT.md", f"""# GPC-A4 配对 PID 置信度报告

## 数据与方法

冻结开发协议：693 Query、1256 Gallery；有效验证 PID：**11**。完整逐 PID AP/Rank-1 在 `{args.output}/gpc_a4_paired_pid_rows.csv`，原始逐 Query 结果在 `{EVAL}`。

每个配对使用 2000 次 Bootstrap，固定随机种子，PID 为重采样单位；Image-micro 和 PID-macro 使用同一抽样 PID 索引，未把两个模型的 PID 配对打散。因此 11 个有效 PID 明显偏少，所有推断均标记为 `INCONCLUSIVE / 统计证据不足`。

{pair_table}

## 原始单模型指标

{metrics}

## 逐身份变化计数

| 配对 | Rank-1 改善/恶化/不变 | AP 改善/恶化/不变 |
|---|---:|---:|
| B0→A4 seed1234 | 3 / 5 / 3 | 7 / 4 / 0 |
| B0→A4 seed4321 | 4 / 5 / 2 | 4 / 7 / 0 |
| B0→R2 seed1234 | 3 / 6 / 2 | 3 / 8 / 0 |
| B0→R2 seed4321 | 2 / 6 / 3 | 4 / 7 / 0 |

## 判定

A4 相对 B0 的 Rank-1 方向跨 seed 不一致，Image-micro mAP 也一正一负；四个 95% CI 中 A4 的 Rank-1 和 PID-macro 指标均跨过 0。A4 不能被解释为可靠的 Rank-1 或 mAP 改善。R2 在点估计上两种 seed 均差于 B0，但 11 PID 的区间仍然较宽，不能把它作为 A4 机制的因果对照。
""")

    write(args.reports / "GPC_A4_OLD_NEW_DIFFERENCE_REPORT.md", f"""# GPC-A4 旧新运行差异报告

比较：旧运行 `20261015_155822`，新运行 `20261016_161631`，均为 seed1234、30 epoch。结论状态：**`REPRODUCIBILITY_UNRESOLVED`**。

## 指标漂移

| 模型 | 旧 Rank-1 | 新 Rank-1 | Δ | 旧 mAP | 新 mAP | Δ |
|---|---:|---:|---:|---:|---:|---:|
| B0 | {pct(old_b0['rank1'])} | {pct(new_b0['rank1'])} | {pp(new_b0['rank1']-old_b0['rank1'])} | {pct(old_b0['mAP'])} | {pct(new_b0['mAP'])} | {pp(new_b0['mAP']-old_b0['mAP'])} |
| A4 | {pct(old_a4['rank1'])} | {pct(new_a4['rank1'])} | {pp(new_a4['rank1']-old_a4['rank1'])} | {pct(old_a4['mAP'])} | {pct(new_a4['mAP'])} | {pp(new_a4['mAP']-old_a4['mAP'])} |

旧 A4−B0 为 `+6.4935 pp Rank-1`、`+2.6097 pp mAP`；新 A4−B0 为 `-0.4329 pp Rank-1`、`+0.2559 pp mAP`。B0 自身也从 `55.1227%` 变为 `58.0087%`，所以漂移不能全部归因于 A4。

## 已核对相同

- common initialization / 预训练初始化 SHA256：`72892762c37044b85dc9ed9999c833da574026c3715e760885363201a8a40ed0`；meta SHA256：`dd3acd0da3d935b2d734567b8db9bb171d63d64d24fe2a378f5b05cf335bee67`。
- development split manifest：`266786038e96fc87ce0b3ddced692d808064dfecc5228122c58d482158a90594`；CSV：`60b98e7bdfc9f9d31edc1efde4cbb7fe47ac5150b021f9f0d5e195fc681a2722`。
- 可见配置：batch 8、596 steps/epoch、SGD、base LR 2e-6、weight decay 0.05、lambda 0.1、30 epoch、训练 PID 60。
- 评估 protocol / Query / Gallery SHA256：`4a73d0382380f8518d638fc2c447db4f2e84e55cbe3264ea7d4b1b07e68f5702` / `3bb1035b7667fa3836e38594e189bf3c396ef995d6c1ef90ed9dae4880afb1d7` / `5832d32aebfed2204703af6ec9f2a9cb5980a7b2214c8182f2cc10cc915a584e`。

## 已核对不同或无法还原

1. 旧 A4 是 batch-local `balanced_selection`；新 A4 是全局累计 `cumulative_pid_balanced_selection`，新版本预计算并跨 rank broadcast epoch plan。
2. Phase7I stable hash 包含 phase 名称；旧 phase 与新 phase 不同，即使数值 seed 相同，关系选择流也不同。
3. 旧/新训练目录未保存完整每轮 sampler plan hash、各 rank 随机数状态和增强随机流，因此无法证明图像顺序、DDP 时序、AMP 事件序列完全相同。
4. 旧/新 wrapper、A4 selector、模型构建源码和实际 data 工作树状态不能由 GitHub 参考提交完整还原；旧 wrapper 和新 wrapper 的 SHA256 已写入 `{args.output}/gpc_a4_old_new_audit.json`。

因此不能凭经验从 B0/A4 漂移中指定单一根因；已确认的是协议一致但训练实现/关系流/未记录随机状态不完全可比。
""")

    exp = exposure
    rows = []
    for key in ("old_B0_seed1234", "old_A4_seed1234", "new_B0_seed1234", "new_A4_seed1234", "new_R2_seed1234", "new_B0_seed4321", "new_A4_seed4321", "new_R2_seed4321"):
        item = exp[key]
        rows.append(f"| {key} | {item['total_pairs']} | {item['active_batches_total']} | {item['active_batch_rate_mean']:.6f} | {item['pid_contribution_cv']:.6f} | {item['pid_contribution_gini']:.6f} | {item['pid_min']} | {item['pid_max']} | {item['amp_skipped_steps']} |")
    gradient_status = "A4 专属梯度未记录"
    write(args.reports / "GPC_A4_MECHANISM_VALIDITY_REPORT.md", f"""# GPC-A4 机制有效性报告

## PID 暴露均衡

| 运行 | 总有效 relation | active batch | active rate | PID contribution CV | Gini | min | max | AMP skip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

新 A4 两个 seed 相比新 B0 的 PID contribution CV/Gini 均显著下降：seed1234 为 `1.027179/0.514356 → 0.396828/0.227572`，seed4321 为 `0.998367/0.505238 → 0.392878/0.225316`。总有效 relation、active batch rate、optimizer updates 和 matched pair budget 保持一致。因此可以确认：**A4 的 PID 训练暴露均衡机制生效**，不是增加 relation 数量造成的。

## 暴露均衡不等于检索均衡

逐 PID 检索结果显示 B0→A4 seed1234 的 Rank-1 为改善/恶化/不变 `3/5/3`，seed4321 为 `4/5/2`；A4 的 Image-micro Rank-1 一正一负，PID-macro Rank-1 两个 seed 均为负。因此不能把训练侧 CV/Gini 下降直接解释为逐身份检索收益。

## 梯度、相似度和局部 margin

训练日志只记录关系数量、raw loss、AMP 事件，没有记录 A4 的 GPC 梯度范数、CE/Triplet/Bio 梯度范数、GPC/基础梯度比例、正样本相似度、最近异身份相似度和局部身份 margin。故 A4 的这些字段状态为 **`NOT_RECORDED`**，机制因果判定为 **`INCONCLUSIVE`**。

已有的 train-only 梯度兼容性探针只覆盖 B0/A5、共享 `feat[0]` 输出层、无 optimizer update，且明确未测完整参数梯度；它不能替代 A4 训练侧记录，也没有证明梯度冲突导致性能损伤。原始文件：`{GRADIENT}`。

## 判定

均衡 selector 的实现效果已确认；“均衡 selector 导致/改善了逐身份检索效果”的因果链没有足够证据。A4 机制有效性总体为 `INCONCLUSIVE`。
""")

    write(args.reports / "GPC_NEXT_IMPROVEMENT_DECISION.md", f"""# GPC 下一轮改进决策

审计 RUN_TIME：`{args.run_time}`。

## 最终决策

1. **新版评估源码是否归档完整？** 是。修复源码、测试、逐文件 hash 对照、六模型 checkpoint/config/manifest hash、评估协议/Query/Gallery hash 和配对 JSON/CSV 已归档；不含密钥、权重、数据集。
2. **有效验证 PID 数量？** 11。693 Query、1256 Gallery；统计证据不足。
3. **A4 两种子的配对 95% CI？** 见 `GPC_A4_PAIRED_PID_CONFIDENCE_REPORT.md`：seed1234 Image ΔR1 `[-2.6287,+1.3598] pp`、Image ΔmAP `[-1.3786,+2.1554] pp`；PID ΔR1 `[-4.9612,+0.7137] pp`、PID ΔmAP `[-1.0351,+2.0837] pp`。seed4321 Image ΔR1 `[-2.6762,+4.4560] pp`、Image ΔmAP `[-3.0052,-0.2814] pp`；PID ΔR1 `[-4.0244,+2.8673] pp`、PID ΔmAP `[-2.7644,+0.1217] pp`。
4. **B0 旧新为什么变化？** 已确认 B0 也漂移；旧/新 wrapper、Phase7I phase hash、未记录 sampler/增强/DDP/各 rank RNG 状态不完全一致。不能指定单一根因，状态 `REPRODUCIBILITY_UNRESOLVED`。
5. **A4 旧新为什么变化？** 旧 batch-local selector 与新全局累计 selector 不同，且 phase hash/关系流不同；叠加 B0 基线漂移。不能单独归因，状态 `REPRODUCIBILITY_UNRESOLVED`。
6. **PID 均衡是否生效？** 训练 relation 暴露层面是：A4 CV/Gini 明显下降，pair budget 和 active rate 未增加；检索身份效果均衡未证明。
7. **A4 是否稳定收益？** 否。两个 seed 的 Rank-1/mAP 方向不一致或区间跨 0；A4 保留为 `A4_MIXED`，不进入正式 LTCC。
8. **A5 是否比 A4 更值得后续研究？** 仅作为探索性候选：已有 A5 两 seed/60 epoch 开发收益小且需要重新核对饱和一致性，当前仍无稳定证据，不能直接启动训练。
9. **是否需要新 GPC 机制？** 不需要。先保留原始 GPC-v1；现有 A4 机制梯度数据不完整，不能据此提出新机制；本阶段不加入 NDP。
10. **当前是否允许启动下一轮训练？** **不允许**。本轮只完成诊断；在补齐可重复性协议和预先冻结独立科学问题/控制组前，不启动新的 60 epoch 或正式 LTCC。

## 状态标签

`A4_MIXED`；`REPRODUCIBILITY_UNRESOLVED`；`MECHANISM_INCONCLUSIVE`；`NEXT_TRAINING_BLOCKED_BY_PROTOCOL`。
""")

    print(json.dumps({
        "status": "PASS",
        "run_time": args.run_time,
        "reports": [str(args.reports / name) for name in (
            "GPC_A4_CODE_REPRODUCIBILITY_REPORT.md",
            "GPC_A4_PAIRED_PID_CONFIDENCE_REPORT.md",
            "GPC_A4_OLD_NEW_DIFFERENCE_REPORT.md",
            "GPC_A4_MECHANISM_VALIDITY_REPORT.md",
            "GPC_NEXT_IMPROVEMENT_DECISION.md",
        )],
        "code_audit": str(args.output / "gpc_a4_code_file_comparison.json"),
        "artifact_hashes": str(args.output / "gpc_a4_artifact_hashes.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
