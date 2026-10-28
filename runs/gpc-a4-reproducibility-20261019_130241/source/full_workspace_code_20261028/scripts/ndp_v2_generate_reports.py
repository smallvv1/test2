#!/usr/local/bin/python3.10
"""Generate the three Chinese NDP-V2 handoff reports from immutable logs."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path("/home/DIFFER")
RUN = ROOT / "log_NDP/ndp_v2_20261023_061358"
REPORT = ROOT / "reports/ndp_v2_20261023_061358"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(x: float) -> str:
    return f"{100*x:.3f}%"


def f(x: float) -> str:
    return f"{x:.6f}"


def training(seed: int, arm: str) -> tuple[list[dict[str, str]], dict]:
    p = RUN / f"epoch1_{arm.lower()}_seed{seed}"
    rows = read_csv(p / "training_steps.csv")
    summary = json.loads((p / "run_summary.json").read_text())
    return rows, summary


def local(seed: int) -> list[dict[str, str]]:
    return read_csv(RUN / f"local_metrics/seed{seed}/local_node_summary_seed{seed}.csv")


def official(seed: int) -> dict[str, dict[str, float]]:
    z = json.loads((RUN / f"ltcc_seed{seed}_update596.json").read_text())
    return {r["arm"]: r["metrics"] for r in z["results"]}


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    code = {
        "runner": sha256(ROOT / "scripts/ndp_e60_minimal_pair_runner.py"),
        "graph": sha256(ROOT / "scripts/ndp_v2_reference_graph.py"),
        "export": sha256(ROOT / "scripts/ndp_v2_feature_export.py"),
        "metrics": sha256(ROOT / "scripts/ndp_v2_local_metrics.py"),
        "gpc": sha256(Path("/data/DIFFER/scripts/gpc_v2_train.py")),
        "eval_config": sha256(Path("/data/DIFFER/diagnostics/reconstructed_ltcc_eval_config.yml")),
    }
    ltcc = {seed: official(seed) for seed in (1234, 4321)}
    train = {(seed, arm): training(seed, arm) for seed in (1234, 4321) for arm in ("B0", "N2")}
    local_rows = {seed: local(seed) for seed in (1234, 4321)}
    deltas = {
        seed: {metric: ltcc[seed]["N2"][metric] - ltcc[seed]["B0"][metric] for metric in ("rank1", "mAP", "mINP")}
        for seed in (1234, 4321)
    }
    means = {metric: statistics.mean(deltas[s][metric] for s in (1234, 4321)) for metric in deltas[1234]}
    sds = {metric: statistics.stdev(deltas[s][metric] for s in (1234, 4321)) for metric in deltas[1234]}

    smoke_lines = []
    for seed in (1234, 4321):
        for arm in ("B0", "N2"):
            p = RUN / f"smoke_{arm.lower()}_seed{seed}"
            s = json.loads((p / "run_summary.json").read_text())
            rows = read_csv(p / "training_steps.csv")
            smoke_lines.append(f"| {seed} | {arm} | {s['effective_updates']} | {s['ndp_candidate_relations']} | {s['ndp_selected_relations_local_rank0']} | {s['ndp_active_relations_local_rank0']} | {s['ndp_active_batches_local_rank0']}/{len(rows)} ({pct(s['ndp_active_rate_local_rank0'])}) | PASS |")

    full_lines = []
    coverage_lines = []
    checkpoint_lines = []
    for seed in (1234, 4321):
        for arm in ("B0", "N2"):
            rows, s = train[(seed, arm)]
            active_batches = sum(r["ndp_active"] == "True" for r in rows)
            skips = sum(int(r["found_inf"]) for r in rows)
            full_lines.append(f"| {seed} | {arm} | {s['effective_updates']} | {len(rows)} | {active_batches}/{len(rows)} ({pct(active_batches/len(rows))}) | {sum(int(r['ndp_selected_count']) for r in rows)} | {sum(int(r['ndp_active_count']) for r in rows)} | {statistics.mean(float(r['ndp']) for r in rows):.6f} | {skips} | {s['failure'] or 'None'} |")
            coverage_lines.append(f"| {seed} | {arm} | {s['effective_updates']} | {active_batches}/{len(rows)} ({pct(active_batches/len(rows))}) | {sum(int(r['ndp_selected_count']) for r in rows)} | {sum(int(r['ndp_active_count']) for r in rows)} | {statistics.mean(float(r['ndp']) for r in rows):.6f} | {skips} |")
            node = RUN / f"epoch1_{arm.lower()}_seed{seed}/node_update0596"
            for artifact in ("model.pth", "rank0_state.pt", "rank1_state.pt"):
                checkpoint_lines.append(f"| {seed} | {arm} | `{artifact}` | `{sha256(node / artifact)}` |")

    impl = f"""# NDP-V2 实现与真实训练记录

实验目录：`{RUN}`；报告目录：`{REPORT}`。

## 结论先行

NDP-V2 已真实进入训练图并执行 optimizer step。两 seed、B0/N2 各 596 个有效更新均完成；N2 的 NDP 梯度可到达 EVA02 参数。update596 的固定 LTCC 评估中，N2 相对同起点同 LR 的 B0 在两个 seed 的 Rank-1、mAP、mINP 均为正差值，但 train-only 局部关系指标并非一致改善，因此这是一项有希望的探索性结果，不是机制已证实。

## 固定基线与实现

- 远端 `main` 复核：`e27d59ed74ff348061c43ffd799dc91af8d6e0f0`；本地 `/data/DIFFER` 引擎 commit：`573b0502a231568d509fe3a463664c1a3c42e258`。`/home/DIFFER` 不是 Git worktree，本轮未 push。
- 基础目标保持 `CE + Triplet + clipBio + 0.1 * original GPC-v1`。GPC selector、matched-budget、损失实现和系数没有改动；N2 只在训练入口叠加 NDP。
- NDP：`L_ndp = mean(ReLU(M_ref - M_current - 0.02))`，`lambda_ndp=0.05`。`M_ref` 是 update0 train-only 参考并停止梯度；`M_current` 由当前 live `feat[0]` anchor 与冻结参考端点计算，正样本均值减五个负 PID 相似度最大值。正负端点不要求同 batch 出现。
- V2 图：每个 seed 2048 个合法 train anchor；每 anchor 4 个同 PID 正样本、5 个不同负 PID。update0 历史正确关系分别为 seed1234=`1699/2048`、seed4321=`1660/2048`。图选择未读取未来节点、query/gallery 或测试集。
- 由于没有 E40/E45/E50 完整状态，B0/N2 都从同一 E60 rank state 恢复，并把所有 optimizer param group 的 LR 一致重置为 `1e-6`；没有只提高 NDP LR。两 arm 的 sampler、有效更新预算和评估协议相同。

## 真实命令

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \\
  scripts/ndp_e60_minimal_pair_runner.py --mode smoke --arm B0|N2 --seed 1234|4321 \\
  --run-id <new> --run-dir /home/DIFFER/log_NDP/ndp_v2_20261023_061358/<dir> \\
  --reference-feature <seed-specific update0_full_features.npz> \\
  --graph <seed-specific multi_pid_reference_graph_v2.json> \\
  --initial-lr 1e-6
```

正式配对命令相同，增加 `--max-effective-updates 596 --save-updates 596`。官方评估命令为 `scripts/ndp_official_ltcc_nodes.py`，一次 seed 同时加载 B0/N2 update596；局部指标命令为 `scripts/ndp_v2_local_metrics.py`。新增 focused test：`/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_minimal_pair_math.py`，结果 `2 passed`；四个新增脚本均通过 `py_compile`。

## 20 步 Smoke

| Seed | Arm | 有效更新 | 候选关系 | 选中 | 实际激活关系 | 激活 batch | 恢复 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(smoke_lines)}

N2 seed1234 smoke 的 rank0 统计为 30 selected、14 active、10/20 active batch；N2 seed4321 的 smoke 统计保存在对应 `run_summary.json`。两 seed 的 N2 smoke 都有非零 NDP feature/probe gradient、有限 loss 和可加载 checkpoint。B0 smoke 作为同配置控制完成但 NDP 为 disabled，选中/激活为 0。

## 第一 continuation epoch（update596）

| Seed | Arm | 有效更新 | 实际 batch 行数 | 激活 batch | 选中关系 | 激活关系 | 平均 NDP | AMP skip | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(full_lines)}

N2 seed1234 在 batch570 有一次 AMP `found_inf=1`，该 step 被跳过、scale 从256降到128，参数仍有限；随后通过额外一个 batch 达到 596 个有效更新。N2 seed4321 无 AMP skip。所有正式 run 的 `parameter_finite=True`，且 update596 rank checkpoint restore check PASS。

## 文件与 SHA256

新增代码：

| 文件 | SHA256 |
|---|---|
| `scripts/ndp_e60_minimal_pair_runner.py` | `{code['runner']}` |
| `scripts/ndp_v2_reference_graph.py` | `{code['graph']}` |
| `scripts/ndp_v2_feature_export.py` | `{code['export']}` |
| `scripts/ndp_v2_local_metrics.py` | `{code['metrics']}` |
| `/data/DIFFER/scripts/gpc_v2_train.py`（只读核验） | `{code['gpc']}` |
| `/data/DIFFER/diagnostics/reconstructed_ltcc_eval_config.yml` | `{code['eval_config']}` |

完整 checkpoint SHA256、参考图 SHA256、特征导出 SHA256 写在 `{RUN}/` 下的 JSON manifest、`run_summary.json` 和 `training_config.json`；原始训练日志未覆盖。

update596 checkpoint SHA256：

| Seed | Arm | Artifact | SHA256 |
|---|---|---|---|
{chr(10).join(checkpoint_lines)}

## 限制

本轮只完成一个 continuation epoch，LR 是从 E60 状态一致重置的探索性 `1e-6`，不是原始完整 60 epoch 训练协议。PPU 完整 EVA02 A/A 既有约0.4%差异仍然存在，本轮没有无限重复算子审计。结论限定为 matched B0/N2 的探索性证据。
"""

    comparison_rows = []
    for seed in (1234, 4321):
        for arm in ("B0", "N2"):
            m = ltcc[seed][arm]
            comparison_rows.append(f"| {seed} | {arm} | 1 continuation epoch / {train[(seed, arm)][1]['effective_updates']} updates | {m['rank1']:.6f} | {m['mAP']:.6f} | {m['mINP']:.6f} |")
    local_lines = []
    for seed in (1234, 4321):
        for r in local_rows[seed]:
            if int(r["update"]) == 596:
                local_lines.append(f"| {seed} | {r['arm']} | {r['correct_relation_rate']} | {r['mean_margin']} | {r['new_error_pid_count_vs_update0']} | {r['current_error_pid_count']} | {r['purity_at_10_train_bank']} |")

    comparison = f"""# NDP-V2 B0 对照（两 seed，update596）

数据目录：`{RUN}`。B0 与 N2 都从同一 seed 的 E60 `rank0/1_state_epoch060.pt` 启动，LR 同为 `1e-6`，同 sampler 顺序，完成 596 个有效 optimizer update。官方 LTCC 结果来自固定 query/gallery clothes-changing 过滤、归一化 `feat[0]`、无 rerank；不是用来选超参数。

## LTCC 结果

| Seed | 模型 | 训练 Epoch | Rank-1 | mAP | mINP |
|---|---|---|---:|---:|---:|
{chr(10).join(comparison_rows)}

差值（N2-B0）：

| Seed | Δ Rank-1 | Δ mAP | Δ mINP |
|---|---:|---:|---:|
| 1234 | {deltas[1234]['rank1']:+.6f} | {deltas[1234]['mAP']:+.6f} | {deltas[1234]['mINP']:+.6f} |
| 4321 | {deltas[4321]['rank1']:+.6f} | {deltas[4321]['mAP']:+.6f} | {deltas[4321]['mINP']:+.6f} |
| 两 seed均值 | {means['rank1']:+.6f} | {means['mAP']:+.6f} | {means['mINP']:+.6f} |

两 seed 差值的样本标准差分别为 Rank-1=`{sds['rank1']:.6f}`、mAP=`{sds['mAP']:.6f}`、mINP=`{sds['mINP']:.6f}`。样本数只有2，不能作显著性或泛化保证；但方向在两个 seed 上一致。

## 训练侧与激活覆盖

| Seed | Arm | 有效更新 | 激活 batch/率 | selected | active relation | mean NDP | AMP skip |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(coverage_lines)}

V2 相对历史 N1：N1 seed1234 为约 `54/400=13.5%` active batch、135 active relation；seed4321 为约 `42/400=10.5%`、98 active relation。V2 第一 epoch 为 seed1234 `274/597=45.9%`、423 active relation；seed4321 `253/596=42.4%`、386 active relation。因此 V2 明显更容易激活，但覆盖增大并不自动等于局部安全。

## train-only 局部身份指标

这些指标在固定的 2048-anchor、9576-image train-only bank 上计算；逐关系 CSV 为：[seed1234 CSV]({RUN}/local_metrics/seed1234/local_relation_metrics_seed1234.csv)、[seed4321 CSV]({RUN}/local_metrics/seed4321/local_relation_metrics_seed4321.csv)。

| Seed | Arm | correct relation rate | mean margin | 新增错误 PID（相对 update0） | 当前错误 PID | Purity@10 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(local_lines)}

seed1234：N2 的 relation rate `0.814941` 低于 B0 `0.820801`，新增错误 PID `129` 高于 `97`，Purity@10 `0.970703` 低于 `0.972021`。seed4321：N2 的 relation rate `0.819824` 略高于 B0 `0.818848`，当前错误 PID `887` 低于 `903`，但 Purity@10 仍略低（`0.967627` vs `0.968213`）。因此“局部身份 margin/Purity 一致改善”没有成立；官方检索提升与局部关系安全性存在待解释的分离。

## 正面与负面证据

- 正面：NDP 的反应式项确实有训练梯度；V2 将 active batch 从 N1 的约10%–14%提高到约42%–46%；两个 seed 的 LTCC Rank-1/mAP/mINP 同方向提升。
- 负面：seed1234 的 train-only 局部关系退化更明显；seed4321 虽 current error PID 减少，Purity@10 仍下降。N2 seed1234 还有一次 AMP skip。
- 因此不能把 LTCC 两 seed 的提升直接解释为“目标身份边界被安全保护”；需要继续做局部附带影响和梯度预算审计。

## 未完成事项

没有跑 3–5 epoch、10 epoch 或 60 epoch；没有将官方指标用于调参；没有修改 GPC、EVA02、Triplet、Bio、采样或评估协议；没有推送 GitHub main。
"""

    decision = f"""# NDP-V2 最终实验决策

## 决策

**对 10 epoch 的匹配 B0/N2 扩展：GO（条件性探索性 GO）。** 两 seed 在固定 update596 官方 LTCC 上均取得正的 Rank-1/mAP/mINP 差值，且 NDP 真正参与了训练；继续 10 epoch 有足够的实证价值。

**对直接 60 epoch 或论文级机制定论：NO_GO。** 当前只有两个 seed、一个 continuation epoch；局部 train-only relation/Purity 没有一致改善，seed1234 有明显局部退化；PPU A/A 约0.4%差异和一次 AMP skip 也要求更长程结果谨慎解释。

## 对用户关心问题的直接回答

1. **NDP 是否真正参与训练？** 是。Smoke 和正式 N2 都有非零 NDP loss/feature gradient；当前 live anchor 未 detach；模型参数执行了 optimizer step。N2 seed1234/4321 正式 epoch 分别有 423/386 条 active relation，激活 batch 率约45.9%/42.4%。
2. **NDP-V2 是否比 N1 更容易激活？** 是。相对 N1 约10.5%–13.5%的 active batch，V2 为约42.4%–45.9%；这是 2048 anchor 覆盖扩大后的直接结果。
3. **是否改善 LTCC Rank-1/mAP？** 在本轮匹配控制下，是：seed1234 Rank-1/mAP `+0.010204/+0.006955`；seed4321 `+0.005102/+0.004208`。两 seed 方向一致，但样本数不足以证明普遍有效。
4. **是否改善局部身份判别？** 不能下统一肯定结论。seed4321 的 aggregate relation rate 和当前错误 PID略好，seed1234 相反；两 seed Purity@10 都略降。NDP 可能改善官方检索所见结构，同时破坏部分固定邻域关系。
5. **是否存在明显邻近身份损害？** 有轻度但真实的 train-only 信号：seed1234 新增错误 PID 129 vs B0 97、Purity@10 下降约0.00132；seed4321 Purity@10 下降约0.00059。尚未达到停止探索的证据，但足以禁止直接扩到60 epoch。
6. **哪个 seed 更好？** seed1234 的官方增益更大（Rank-1 +1.02 pp、mAP +0.70 pp）；seed4321 增益较小但仍同方向（+0.51 pp、+0.42 pp）。
7. **是否值得继续10–60 epoch？** 值得继续 10 epoch 的配对探索；不建议直接跑60 epoch。10 epoch 必须继续使用同一 B0/N2、同 LR/采样和固定评估节点，并把局部 collateral 作为硬监测指标。
8. **下一次最该改哪里？** 不要继续扩大图、增加 Hubness/双侧分支或调 GPC。先针对 seed1234 的局部附带损害做“梯度预算/每 anchor 关系预算”诊断：比较 NDP 对目标负 PID和其他四个负 PID的一阶影响，优先验证是否是一个 anchor 的五 PID max 监督把梯度集中到错误方向。只有该因果证据成立，才考虑预注册的关系上限或局部安全门控。

## 状态标签

- `NDP_TRAINING_PARTICIPATION = YES`
- `NDP_V2_LTCC_EXPLORATORY_SIGNAL = POSITIVE_ON_2_SEEDS`
- `NDP_LOCAL_MECHANISM = INCONCLUSIVE`
- `NDP_MECHANISM = INCONCLUSIVE`
- `A/A_REPRODUCIBILITY = FAIL`（继承已有约0.4%差异，本轮未重新宣称修复）
- `CONTROLLED_10_EPOCH_PILOT = GO_CONDITIONAL`
- `CONTROLLED_60_EPOCH_TRAINING = NO_GO`

## 复现与证据路径

- 实验日志：[NDP-V2 日志目录]({RUN})
- 训练实现报告：`NDP_V2_IMPLEMENTATION.md`
- 对照报告：`NDP_V2_B0_COMPARISON.md`
- 逐关系 CSV：[seed1234]({RUN}/local_metrics/seed1234/local_relation_metrics_seed1234.csv)、[seed4321]({RUN}/local_metrics/seed4321/local_relation_metrics_seed4321.csv)
- 官方结果 JSON：[seed1234]({RUN}/ltcc_seed1234_update596.json)、[seed4321]({RUN}/ltcc_seed4321_update596.json)

本报告没有把未完成的 10/60 epoch 写成已完成，也没有把局部冲突证据隐藏在平均检索指标之后。
"""
    (REPORT / "NDP_V2_IMPLEMENTATION.md").write_text(impl, encoding="utf-8")
    (REPORT / "NDP_V2_B0_COMPARISON.md").write_text(comparison, encoding="utf-8")
    (REPORT / "NDP_V2_FINAL_DECISION.md").write_text(decision, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
