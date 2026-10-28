#!/usr/local/bin/python3.10
"""Generate final NDP-V2 reports after the matched 1- and 3-epoch runs."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path("/home/DIFFER")
SHORT = ROOT / "log_NDP/ndp_v2_20261023_061358"
LONG = ROOT / "log_NDP/ndp_v2_20261023_070201"
REPORT = ROOT / "reports/ndp_v2_20261023_070201"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train(arm: str, seed: int) -> tuple[list[dict[str, str]], dict]:
    p = LONG / f"epoch3_{arm.lower()}_seed{seed}"
    return rows(p / "training_steps.csv"), json.loads((p / "run_summary.json").read_text())


def eval_json(base: Path, seed: int, update: int) -> dict[str, dict[str, float]]:
    z = json.loads((base / f"ltcc_seed{seed}_update{update}.json").read_text())
    return {r["arm"]: r["metrics"] for r in z["results"]}


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    code = {p.name: sha(p) for p in (ROOT / "scripts/ndp_e60_minimal_pair_runner.py", ROOT / "scripts/ndp_v2_reference_graph.py", ROOT / "scripts/ndp_v2_feature_export.py", ROOT / "scripts/ndp_v2_local_metrics.py")}
    code["gpc_v2_train.py"] = sha(Path("/data/DIFFER/scripts/gpc_v2_train.py"))
    code["eval_config"] = sha(Path("/data/DIFFER/diagnostics/reconstructed_ltcc_eval_config.yml"))
    e1 = {s: eval_json(SHORT, s, 596) for s in (1234, 4321)}
    e3 = {s: eval_json(LONG, s, 1788) for s in (1234, 4321)}
    local_rows = {s: rows(SHORT / f"local_metrics/seed{s}/local_node_summary_seed{s}.csv") for s in (1234, 4321)}

    def delta(e, seed, metric):
        return e[seed]["N2"][metric] - e[seed]["B0"][metric]

    d1 = {s: {m: delta(e1, s, m) for m in ("rank1", "mAP", "mINP")} for s in (1234, 4321)}
    d3 = {s: {m: delta(e3, s, m) for m in ("rank1", "mAP", "mINP")} for s in (1234, 4321)}
    m1 = {m: statistics.mean(d1[s][m] for s in d1) for m in d1[1234]}
    m3 = {m: statistics.mean(d3[s][m] for s in d3) for m in d3[1234]}

    long_train = []
    long_coverage = []
    checkpoint_lines = []
    for seed in (1234, 4321):
        for arm in ("B0", "N2"):
            rr, sm = train(arm, seed)
            active_b = sum(r["ndp_active"] == "True" for r in rr)
            skips = sum(int(r["found_inf"]) for r in rr)
            selected = sum(int(r["ndp_selected_count"]) for r in rr)
            active = sum(int(r["ndp_active_count"]) for r in rr)
            mean_ndp = statistics.mean(float(r["ndp"]) for r in rr)
            long_train.append(f"| {seed} | {arm} | {sm['effective_updates']} | {len(rr)} | {active_b}/{len(rr)} ({100*active_b/len(rr):.2f}%) | {selected} | {active} | {mean_ndp:.6f} | {skips} | {sm['failure'] or 'None'} |")
            long_coverage.append(f"| {seed} | {arm} | {sm['effective_updates']} | {active_b}/{len(rr)} ({100*active_b/len(rr):.2f}%) | {selected} | {active} | {mean_ndp:.6f} | {skips} |")
            node = LONG / f"epoch3_{arm.lower()}_seed{seed}/node_update1788"
            for artifact in ("model.pth", "rank0_state.pt", "rank1_state.pt"):
                checkpoint_lines.append(f"| {seed} | {arm} | `{artifact}` | `{sha(node / artifact)}` |")

    local_lines = []
    for seed in (1234, 4321):
        for r in local_rows[seed]:
            if int(r["update"]) == 596:
                local_lines.append(f"| {seed} | {r['arm']} | {r['correct_relation_rate']} | {r['mean_margin']} | {r['new_error_pid_count_vs_update0']} | {r['current_error_pid_count']} | {r['purity_at_10_train_bank']} |")

    def result_lines(ev, epoch_text):
        return "\n".join(f"| {s} | {a} | {epoch_text} | {ev[s][a]['rank1']:.6f} | {ev[s][a]['mAP']:.6f} | {ev[s][a]['mINP']:.6f} |" for s in (1234, 4321) for a in ("B0", "N2"))

    impl = f"""# NDP-V2 实现与训练记录（最终）

本报告汇总两个独立新时间戳：1 continuation epoch 的 `{SHORT}`，以及按训练正常后继续执行的 3 continuation epochs `{LONG}`。报告目录：`{REPORT}`。

## 实现结论

NDP-V2 已真实参与训练：N2 的 live anchor 保留 autograd，参考 update0 特征停止梯度；NDP loss、feature gradient 和模型参数更新均被观察到。基础损失始终是 `CE + Triplet + clipBio + 0.1 * original GPC-v1`，GPC selector、matched-budget、EVA02、sampler、Bio、Triplet 和评估协议未修改。

V2 使用每 seed 2048 个 train-only update0 anchor，每 anchor 4 个正样本和 5 个不同负 PID；历史正确关系为 seed1234=`1699/2048`、seed4321=`1660/2048`。`lambda_ndp=0.05`、`epsilon=0.02`。由于没有 E40/E45/E50 完整状态，B0/N2 均从同一 E60 rank state 恢复，并一致设置 LR=`1e-6`。

## 真实运行命令

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \\
  scripts/ndp_e60_minimal_pair_runner.py --mode full --arm B0|N2 --seed 1234|4321 \\
  --max-effective-updates 1788 --save-updates 596,1192,1788 \\
  --run-id ndp_v2_20261023_070201_epoch3_<arm>_s<seed> \\
  --run-dir /home/DIFFER/log_NDP/ndp_v2_20261023_070201/epoch3_<arm>_seed<seed> \\
  --reference-feature <seed update0 full feature> --graph <seed V2 graph> --initial-lr 1e-6
```

20 步 smoke 命令和 1 epoch 命令保存在 `{SHORT}` 的 `training_config.json` 与历史报告中；官方评估使用 `scripts/ndp_official_ltcc_nodes.py`，train-only 局部指标使用 `scripts/ndp_v2_local_metrics.py`。测试：`/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_minimal_pair_math.py` 为 `2 passed`，全部新增脚本通过 `py_compile`。

## 3-epoch真实训练统计

| Seed | Arm | 有效更新 | CSV batch行 | 激活 batch/率 | selected | active relation | mean NDP | AMP skip | failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(long_train)}

seed1234 B0/N2 分别有 2/1 次 AMP skip，seed4321 B0/N2 分别有 2/1 次 AMP skip；skip 后通过额外 batch 达到 1788 个有效更新，参数始终有限。四组 `node_update0596/1192/1788` 和 restore check 均已落盘。NDP 3 epoch active batch 率为 seed1234=`46.34%`、seed4321=`40.75%`，明显高于 N1 历史约10.5%–13.5%。

## update1788 checkpoint SHA256

| Seed | Arm | Artifact | SHA256 |
|---|---|---|---|
{chr(10).join(checkpoint_lines)}

## 代码与固定引擎 SHA256

| 文件 | SHA256 |
|---|---|
| `scripts/ndp_e60_minimal_pair_runner.py` | `{code['ndp_e60_minimal_pair_runner.py']}` |
| `scripts/ndp_v2_reference_graph.py` | `{code['ndp_v2_reference_graph.py']}` |
| `scripts/ndp_v2_feature_export.py` | `{code['ndp_v2_feature_export.py']}` |
| `scripts/ndp_v2_local_metrics.py` | `{code['ndp_v2_local_metrics.py']}` |
| `/data/DIFFER/scripts/gpc_v2_train.py`（只读） | `{code['gpc_v2_train.py']}` |
| `reconstructed_ltcc_eval_config.yml` | `{code['eval_config']}` |

远端 main 复核为 `e27d59ed74ff348061c43ffd799dc91af8d6e0f0`；本地 `/data/DIFFER` commit 为 `573b0502a231568d509fe3a463664c1a3c42e258`；未向 GitHub push。
"""

    comparison = f"""# NDP-V2 B0/N2 对照（1 epoch 与 3 epoch）

## 固定 LTCC update596：短程结果

| Seed | 模型 | 训练 Epoch | Rank-1 | mAP | mINP |
|---|---|---|---:|---:|---:|
{result_lines(e1, '1 epoch / 596 updates')}

| Seed | Δ Rank-1 | Δ mAP | Δ mINP |
|---|---:|---:|---:|
| 1234 | {d1[1234]['rank1']:+.6f} | {d1[1234]['mAP']:+.6f} | {d1[1234]['mINP']:+.6f} |
| 4321 | {d1[4321]['rank1']:+.6f} | {d1[4321]['mAP']:+.6f} | {d1[4321]['mINP']:+.6f} |
| 均值 | {m1['rank1']:+.6f} | {m1['mAP']:+.6f} | {m1['mINP']:+.6f} |

## 固定 LTCC update1788：3 epoch结果

| Seed | 模型 | 训练 Epoch | Rank-1 | mAP | mINP |
|---|---|---|---:|---:|---:|
{result_lines(e3, '3 epochs / 1788 updates')}

| Seed | Δ Rank-1 | Δ mAP | Δ mINP |
|---|---:|---:|---:|
| 1234 | {d3[1234]['rank1']:+.6f} | {d3[1234]['mAP']:+.6f} | {d3[1234]['mINP']:+.6f} |
| 4321 | {d3[4321]['rank1']:+.6f} | {d3[4321]['mAP']:+.6f} | {d3[4321]['mINP']:+.6f} |
| 均值 | {m3['rank1']:+.6f} | {m3['mAP']:+.6f} | {m3['mINP']:+.6f} |

1 epoch 的两个 seed 方向均为正；3 epoch 的 seed1234 变为负，seed4321 仍为正，均值 Rank-1 仅约 `{m3['rank1']:+.6f}`、mAP `{m3['mAP']:+.6f}`、mINP `{m3['mINP']:+.6f}`，不支持稳定的长程增益。3 epoch 结果完整来自 `{LONG}/ltcc_seed*_update1788.json`，没有用中间节点反向选参。

## train-only局部指标（update596）

局部指标来自固定 2048 anchor 和 9576-image train-only bank；逐关系 CSV：[seed1234]({SHORT}/local_metrics/seed1234/local_relation_metrics_seed1234.csv)、[seed4321]({SHORT}/local_metrics/seed4321/local_relation_metrics_seed4321.csv)。

| Seed | Arm | correct relation rate | mean margin | 新增错误 PID | 当前错误 PID | Purity@10 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(local_lines)}

seed1234 的 N2 局部 relation rate/Purity@10 比 B0 差，seed4321 current error PID 减少但 Purity@10 仍略降；局部安全性没有稳定证据。3 epoch 未重新导出全量局部特征，避免重复昂贵前向；因此不能把 update596 局部表误标为 update1788。

## 训练与评估完整性

- B0/N2 同 E60 起点、同 LR、同 sampler、同有效更新预算；N2 仅额外计算独立 NDP。
- 官方 query/gallery 只在预先固定的 update596/update1788节点报告，不参与参数选择。
- PPU EVA02 A/A 既有约0.4%差异保留；本轮不声称已修复。
- 正面：NDP 确实训练、V2 激活率显著高于 N1、1 epoch LTCC 两 seed 同向提升。
- 负面：3 epoch 增益不一致，seed1234 长程退化；局部指标存在附带损害；N2长跑仍有 AMP skip。
"""

    decision = f"""# NDP-V2 最终决策（含3 epoch结果）

## 最终结论

**不建议继续当前 NDP-V2 到 10–60 epoch；`CONTROLLED_10_EPOCH_PILOT = NO_GO`。** 1 epoch 的配对 LTCC 信号是正的，但 3 epoch 只在 seed4321保留、seed1234转负，均值接近零/略负；同时 update596 train-only 局部指标已有邻近身份附带损害。继续堆训练时长没有足够因果依据。

这不等于 NDP 已被证伪：它确实能训练且短程可能改善检索；但当前“2048 anchor + 5 PID max reactive protection”还不是可安全扩展的版本。

## 八个问题的直接回答

1. **NDP 是否真正参与训练？** 是。N2 smoke 和 1/3 epoch均有 live anchor 梯度、NDP loss 和参数更新。
2. **V2 是否比 N1 更容易激活？** 是。N1约10.5%–13.5% active batch；V2 1 epoch约42.4%–45.9%，3 epoch约40.75%–46.34%。
3. **是否改善 Rank-1/mAP？** 只在 1 epoch 两 seed 同向：seed1234 `+1.02/+0.70 pp`，seed4321 `+0.51/+0.42 pp`。3 epoch seed1234为 `-0.51/-0.33 pp`，seed4321为 `+0.77/+0.33 pp`，均值 Rank-1约`+0.13 pp`、mAP约`-0.004 pp`，不能称稳定改善。
4. **是否改善局部身份判别？** 未证明。update596 局部 margin/rate/Purity跨 seed不一致，Purity两 seed均略降；3 epoch没有相应全量局部导出，因此不做过度外推。
5. **是否损害附近身份？** 有真实轻度信号。seed1234 update596 N2 新增错误 PID=`129` vs B0=`97`、Purity=`0.970703` vs `0.972021`；seed4321 Purity=`0.967627` vs `0.968213`。这足以阻止当前版本长程扩展。
6. **哪个 seed 更好？** 短程 seed1234增益更大；3 epoch seed4321更好，seed1234变差，说明 seed依赖明显。
7. **是否值得继续10–60 epoch？** 当前版本不值得直接继续；10/60均 NO_GO。先做一个小范围、无网络训练的局部梯度预算/附带影响诊断，再决定新版本是否重新开短程训练。
8. **下一处具体改进？** 不是再扩大 anchor 或增加复杂分支。先按 anchor 分解五个负 PID 的 NDP 梯度和邻域一阶影响，确认 max-negative 监督是否把一个 anchor 的预算集中到错误方向；以此为因果证据，预注册每-anchor关系预算/局部 collateral gate，再做新的匹配 B0/N3短程实验。

## 状态标签

- `NDP_TRAINING_PARTICIPATION = YES`
- `NDP_V2_SHORT_EPOCH_SIGNAL = POSITIVE_ON_2_SEEDS`
- `NDP_V2_3_EPOCH_SUSTAINABILITY = FAIL_INCONSISTENT`
- `NDP_LOCAL_MECHANISM = INCONCLUSIVE`
- `NDP_MECHANISM = INCONCLUSIVE`
- `A/A_REPRODUCIBILITY = FAIL`
- `CONTROLLED_10_EPOCH_PILOT = NO_GO`
- `CONTROLLED_60_EPOCH_TRAINING = NO_GO`

## 证据路径

- 最终日志：[3-epoch日志]({LONG})
- 1 epoch日志：[短程日志]({SHORT})
- 实现报告：[NDP_V2_IMPLEMENTATION.md]({REPORT}/NDP_V2_IMPLEMENTATION.md)
- 对照报告：[NDP_V2_B0_COMPARISON.md]({REPORT}/NDP_V2_B0_COMPARISON.md)
- 官方结果：[{LONG}/ltcc_seed1234_update1788.json]({LONG}/ltcc_seed1234_update1788.json)、[{LONG}/ltcc_seed4321_update1788.json]({LONG}/ltcc_seed4321_update1788.json)

本结论没有把未完成的10/60 epoch写成完成，也没有用1 epoch正结果掩盖3 epoch seed1234的负结果。
"""
    (REPORT / "NDP_V2_IMPLEMENTATION.md").write_text(impl, encoding="utf-8")
    (REPORT / "NDP_V2_B0_COMPARISON.md").write_text(comparison, encoding="utf-8")
    (REPORT / "NDP_V2_FINAL_DECISION.md").write_text(decision, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
