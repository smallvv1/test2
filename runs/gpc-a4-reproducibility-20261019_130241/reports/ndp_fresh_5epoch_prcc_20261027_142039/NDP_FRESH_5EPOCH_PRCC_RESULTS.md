# NDP-Fresh PRCC 五轮三臂配对实验结果

## 结论摘要

本轮完成了 PRCC train-only 上的 B0、固定参考 TOP5、epoch 级更新参考的 NFRESH 三臂实验，两个 seed 均运行到 5 个 continuation raw epoch，并在 continuation epoch 1、3、5 固定评估 C→A。

结论：NFRESH 的 RNG 修复和真实训练链路均有效，但收益没有稳定复现。Seed1234 在第 5 个 continuation epoch 相对 B0 为 Rank-1 `+1.214 pp`、mAP `+0.715 pp`；Seed4321 同一节点则为 Rank-1 `-0.649 pp`、mAP `-0.706 pp`。因此当前不能声称“每 epoch 更新参考特征能在两个 seed 上稳定优于冻结参考和 B0”。

本轮不建议直接扩展 10/60 epoch，也不建议继续堆叠相似 Margin 变体。NFRESH 可以保留为一个有证据但未稳定的候选，下一步应优先做独立 seed 重复和关系级增量监督核查。

## 实验身份与可复现信息

- GitHub `main` 实际核对结果：`e46b8f761c2dd886f532449c9f4883292157792b`。
- 实验目录：`/home/DIFFER/log_NDP/ndp_fresh_5epoch_prcc_20261027_142039/`。
- 本服务器 `/home/DIFFER/` 不是 Git checkout；因此报告不冒充本地 HEAD 等于用户指定提交。运行代码和所有新日志均来自该服务器工作树。
- PRCC 训练图像：17896 张 train-only 图像；参考图未使用 query/gallery。
- GPC-v1 源快照 SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`。
- NDP runner SHA256：`8c3cd14d08c8fd35a18f717aaee617a7c7b220cdf2376a3908cdaf363f2f6aef`。
- 测试文件 SHA256：`3bf5a90f15ff6674f66aba657dfb6b7430e6d4f767c7fb3fe01a6935f5dc48ef`。
- Seed1234 common 模型 SHA256：`2cb7e822fdef9133aa250ac81fe2ab21295ad8af98f2e4c5811f8f56af4da98f`。
- Seed4321 common 模型 SHA256：`3155a876f0b5b067ccf4c3f79a67ff31a93e81f0210ff693dd844f632956861b`；该 seed 的 rank state 在正式运行前被清理，因此从同一模型权重重建了 `common_seed4321_rebuilt`，三个 seed4321 arm 均使用该重建状态。
- 固定 GPC run-id：`ndp_prcc_baseline_fix_20261025_130457`；arm 名称没有进入 GPC 哈希上下文。
- 固定参数：`lambda_ndp=0.05`；原始 Top5 动态关系图；t1 Margin 目标；原始 B0、GPC、EVA02、Triplet、Bio、sampler 和 PRCC C→A 评估协议均未改动。

## 实际运行命令

正式训练使用如下命令模板，B0 不传 feature/graph，TOP5/NFRESH 传入 seed-specific t1 特征和 Top5 图：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train --seed {1234|4321} \
  --arm {B0|NDYN_TOP5|NFRESH} \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --resume-checkpoint /home/DIFFER/log_NDP/ndp_fresh_5epoch_prcc_20261027_142039/{common_seed1234|common_seed4321_rebuilt} \
  --epochs 5 --eval-epochs 1,3,5 \
  --save-updates 551,1102,1653,2755 --diagnostic-steps 20
```

Seed1234 NFRESH 在 update1653 后遭遇一次系统 SIGKILL；rank0/rank1 状态完整，随后从该节点在 `nfresh_seed1234_5raw_recovery_from03/` 恢复剩余两 raw epoch。Seed4321 NFRESH 同样在第 3 节点后结束，使用 `nfresh_seed4321_5raw_recovery_from03/` 完成剩余两轮。前三轮没有重跑，所有恢复节点和 AMP 事件均保留。

## 单元测试与 Smoke

执行：

```bash
/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_v5_b0_loss_contract.py
/usr/local/bin/python3.10 -m py_compile scripts/ndp_v5_prcc_runner.py tests/test_ndp_v5_b0_loss_contract.py
```

结果：`17 passed`，py_compile 通过。

此外完成了 B0、TOP5、NFRESH 的 20 步 Smoke；NFRESH 延长 Smoke 至 551 raw batch，确认出现非零 NDP Loss，记录到的最大 NDP 特征梯度约 `0.01254`。正式 B0 的 `ndp`、NDP 关系数和 NDP 梯度列均为零。

## PRCC C→A 结果

下表单位为百分比；“epoch”是 continuation epoch，不是从 common state 保存的绝对 epoch。Seed4321 B0 只采用重建 common state 后的后三行，早先状态清理前残留在同一 CSV 的旧行不纳入比较。

| Seed | 模型 | epoch 1 Rank-1 / mAP | epoch 3 Rank-1 / mAP | epoch 5 Rank-1 / mAP |
|---|---|---:|---:|---:|
| 1234 | B0 | 62.38 / 56.66 | 63.67 / 59.21 | 62.86 / 59.51 |
| 1234 | TOP5 | 63.20 / 57.00 | 63.70 / 59.46 | 63.90 / 60.29 |
| 1234 | NFRESH | 63.03 / 56.76 | 63.73 / 59.34 | **64.07 / 60.22** |
| 4321 | B0 | 62.07 / 56.96 | 63.22 / 58.90 | **64.41 / 60.95** |
| 4321 | TOP5 | 61.45 / 56.86 | 62.57 / 59.10 | 64.04 / 60.51 |
| 4321 | NFRESH | 61.78 / 56.66 | 63.65 / 58.74 | 63.76 / 60.25 |

### NFRESH 相对 B0 / TOP5 的百分点差值

| Seed | epoch | NFRESH−B0 Rank-1 / mAP | NFRESH−TOP5 Rank-1 / mAP |
|---|---:|---:|---:|
| 1234 | 1 | +0.649 / +0.097 | -0.169 / -0.237 |
| 1234 | 3 | +0.056 / +0.126 | +0.028 / -0.125 |
| 1234 | 5 | **+1.214 / +0.715** | +0.169 / -0.065 |
| 4321 | 1 | -0.282 / -0.295 | +0.339 / -0.200 |
| 4321 | 3 | +0.423 / -0.168 | +1.073 / -0.359 |
| 4321 | 5 | **-0.649 / -0.706** | -0.282 / -0.264 |

mINP：当前 PRCC 工程使用的 `R1_mAP_eval` 没有暴露 mINP，所有节点均标记为 unavailable；没有用其他指标替代或编造 mINP。

## NDP 激活、梯度与 AMP

### 固定节点统计

`active batch rate` 使用 runner 的 DDP 全局统计字段；节点行的 selected/active relation 是截至该评估节点的累计统计。NFRESH：

| Seed | epoch | active batch rate | selected relations | active relations | NDP mean loss | 非零梯度证据 |
|---|---:|---:|---:|---:|---:|---|
| 1234 | 1 | 10.34% | 253 | 140 | 0.000481 | 有 |
| 1234 | 3 | 8.94% | 509 | 232 | 0.000453 | 有 |
| 1234 | 5（恢复段） | 7.40% | 504 | 185 | 0.000407 | 有 |
| 4321 | 1 | 9.62% | 188 | 122 | 0.000427 | 有 |
| 4321 | 3 | 7.40% | 377 | 181 | 0.000362 | 有 |
| 4321 | 5（恢复段） | 6.62% | 375 | 166 | 0.000394 | 有 |

完整 5 raw epoch 的累计 active batch 粗略为 Seed1234 `474/2755`（约 17.2% 的本地 raw batch，runner 的 DDP 全局 active-rate 口径约 8.6%），Seed4321 `309/2755`（本地约 11.2%，全局口径约 5.6%）；节点表保留 runner 原始口径，避免把 DDP 聚合数和本地 batch 数混淆。NDP 特征梯度在正式训练日志中确实非零，但只在诊断记录的激活步骤中采样；这证明梯度链路，不代表每个 batch 都有相同梯度。

AMP：

- B0 Seed1234：2754/2755 effective updates，1 次 skip。
- B0 Seed4321（重建 common）：2754/2755 effective updates，1 次 skip。
- TOP5 Seed1234：2755/2755 effective updates。
- TOP5 Seed4321：2754/2755 effective updates，1 次 skip。
- NFRESH Seed1234：正式主段及恢复段合计存在 1 次记录到的 skip；最终状态有效，恢复段 1101/1102 effective updates。
- NFRESH Seed4321：主段和恢复段均完成，恢复段 1102/1102 effective updates。

raw batch 和 effective optimizer update 分开记录，没有通过补跑 batch 把不同 AMP 轨迹伪装成严格相同。

## Fresh bank RNG 验证

`extract_train_feature_bank()` 现在在辅助特征提取前保存 Python、NumPy、CPU torch 及 rank-local PPU/CUDA RNG；使用独立 CPU DataLoader generator、`shuffle=False`、`num_workers=0`，并在提取结束后恢复 RNG 和模型 train/eval 状态。

所有正式 NFRESH fresh-bank manifest 的 `rng_state_preserved` 均为 `true`，且 before/after digest 相同。例如 Seed1234 第一个 bank 的 digest 为：

```text
0383c3addac83dec45c2ee72a4bc6b67309c0c3489ab6109d659fb1b8accc376
```

Seed1234 固定关系 PID 数为 256，Seed4321 为 192；每个 bank 形状均为 `17896 x 1024`。固定 Top5 关系集合在 NFRESH 训练中没有重新排序或重新选择。

NFRESH 参考 bank Margin 均值变化：

- Seed1234：约 `0.005298 → 0.011194`，变化 `+0.005896`。
- Seed4321：约 `0.006332 → 0.008906`，变化 `+0.002574`。

这确认 NFRESH 的唯一变化确实包括 epoch 级参考端点更新；但 bank Margin 变化本身不能证明网络识别性能提升。

## 判断

1. **RNG 修复是否通过？** 通过。实际 manifest 和单元测试均证明辅助 bank 提取前后 RNG 状态保持；模型训练模式也被恢复。
2. **NFRESH 是否真实参与训练？** 是。正式日志有非零 NDP Loss、NDP 选中/激活关系及非零特征梯度；B0 的 NDP 列严格为零。
3. **是否稳定优于冻结 TOP5？** 否。Seed1234 epoch5 的 Rank-1 略高但 mAP 略低；Seed4321 epoch5 两项均低于 TOP5。
4. **是否稳定优于 B0？** 否。Seed1234 epoch5 明显提升，但 Seed4321 epoch5 两项下降；epoch1/3 也没有两个 seed 的两项同时为正。
5. **是否证明 fresh bank 方法有效？** 没有。结果说明它可能在特定 seed/训练轨迹上有增量，但当前证据不足以排除 seed 依赖、PPU 非确定性、AMP 轨迹和参考关系分布差异。

## 未完成事项与下一步

- 本轮未做 LTCC；PRCC 结果未稳定通过双 seed 门禁，因此不直接迁移到 LTCC。
- PPU 完整 EVA02 反向 A/A 约 0.4% 的既有数值非确定性仍是结果解释限制；本轮没有重新进行整套算子审计。
- NFRESH 主段被 SIGKILL 后采用完整 rank state 恢复，恢复段与原始三轮日志均保留；没有隐藏失败。
- 不建议继续 10/60 epoch 或继续扫描静态 Margin 参数。
- 若继续，最值得做的是：用同一 strict common-state 生成至少一个独立 seed，先验证 NFRESH 的方向一致性；随后比较真正只在原 Triplet 当前 batch 之外提供监督的关系，并单独报告 relation-level 增量，而不是继续增加保护强度。

**最终结论：本轮 NFRESH = 真实可训练、RNG 修复通过、性能信号存在但不稳定；不能认定为稳定收益。**
