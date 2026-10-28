# NDP-Listwise PRCC 五轮验证结果

实验目录：`/home/DIFFER/log_NDP/ndp_listwise_5epoch_prcc_20261028_135736/`
用户指定最新提交：`926d8d2b2c0fa3a5a3708fce1314d92625febd3e`
固定 GPC run-id：`ndp_prcc_baseline_fix_20261025_130457`

## 结论

本轮已完成两个 seed 的 B0/NLIST 五个 raw continuation epoch。由于旧第 3 epoch rank state 被清理，四组均从同一 seed-specific t1 common state 重新建立完整训练片段；因此本轮 B0/NLIST 在每个 seed 内严格配对。

五轮结果没有显示跨 seed 的稳定收益：

- Seed4321：NLIST 在 epoch3、epoch5 的 Rank-1/mAP 均低于 B0。
- Seed3456：NLIST 在 epoch3、epoch5 小幅高于 B0。
- 两个 seed 的方向在第5 epoch 不一致，因此不建议继续扩大到 10/60 epoch，也不建议现在新增 Seed1234 或迁移 LTCC。

## 1. 固定配置与配对状态

保持不变：

- `CE + Triplet + clipBio + 0.1 × GPC`
- NLIST 权重 `0.05`
- listwise temperature `tau=0.1`
- 每 anchor 最多 3 个不同负 PID
- 每 epoch 重建 PRCC train-only 在线邻域
- GPU/PPU 分块检索 block size `512`
- EVA02、CE、Triplet、Bio、GPC-v1、sampler、optimizer/scheduler 和 PRCC 评估协议均未修改

旧实验目录中的第 3 epoch rank state 已核对：B0 Seed4321、B0 Seed3456、NLIST Seed4321 均缺失双 rank state；NLIST Seed3456 也不是完整双 rank state。因此没有直接使用 model-only 续训伪装成完整恢复，而是从旧 t1 模型重新创建了：

- 模型权重
- 新建 optimizer
- scheduler
- GradScaler
- 每 rank RNG

新的 common state：

- `/home/DIFFER/log_NDP/ndp_listwise_5epoch_prcc_20261028_135736/common_seed4321/`
- `/home/DIFFER/log_NDP/ndp_listwise_5epoch_prcc_20261028_135736/common_seed3456/`

每个 seed 的 B0/NLIST 均从同一份 common state 启动，raw 计划为 5 × 551 = 2,755 个 batch/update。

## 2. PRCC C→A 固定节点结果

百分比单位。官方评估节点为 continuation epoch 1、3、5；epoch 2、4 未对 B0/NLIST 同时执行官方评估，因此不填充推测值。

| Seed | Epoch | B0 Rank-1 | NLIST Rank-1 | Δ Rank-1 | B0 mAP | NLIST mAP | Δ mAP | B0 updates | NLIST updates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4321 | 1 | 61.953% | 62.179% | +0.226 pp | 56.863% | 56.763% | -0.100 pp | 551 | 551 |
| 4321 | 3 | 63.336% | 62.320% | -1.016 pp | 58.880% | 58.247% | -0.633 pp | 1653 | 1653 |
| 4321 | 5 | 64.352% | 63.618% | -0.734 pp | 60.554% | 60.425% | -0.129 pp | 2755 | 2755 |
| 3456 | 1 | 62.997% | 62.320% | -0.677 pp | 56.665% | 56.840% | +0.175 pp | 551 | 551 |
| 3456 | 3 | 63.251% | 63.703% | +0.452 pp | 59.295% | 59.467% | +0.172 pp | 1653 | 1653 |
| 3456 | 5 | 63.110% | 63.393% | +0.282 pp | 60.219% | 60.430% | +0.211 pp | 2755 | 2755 |

`mINP` 未被当前仓库的 PRCC C→A 评估器暴露，日志标记为 `not exposed by repository R1_mAP_eval`，因此未编造该指标。

## 3. 在线邻域、Loss 和梯度

每个 NLIST epoch 的在线图均为：

- 17,896 个有效 anchor
- 53,688 条负邻域关系
- 150 个训练 PID
- 每 anchor 3 个不同负 PID
- 正参考排除 anchor 自身

NLIST 累计统计如下：

| Seed | Epoch 1 NLIST Loss | Epoch 3 NLIST Loss | Epoch 5 NLIST Loss | Epoch 1→2 邻域变化率 | Epoch 2→3 | Epoch 4→5 |
|---:|---:|---:|---:|---:|---:|---:|
| 4321 | 0.122523 | 0.121484 | 0.119435 | 约 80.8% | 81.2% | 88.4% |
| 3456 | 0.123449 | 0.122479 | 0.120480 | 约 79.5% | 83.5% | 75.1% |

所有有效 anchor 都参与 listwise 项，因此 `active_relations / selected_relations = 100%` 是列表损失定义的结果，不等同于每条关系分别触发 hinge。

合并最终有效轨迹的前 20 个诊断 step（每个阶段均记录诊断，排除被 SIGKILL 运行中多出的未采用部分）：

| Seed | NLIST feature gradient mean | NLIST probe parameter gradient mean | main feature gradient mean | total model gradient mean |
|---:|---:|---:|---:|---:|
| 4321 | 1.112e-3 | 5.802e-3 | 2.021e-1 | 50.91 |
| 3456 | 1.095e-3 | 5.884e-3 | 1.841e-1 | 55.02 |

这确认 NLIST 在训练中真实反向传播到模型参数，但梯度存在不等价于获得稳定检索收益。

## 4. AMP、资源中断与恢复

- B0 Seed4321：2,755/2,755 effective updates，无 `found_inf`。
- B0 Seed3456：2,755/2,755 effective updates，无 `found_inf`。
- NLIST Seed4321：原始进程在完成 epoch3 后、epoch4 训练中收到 rank1 `SIGKILL`；从 `node_rawepoch03_update1653` 恢复 e4/e5，最终达到 2,755 effective updates。
- NLIST Seed3456：原始进程在完成 epoch3 后、epoch4 训练中收到 rank0 `SIGKILL`；从 `node_rawepoch03_update1653` 恢复 e4/e5，最终达到 2,755 effective updates。
- 两次 SIGKILL 均未伴随 NaN/Inf 或 AMP `found_inf` 记录，属于资源中断；失败进程日志和恢复目录均保留。

恢复目录：

- `nlist_seed4321_5raw_recovery_e4e5/`
- `nlist_seed3456_5raw_recovery_e4e5/`

被 SIGKILL 的原始运行在 epoch4 已产生少量未完成 batch 记录，但这些记录没有并入最终指标；最终轨迹严格采用 epoch3 完整 checkpoint 加恢复片段的 e4/e5。

最终模型 SHA256：

- B0 Seed4321：`e8169e29d046d00f80d8f2fddcbe1c27b874fd4c25a921d7f6272d9c24ee0389`
- NLIST Seed4321：`fb8f4d3116ffe424435c9f16d1763a6abfa0d04f6ef1d52cc035ba36e43102ed`
- B0 Seed3456：`07efc4103e2874fd3c8ed2313a6a322a9a179a54afd38966709d2bfcacbf6eaf`
- NLIST Seed3456：`b0bbb7faa311184e291ee04a5eccf3b2465d4d5a2ddcf2bc9060a21a7f85e8d6`

## 5. 代码与固定 GPC 校验

- 用户指定源提交：`926d8d2b2c0fa3a5a3708fce1314d92625febd3e`
- `scripts/ndp_v5_prcc_runner.py` SHA256：`1caf0ac10344381ed35401de0ad378b42c2e9e11bf76048eb05b6c172177d1d3`
- `tests/test_ndp_v5_b0_loss_contract.py` SHA256：`ff8cfca14a5ea6f86e95ee7e5a7e9ea63e1414115f95274e8299ed6f46b6aedc`
- 固定 GPC `/data/DIFFER/scripts/gpc_v2_train.py` SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`

前置单元测试结果：`24 passed, 7 warnings`；本轮训练未改动 GPC 源码或权重。

## 6. 决策

NLIST 五轮验证的跨 seed 结论为：`NO_STABLE_GAIN`。

Seed3456 的 epoch3/5 小幅正向不能抵消 Seed4321 在相同节点的下降。特别是 Seed4321 的 epoch3 Rank-1 `-1.016 pp`，说明 NLIST 的局部列表约束可能在部分训练轨迹中损害主识别结构。

因此：

- 不继续 10/60 epoch。
- 不新增 Seed1234 作为当前 NLIST 的默认扩展。
- 不扫描 tau、lambda 或负 PID 数量。
- 不把本轮结果宣称为 NDP 已有效。

下一步若继续研究，应转向证明 NDP 相对于真实 Batch-Hard Triplet 的增量监督关系及其局部副作用，而不是继续堆叠相近的 Margin/Listwise 损失变体。
