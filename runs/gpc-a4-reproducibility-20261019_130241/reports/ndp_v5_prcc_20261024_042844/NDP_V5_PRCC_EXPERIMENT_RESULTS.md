# NDP-V5 PRCC 实验结果

日期：2026-10-24
实验目录：`/home/DIFFER/log_NDP/ndp_v5_prcc_20261024_042844/`
远端 `main` 实际核对提交：`8bf1e3a972fbc6fa826a977bd570d82a9b341b67`

## 结论摘要

PRCC 双 seed 的 B0/N5 配对训练均完成，每组达到 1653 个有效 optimizer step，NDP-V5 在训练中确实产生了非零特征梯度并更新了 EVA02 参数。

但 PRCC 的 Rank-1 结果不稳定：Seed4321 提升 1.98 pp，Seed1234 下降 1.21 pp；mAP 两个 seed 均提高，但幅度分别为 0.19 pp 和 0.63 pp。因此本实验不能证明 NDP-V5 在 PRCC 上稳定优于固定 DIFFER+GPC，也不建议直接进入完整长程训练。

## 1. 实验协议与数据来源

- 基线：`CE + Triplet + clipBio + 0.1 * GPC-v1`。
- N5：在上述基础上加入 `0.05 * L_NDP-V5`。
- NDP：`epsilon=0.02`，每个负 PID 独立计算历史/当前 Margin，anchor 内负 PID 损失平均，再对有效 anchor 平均。
- 主干与基础损失：沿用 PRCC Bio-only 模型；Reverse 为 OFF，没有混入额外 Reverse 模块。
- GPC：调用固定 `/data/DIFFER/scripts/gpc_v2_train.py` 的 `selected_v2` matched-budget 选择逻辑；该文件未修改。
- PRCC 训练集：17,896 张图像、150 个训练 PID、3 个 camera、551 个 batch/epoch，batch size 16，`NUM_INSTANCES=4`。
- 参考库：只使用 PRCC train-only update0 特征；没有读取 query/gallery，也没有用未来 checkpoint 选择关系。
- 官方主指标：PRCC C→A 换装协议。仓库评估器未实现 mINP，故 mINP 记为 N/A。

服务器没有找到可可靠恢复的 PRCC Bio+GPC E60 完整状态。本轮使用已验证的 PRCC Bio-only common initialization 和新建、匹配的 optimizer；这不是历史 E60 continuation，已在各 run 的 `training_config.json` 中明确记录。B0/N5 使用相同 common-init、seed、采样器、学习率调度和有效更新预算。

参考库摘要：

- 特征：`[17896, 1024]`。
- 合法 anchor：17,896。
- 含历史正确 PID 关系的 graph row：2,351。
- update0 历史正确负 PID 关系：6,892。
- 参考文件：`reference/prcc_update0_features.npz`、`reference/prcc_v5_k5_graph.json`。

## 2. Smoke Test

N5 Seed1234 的 20 步 Smoke Test PASS：

| 项目 | 实测 |
|---|---:|
| 有效 optimizer updates | 20 |
| 原始 batch index | 21 |
| optimizer step 执行 | 是 |
| NDP active batches（双 rank 汇总） | 14 |
| NDP selected relations（双 rank 汇总） | 215 |
| NDP active relations（双 rank 汇总） | 20 |
| NDP active rate | 33.33% |
| NDP mean loss | 0.0005186 |
| NDP feature gradient norm，20 个有限诊断值均值 | 0.0009788 |
| NDP feature gradient norm，最大值 | 0.0050084 |
| block0 probe parameter gradient norm，最大值 | 0.2107121 |
| 主损失 feature gradient norm均值 | 0.4665226 |
| NaN/Inf loss | 无 |

Smoke 中发生 1 次 AMP skip（batch 15，scale 512→256），之后继续完成 20 个有效更新；这被记录为数值事件，不是静默丢步。

## 3. PRCC C→A 官方评估

以下均为固定 update1653 节点的结果。Seed1234 在第一个 epoch 中有一次 AMP skip，因此为达到相同有效更新数多执行了一个 replacement batch；B0/N5 两臂均按同一 runner 处理，未改变目标预算。

| Seed | 模型 | 有效更新 | Rank-1 | mAP | mINP |
|---:|---|---:|---:|---:|---|
| 1234 | B0 | 1653 | 58.51% | 51.36% | N/A |
| 1234 | N5 | 1653 | 57.30% | 51.55% | N/A |
| 4321 | B0 | 1653 | 57.55% | 51.70% | N/A |
| 4321 | N5 | 1653 | 59.53% | 52.32% | N/A |

相对 B0：

| Seed | Rank-1 差值 | mAP 差值 |
|---:|---:|---:|
| 1234 | -1.21 pp | +0.19 pp |
| 4321 | +1.98 pp | +0.63 pp |
| 两 seed 算术平均 | +0.38 pp | +0.41 pp |

平均值不能替代 seed 一致性判断；Rank-1 的方向相反，故不能称为稳定提升。

同衣物 sanity 指标没有用于选择配置：Seed1234 B0/N5 为 99.87%/96.62% 与 99.90%/96.99%（Rank-1/mAP），Seed4321 B0/N5 为 99.92%/96.91% 与 99.85%/96.77%。

## 4. NDP 实际激活与局部诊断

下表取 3 个正常训练周期的聚合训练统计；Seed1234 的最后一行 replacement batch 单独保留在 CSV，不用它替换主统计。

| Seed | NDP active batches | active rate | selected relations | active relations | 当前错误 PID | 当前正确 PID | 平均当前 Margin | Purity@10 | NDP mean loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 883 | 26.71% | 20,448 | 2,000 | 5,818 | 14,630 | 0.009534 | 0.30253 | 0.0006978 |
| 4321 | 864 | 26.13% | 20,301 | 1,978 | 5,844 | 14,457 | 0.009377 | 0.30495 | 0.0007042 |

NDP 不是空跑：参考端保持冻结，当前 batch 的 `feat[0]` 保留计算图；Smoke 中 block0 参数 probe 的非零梯度证明梯度回到了 EVA02。正式训练的 NDP 梯度相对主损失较小，但不是零：

- Seed1234：NDP feature gradient norm 的诊断均值约 0.000932，主 feature gradient norm 约 0.462。
- Seed4321：NDP feature gradient norm 的诊断均值约 0.000420，主 feature gradient norm 约 0.468。

局部 Purity@10 与 B0 接近：Seed1234 为 B0 0.30248、N5 0.30253；Seed4321 为 B0 0.30503、N5 0.30495。当前聚合指标没有显示明显的邻域 Purity 损害，但这不是完整的逐邻域因果审计，不能据此证明不存在局部副作用。

## 5. AMP、checkpoint 与可复现记录

- B0 Seed1234：1653 有效更新，batch 351 发生一次 AMP skip，保存 update551/1102/1653。
- N5 Seed1234：1653 有效更新，batch 505 发生一次 AMP skip，保存 update551/1102/1653。
- B0 Seed4321：无 AMP skip，1653 有效更新，保存 update551/1102/1653。
- N5 Seed4321：无 AMP skip，1653 有效更新，保存 update551/1102/1653。
- 四个正式 run 的 `run_summary.json` 均为 `status=PASS`，模型、optimizer、scheduler、GradScaler、RNG 均写入 rank state。
- Smoke、四个正式 run 和 reference artifacts 均写入新的时间戳目录，没有覆盖历史实验。

实际执行入口示例：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone \
  --nproc_per_node=2 scripts/ndp_v5_prcc_runner.py \
  --mode train --seed 1234 --arm N5 \
  --run-id ndp_v5_prcc_20261024_042844 \
  --run-dir /home/DIFFER/log_NDP/ndp_v5_prcc_20261024_042844/n5_seed1234_3e \
  --feature /home/DIFFER/log_NDP/ndp_v5_prcc_20261024_042844/reference/prcc_update0_features.npz \
  --graph /home/DIFFER/log_NDP/ndp_v5_prcc_20261024_042844/reference/prcc_v5_k5_graph.json \
  --epochs 3 --save-updates 551,1102,1653 --diagnostic-steps 20
```

代码与数据哈希：

- 新 runner `scripts/ndp_v5_prcc_runner.py`：`9e21fc2d00a4b9865964fac2a918459ab8b6578044cf549289af23cae5738f8a`。
- 固定 GPC `/data/DIFFER/scripts/gpc_v2_train.py`：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`。
- PRCC common init：`db5c71389b37a859ec82b3e529c1cd7149c21a85df4a1b34f9650c9f6bd9378f`。
- update0 reference features：`5bff5f6eb194b1dae71e57cd833ae58e09c569312fb9c0f53ab83d2e77d2997e`。
- K=5 reference graph：`30c08494f15bff44ceccdbfec4a014ebb601c6abf1cc4334262d3fc3ba14f2a4`。
- `python3.10 -m py_compile scripts/ndp_v5_prcc_runner.py`：PASS。

最终 update1653 模型 checkpoint SHA256：B0/1234 `2e63a254db95221580011d0b49ba4f00620f2c89d3ecfa2de4b92e39dada9bba`；N5/1234 `a59d8397df56ae389cf2878ebc73336f7d7ad4101c3f24bbfbf8c4418eee9524`。B0/4321 与 N5/4321 的 checkpoint manifest 也已写入各自 run 目录。

此前完整 EVA02 PPU A/A 约 0.4% 的非确定性仍然适用。本轮未重复算子级审计；小幅 mAP 变化必须结合该限制解释。

## 6. 决策

1. **NDP-V5 是否在 PRCC 真正激活？** 是。20 步 Smoke 和两 seed 正式训练均有非零 NDP loss、选中关系和回传到 EVA02 的梯度。
2. **是否比固定 DIFFER+GPC 稳定提升？** 否。Seed4321 的 Rank-1/mAP 提升，Seed1234 的 Rank-1 下降；当前证据是 seed-dependent。
3. **是否比 LTCC 更稳定？** 不能确认。PRCC Rank-1 方向不一致，不能视为跨数据集稳定迁移。
4. **是否有明显邻域身份损害？** 聚合 Purity@10 未见明显下降，但 Seed1234 的 Rank-1 退化说明不能排除更细粒度关系损害或训练轨迹差异。
5. **是否值得直接做 PRCC 长程/60 epoch？** NO-GO。当前没有双 seed 一致的 Rank-1 提升，也没有历史 E60 恢复条件。
6. **下一步是否值得继续？** 可做少量独立 PRCC seed 复验，前提是继续使用完全相同的 B0/N5 配对协议；在获得一致方向前，不建议修改 NDP loss、增加复杂门控或开始长程正式训练。

最终判断：本轮证明了 NDP-V5 在 PRCC 上可运行、可激活、可更新，但没有证明其相对于固定 DIFFER+GPC 的稳定性能增益。
