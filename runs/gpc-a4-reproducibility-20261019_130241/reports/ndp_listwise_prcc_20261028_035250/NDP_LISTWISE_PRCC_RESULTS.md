# NDP-Listwise（NLIST）PRCC 双种子实验结果

实验日期：2026-10-28
实验目录：`/home/DIFFER/log_NDP/ndp_listwise_prcc_20261028_035250/`
用户指定源提交：`558d42be111b1ed9a79b406daf19205cf77615d6`

## 结论

NLIST 已完成 Seed4321、Seed3456 的 B0/NLIST 严格配对训练，各 3 个 raw continuation epoch，并在 continuation epoch 1、3 完成 PRCC C→A 评估。

- 两个 seed 在 epoch 1、3 的 Rank-1 和 mAP 均高于本轮匹配 B0。
- 提升幅度较小，但方向在两个 seed、两个固定节点上是一致的。
- NLIST 的 listwise loss 和梯度均正常；B0 的 NDP loss/梯度严格为零。
- 这组结果支持进行一次固定配置的 5 epoch 验证，但还不足以声称 NLIST 已经稳定有效，也不支持参数扫描或直接迁移 LTCC。

## 1. NLIST 实现

新增独立 arm：`NLIST`。

NLIST 完全复用 NONLINE 的每 epoch train-only fresh bank、GPU/PPU 分块在线邻居图和同身份正参考。每个 anchor 最多选择 3 个不同负 PID；正参考排除 anchor 自身；参考端点停止梯度，当前 batch anchor 保留反向传播。

唯一新增损失为：

```text
logits = concat([s_pos, s_neg_1, ..., s_neg_K]) / 0.1
L_NLIST = 0.1 × (logsumexp(logits) - logits[0])
L_total = CE + Triplet + clipBio + 0.1 × GPC + 0.05 × L_NLIST
```

空负邻居列表返回与当前 anchor 图相连的零损失。B0 仍然只执行固定的 Bio+GPC 主损失，不加载、计算或反向传播 NDP。

未修改 EVA02、CE、Triplet、clipBio、GPC-v1、GPC 正样本选择、GPC 权重 0.1、sampler、optimizer/scheduler 或 PRCC 评估器。

## 2. 测试与 Smoke

测试命令：

```bash
/usr/local/bin/python3.10 -m py_compile scripts/ndp_v5_prcc_runner.py
/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_v5_b0_loss_contract.py
```

结果：`24 passed, 7 warnings`，语法编译通过。

NLIST Smoke：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train --seed 4321 --arm NLIST \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --run-dir /home/DIFFER/log_NDP/ndp_listwise_prcc_20261028_035250/smoke_nlist_seed4321_20 \
  --resume-checkpoint /home/DIFFER/log_NDP/ndp_listwise_prcc_20261028_035250/common_seed4321 \
  --steps 20 --eval-epochs '' --save-updates 20 --diagnostic-steps 20 \
  --online-max-negative-pids 3 --online-block-size 512
```

Smoke 结果：20/20 optimizer steps 成功，无 NaN/Inf；首批 `L_NLIST=0.12408`，48 条候选关系全部进入 listwise，NDP feature gradient `1.129e-3`，NDP 参数探针 gradient `5.193e-3`，`found_inf=0`。

Fresh bank 提取前后 RNG digest 一致，在线图的同 PID 正参考排除了 anchor 自身，负 PID 均与 anchor PID 不同。

## 3. 配对状态与严格性

旧实验的 `common_seed4321/3456` 完整 rank state 已被清理，只剩模型权重。因此本轮没有把旧 NONLINE 结果冒充严格消融，而是从对应 t1 模型重新创建了共同状态：模型权重相同，optimizer、scheduler、GradScaler、RNG 同步由同一次 warm-start 生成。

本轮 B0 和 NLIST 在每个 seed 内从同一份完整 common state 分支，raw batch 计划均为 3 个 epoch × 551 = 1,653 个 raw batch index。故 B0/NLIST 是本轮严格配对；旧 NONLINE 仅作探索性参照。

共同起点模型 SHA256：

- Seed4321：`3155a876f0b5b067ccf4c3f79a67ff31a93e81f0210ff693dd844f632956861b`
- Seed3456：`44bed39e6659602354f40042840334cda19619d2628a4daaadd7ff6e2fa346d3`

固定 GPC 源 SHA256：

`/data/DIFFER/scripts/gpc_v2_train.py`
`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`

本地训练代码 SHA256：

- `scripts/ndp_v5_prcc_runner.py`: `1caf0ac10344381ed35401de0ad378b42c2e9e11bf76048eb05b6c172177d1d3`
- `tests/test_ndp_v5_b0_loss_contract.py`: `ff8cfca14a5ea6f86e95ee7e5a7e9ea63e1414115f95274e8299ed6f46b6aedc`

`/home/DIFFER` 没有 `.git` 元数据，因此用户提交号按用户提供值记录，未在本地伪造 git 校验。

## 4. PRCC C→A 结果

百分比单位。表中 B0/NLIST 均来自本轮严格配对训练。

| Seed | continuation epoch | B0 Rank-1 | NLIST Rank-1 | Δ Rank-1 | B0 mAP | NLIST mAP | Δ mAP | B0 effective updates | NLIST effective updates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4321 | 1 | 61.868% | 62.433% | +0.564 pp | 56.784% | 56.872% | +0.088 pp | 551 | 551 |
| 4321 | 3 | 62.546% | 63.054% | +0.508 pp | 58.627% | 58.796% | +0.169 pp | 1653 | 1653 |
| 3456 | 1 | 63.167% | 63.506% | +0.339 pp | 57.282% | 57.353% | +0.071 pp | 551 | 551 |
| 3456 | 3 | 64.070% | 64.465% | +0.395 pp | 59.320% | 59.597% | +0.276 pp | 1653 | 1652 |

主结果来自各 arm 的 `metrics_per_epoch.csv` 字段 `cc_rank1`、`cc_mAP`。general protocol 输出未混入上表。

当前仓库的 PRCC C→A 评估器没有暴露 mINP（日志为 `not exposed by repository R1_mAP_eval`），因此没有填充推测值。

## 5. NLIST 激活、Loss 与梯度

在线图每个 epoch 均为：17,896 个有效 anchor、53,688 条关系、150 个 PID、每 anchor 3 个不同负 PID；GPU/PPU 相似度检索使用 block size 512，正参考排除 anchor 自身。

| Seed | epoch 1 NLIST Loss | epoch 3 NLIST Loss | epoch 1 选中关系 | epoch 3 累计选中关系 | epoch 1→2 邻域变化率 | epoch 2→3 邻域变化率 |
|---:|---:|---:|---:|---:|---:|---:|
| 4321 | 0.122532 | 0.121403 | 52,896 | 105,792 | 80.79% | 76.59% |
| 3456 | 0.123452 | 0.122548 | 52,896 | 105,792 | 79.51% | 73.63% |

NLIST 的 listwise loss 对有负 PID 的有效 anchor 始终为正，因此本轮 `active_relations` 等于选中的关系槽位；这不代表每一条关系都单独触发 hinge，而是表示该关系参与了同一 anchor 的列表排序项。

前 20 个诊断 step（两 rank 诊断记录汇总）的量级：

| Seed | NDP feature gradient mean | NDP probe parameter gradient mean | main feature gradient mean | total model gradient mean |
|---:|---:|---:|---:|---:|
| 4321 | 1.085e-3 | 5.107e-3 | 2.028e-1 | 55.07 |
| 3456 | 1.087e-3 | 5.491e-3 | 2.001e-1 | 50.31 |

这证明 NLIST 真实参与了 EVA02 反向传播，但梯度存在不等于方法带来因果性能提升。

## 6. 与 NONLINE 的探索性比较

旧 NONLINE 结果位于：`/home/DIFFER/log_NDP/ndp_online_prcc_20261028_120000/`。由于旧 common rank state 已清理，本节不作为严格消融，只用于损失/激活规模参照。

- NONLINE 的 NDP Loss 约为 `0.0028–0.0036`，NDP feature gradient 诊断均值约 `0.90e-3–0.96e-3`。
- NLIST 的 Loss 约为 `0.121–0.123`，NDP feature gradient 约 `1.08e-3`。
- NONLINE 的关系激活率约为 `34.7%–44.3%`；NLIST 的所有有效 anchor 都形成 listwise loss，关系槽位参与率为 100%。

NLIST Loss 数值更大是损失定义改变的直接结果，不能仅凭 Loss 大小判断监督更强或更有效。当前严格比较应以本轮 B0/NLIST 检索结果为准。

## 7. AMP 与运行状态

- Seed4321 B0/NLIST：1,653/1,653 effective updates，无 AMP skip。
- Seed3456 B0：1,653/1,653 effective updates，无 AMP skip。
- Seed3456 NLIST：1,652/1,653 effective updates；rank 0 在 batch index 1285、epoch 7、global step 3489 出现一次 `found_inf=1`，GradScaler 从 512 降为 256。
- 四个正式任务均正常退出，无 NaN/Inf 终止。

因此 Seed3456 epoch3 的 NLIST 对比存在 1 个 effective update 差异，已在结果表中如实保留。

## 8. 下一步决策

本轮 NLIST 在两个 seed、两个固定节点上都高于严格匹配 B0，属于目前值得继续验证的正向信号；但增益较小，且仍可能受 PPU/AMP 非确定性影响。

建议：只进行一次相同配置的 5 raw continuation epoch 配对验证，固定 `tau=0.1`、`lambda_ndp=0.05`、每 anchor 3 个负 PID，不扫描超参数。若 5 epoch 的两个 seed 仍同时保持 Rank-1/mAP 正向，再考虑独立 seed；若收益消失，则停止继续堆叠 listwise/Margin 变体，转向验证 NDP 是否提供原始 Batch-Hard Triplet 之外的真实增量关系监督。
