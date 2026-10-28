# NDP-Online（NONLINE）PRCC 双种子实验结果

日期：2026-10-28
实验目录：`/home/DIFFER/log_NDP/ndp_online_prcc_20261028_120000/`
研究基线：`CE + Triplet + clipBio + 0.1 × GPC`
NONLINE：`L_base + 0.05 × L_NONLINE`

## 结论先行

NONLINE 已完成 Seed4321、Seed3456 的 B0/NONLINE 配对训练，各 3 个 raw continuation epoch，并在 continuation epoch 1、3 完成 PRCC C→A 固定评估。

- Seed4321：NONLINE 在两个节点均优于 B0，epoch 3 为 Rank-1 `+0.762 pp`、mAP `+0.035 pp`。
- Seed3456：NONLINE 在两个节点均低于 B0，epoch 3 为 Rank-1 `-0.339 pp`、mAP `-0.435 pp`。
- 因此没有得到跨 seed 的稳定收益证据，不满足继续扩展到 5 epoch 或迁移 LTCC 的门槛。
- 本轮在线邻域确实产生了大量有效关系和非零梯度，但“关系覆盖增加”没有转化为稳定检索提升。

`mINP` 未被当前仓库的 PRCC C→A 评估器暴露（日志明确为 `not exposed by repository R1_mAP_eval`），本报告不填充推测值。

## 1. 实现内容

新增训练 arm：`NONLINE`。

每个 continuation epoch 开始时，对 PRCC train-only 的 17,896 张图像提取归一化 `feat[0]`，然后在 GPU/PPU 上以 block size 512 分块计算当前特征与全体训练特征的相似度。相同 PID 被屏蔽后，每个 anchor 选择最多 3 个不同负 PID，每个 PID 保留一个最相似负样本。

正身份参考由同 PID 的其他训练图像均值构造，明确排除了 anchor 自身。在线图只使用训练集，不使用 query/gallery。

当前 batch 的 anchor 特征保持计算图；正负参考端点停止梯度。每个 anchor 对最多 3 个负 PID 的约束取平均，再对有效 anchor 取平均：

```text
s_pos = cosine(anchor, positive_reference)
s_neg = cosine(anchor, negative_reference)
L_NONLINE = mean(ReLU(0.02 - (s_pos - s_neg)))
L_total = L_base + 0.05 * L_NONLINE
```

B0 分支不构造、不计算、不加入 NDP Loss。

Fresh-bank 提取前后保存并恢复 Python、NumPy、PyTorch CPU 及 PPU/CUDA RNG 状态；两个正式 NONLINE 运行的每个 fresh-bank JSON 均记录 `rng_state_preserved: true`，前后 RNG digest 相同。

## 2. 代码与固定基线校验

- 用户指定源提交：`900108a10a933e84d2e032b575fff407e148c541`
- `/home/DIFFER` 本地目录没有 `.git` 元数据，因此无法在服务器上用 `git rev-parse` 再次验证该提交；本报告使用用户提供的提交标识。
- 实际训练代码 SHA256：

  `scripts/ndp_v5_prcc_runner.py`
  `e248ce058761e4dece5f5c0bef10fe1ca4ecea355b305dcd73a83b2b71ccbfd0`

- 新增/修改测试 SHA256：

  `tests/test_ndp_v5_b0_loss_contract.py`
  `2b6bd73ea23b7cb4cc8d8f95a88ecb32db625d9ad19479602e0253144dcd69d5`

- 固定 GPC 源 SHA256：

  `/data/DIFFER/scripts/gpc_v2_train.py`
  `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`

测试命令：

```bash
/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_v5_b0_loss_contract.py
/usr/local/bin/python3.10 -m py_compile scripts/ndp_v5_prcc_runner.py
```

结果：`21 passed, 7 warnings`；语法编译通过。

## 3. Smoke Test

Smoke 输出目录：

- B0：`smoke_b0_seed4321_20`
- NONLINE 首次失败（保留）：`smoke_nonline_seed4321_20`
- NONLINE 修复后：`smoke_nonline_seed4321_20_retry`

首次 NONLINE smoke 出现一次 `CUDA error: device-side assert`。原因是在线图构建后，组合参考 bank 被后续无条件赋值覆盖，导致正参考索引越界；这是训练入口的索引管理错误，不是数值训练失败。修复后未覆盖失败目录，retry smoke 通过：

- 在线图：17,896 个有效 anchor、53,688 条关系、150 个 PID、每 anchor 3 个不同负 PID。
- 20/20 optimizer steps 成功，`found_inf=0`。
- NDP Loss 有限；NDP feature gradient 最大约 `1.218e-3`，参数探针梯度最大约 `4.049e-2`。
- B0 smoke 的 NDP 关系数和 NDP 梯度均为 0。
- RNG 前后 digest 一致。

## 4. 配对训练命令

两个 arm 均使用同一个 seed-specific common state，GPC 的 `--run-id` 固定为 `ndp_prcc_baseline_fix_20261025_130457`，没有将 arm 名称加入 GPC 哈希上下文。

实际命令模板如下，`<seed>` 和 `--arm` 分别替换为四个正式任务：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py \
  --mode train --seed <seed> --arm <B0|NONLINE> \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --run-dir /home/DIFFER/log_NDP/ndp_online_prcc_20261028_120000/<arm>_seed<seed>_3raw \
  --resume-checkpoint /home/DIFFER/log_NDP/ndp_online_prcc_20261028_120000/common_seed<seed> \
  --epochs 3 --eval-epochs 1,3 --save-updates 551,1653 \
  --diagnostic-steps 20 \
  --online-max-negative-pids 3 --online-block-size 512
```

共同起点：

- Seed4321：`common_seed4321`
- Seed3456：`common_seed3456`

两份 common state 均包含模型、optimizer、scheduler、GradScaler、RNG；B0/NONLINE 在每个 seed 内从同一份状态分支。

## 5. PRCC C→A 结果

百分比单位；epoch 是 continuation epoch，不是绝对训练 epoch。

| Seed | Continuation epoch | B0 Rank-1 | NONLINE Rank-1 | Δ Rank-1 | B0 mAP | NONLINE mAP | Δ mAP | B0 有效更新 | NONLINE 有效更新 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4321 | 1 | 61.445% | 62.094% | +0.649 pp | 56.400% | 56.946% | +0.546 pp | 551 | 550 |
| 4321 | 3 | 62.602% | 63.364% | +0.762 pp | 58.890% | 58.924% | +0.035 pp | 1653 | 1652 |
| 3456 | 1 | 63.167% | 62.546% | -0.621 pp | 57.237% | 56.583% | -0.654 pp | 551 | 550 |
| 3456 | 3 | 64.126% | 63.788% | -0.339 pp | 59.648% | 59.213% | -0.435 pp | 1653 | 1652 |

主结果来源：各 arm 的 `metrics_per_epoch.csv` 字段 `cc_rank1`、`cc_mAP`。评估输出中的 general protocol 结果没有混入上表。

## 6. 在线邻域与 NDP 激活

每个 seed、每个 epoch 的在线图规模固定为：

- 17,896/17,896 有效 anchor。
- 53,688 条关系，即每个 anchor 3 条不同负 PID。
- 150 个训练 PID。
- 相似度检索设备记录为 `cuda:0`，block size 512。
- `positive_reference_excludes_anchor=true`。

在线图随 epoch 变化明显：

| Seed | epoch 1→2 邻域变化率 | epoch 2→3 邻域变化率 | epoch 1 NDP 关系激活率 | epoch 3 累计 NDP 关系激活率 | epoch 3 平均 NDP Loss |
|---:|---:|---:|---:|---:|---:|
| 4321 | 85.53% | 79.65% | 39.80% | 34.68% | 0.0028373 |
| 3456 | 89.72% | 85.70% | 44.32% | 39.39% | 0.0031817 |

这里的“关系激活率”是 `ndp_active_relations / ndp_selected_relations`；NDP active batch 比例在两个 NONLINE 运行中接近 100%，因此没有把 active batch 比例误当成 hinge 关系激活率。

累计统计：

- Seed4321：epoch 1 选中 52,896、激活 21,052；epoch 3 累计选中 105,792、激活 36,687。
- Seed3456：epoch 1 选中 52,896、激活 23,441；epoch 3 累计选中 105,792、激活 41,668。

前 20 个诊断 step（两 rank 汇总的诊断记录）中：

- Seed4321：NDP feature gradient mean `8.96e-4`，NDP 参数探针 gradient mean `3.50e-3`，主损失 feature gradient mean `2.04e-1`。
- Seed3456：NDP feature gradient mean `9.55e-4`，NDP 参数探针 gradient mean `5.10e-3`，主损失 feature gradient mean `2.04e-1`。

这些值证明 NONLINE 在训练中确实反向传播；它们不是全程梯度均值，而是 `--diagnostic-steps 20` 范围内的诊断统计。

## 7. AMP、raw batch 与 effective update

四组正式任务均计划 3 个 raw continuation epoch、每 epoch 551 个 raw batches，共 1,653 个 raw batch index。

- 两个 B0：1,653/1,653 effective updates，无 AMP skip 记录。
- 两个 NONLINE：1,652/1,653 effective updates，各出现 1 次 AMP skip。
  - Seed4321：rank 0，batch index 101，epoch 5，global step 2305，scale 从 512 降为 256。
  - Seed3456：rank 0，batch index 63，epoch 5，global step 2267，scale 从 512 降为 256。
- 没有 NaN/Inf 终止，四个正式进程均正常退出。

因此这是一组相同 raw 计划、但 NONLINE 各少 1 个 successful optimizer update 的配对实验；结果不能描述为 effective-update 完全相同，差异已在表格中保留。

## 8. 决策

本轮结论为：`NONLINE = 未获得稳定收益`。

Seed4321 的小幅收益与 Seed3456 的对称下降同时存在，且 mAP 在 Seed3456 的两个节点均下降。在线危险邻域确实比固定历史图产生了更高覆盖和明显的 epoch-to-epoch 邻域变化，但当前的固定 `0.02 - margin` 约束仍可能与主损失产生有害的局部竞争，或者在线近邻本身包含不适合保护的关系。

按预注册门禁：

- 不继续到 5 epoch。
- 不迁移 LTCC 作为下一步主实验。
- 不继续扫描 epsilon、lambda、负 PID 数量或其他相似 Margin 变体。
- 后续若继续研究，应直接检查在线关系相对于真实 Batch-Hard Triplet 的增量监督价值和关系质量，而不是再堆叠另一种静态/反应式 Hinge。

历史 NDP-Fresh/NCOMP 结果仅作为探索背景；本轮 NONLINE 的正式证据本身不足以支持 NDP 已有效。
