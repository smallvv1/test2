# NDP-GradientGate（NGATE）PRCC 实验结果

## 结论摘要

本轮完成了 PRCC 双 seed 的重新匹配 B0/NGATE 三 raw continuation epoch，并在 continuation 1、3 固定节点完成 C→A 评估。NGATE 真实计算了 `main_total` 与 NDP Loss 对当前 `feat[0]` 的梯度 cosine，并按 `cosine >= 0` 保留或拒绝 NDP。

结论：**NGATE 能够实际过滤一部分与主损失方向冲突的 NDP Batch，但没有消除 seed 依赖，也没有形成稳定的 Rank-1/mAP 提升。因此不建议继续扩展到 5 epoch。**

Seed1234 第 3 节点相对匹配 B0 为 `+0.734 pp Rank-1 / +0.518 pp mAP`；Seed4321 同节点为 `-0.903 pp / -1.381 pp`。这不是稳定收益。

## 实验身份与配对条件

- 用户指定提交：`70e62c916477f509378439abb29f429dfa9d7304`，在服务器 GitHub 归档中核对为 HEAD。
- 新日志根目录：`/home/DIFFER/log_NDP/ndp_gradient_gate_prcc_20261026_133314/`。
- 数据：PRCC train-only；训练关系和 Top5 参考图没有使用 query/gallery。
- 官方评估：PRCC C→A；评估器没有暴露 mINP。
- 固定 run-id：`ndp_prcc_baseline_fix_20261025_130457`。
- Seed1234 GPC phase：`ndp_v5_prcc_gpc_pair_ndp_prcc_baseline_fix_20261025_130457_seed1234`。
- Seed4321 GPC phase：`ndp_v5_prcc_gpc_pair_ndp_prcc_baseline_fix_20261025_130457_seed4321`。
- GPC 源 SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`。

上一轮 `common_seed1234/4321` 的完整 rank state 已被清理，本轮因此从上一轮相同 t1 model-only 权重重新建立 fresh optimizer/scheduler/GradScaler/RNG 共同状态，并重新运行 B0。NGATE 与 B0 在每个 seed 内从同一份新共同状态开始；旧 Top5 结果只作探索性参照，不能与本轮视为完全同一 optimizer 状态的严格消融。

共同状态：

- Seed1234 t1/model SHA256：`2cb7e822fdef9133aa250ac81fe2ab21295ad8af98f2e4c5811f8f56af4da98f`
- Seed4321 t1/model SHA256：`3155a876f0b5b067ccf4c3f79a67ff31a93e81f0210ff693dd844f632956861b`

## NGATE 实现

NGATE 复用上一轮 NDYN-TOP5 的 t0/t1 特征、Top5 图、风险排序、目标 Margin 和 `lambda_ndp=0.05`。仅新增：

```python
g_base = torch.autograd.grad(main_total, feat[0], retain_graph=True, allow_unused=True)[0]
g_ndp = torch.autograd.grad(ndp_loss, feat[0], retain_graph=True, allow_unused=True)[0]
cosine = cosine_similarity(g_base, g_ndp)
```

- cosine `>= 0`：保留 `0.05 * NDP`；
- cosine `< 0`：本 Batch 使用零 NDP contribution；
- 空梯度、零范数、非有限值：保守拒绝；
- 门控值全部 detached，不对门控求梯度；
- `main_total` 不变，不增加额外 optimizer step。

另外修复了一个 DDP 工程问题：若只有一个 rank 激活而另一个 rank 不激活，两个 rank 的 `autograd.grad` 调用序列会不同并可能卡住。现在先对局部激活标志做全局 `all_reduce`，任一 rank 激活时两个 rank 执行同序列梯度提取，但只有本地有效关系才应用 NDP。

## PRCC C→A 结果

Rank-1/mAP 以百分比报告。括号为 NGATE 相对本轮重新匹配 B0 的百分点差。旧 NDYN-TOP5 是上一轮结果，仅作参照。

| Seed | 节点 | B0 Rank-1 | B0 mAP | NGATE Rank-1 | NGATE mAP | NGATE−B0 | 旧 Top5 Rank-1/mAP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 62.687 | 56.648 | 62.518 | 56.697 | -0.169 / +0.049 pp | 62.885 / 56.881 |
| 1234 | 3 | 63.731 | 59.087 | 64.465 | 59.605 | +0.734 / +0.518 pp | 64.211 / 59.195 |
| 4321 | 1 | 62.264 | 56.900 | 61.897 | 56.702 | -0.367 / -0.198 pp | 61.840 / 56.736 |
| 4321 | 3 | 62.941 | 59.026 | 62.038 | 57.646 | -0.903 / -1.381 pp | 63.082 / 58.847 |

NGATE 在 Seed1234 第 3 节点优于 B0，但 Seed4321 第 3 节点 Rank-1/mAP 同时下降；两个 seed 没有一致方向。

## 梯度门控统计

以下为正式训练中全局 rank 汇总的 gate 统计。`gate evaluated` 是有正 NDP Hinge 并进行门控判断的 Batch 数；`negative rate` 是 cosine<0 的比例；`keep rate` 是 cosine>=0 的比例。

| Seed | 节点 | 活跃 Batch | 选中关系 | 正 Hinge 关系 | Gate evaluated | cosine 均值 | 负值比例 | 保留比例 | 有效 NDP Loss | 原始 NDP Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 109 | 253 | 137 | 109 | 0.034735 | 13.761% | 86.239% | 0.0004079 | 0.0004760 |
| 1234 | 3 | 228 | 509 | 270 | 228 | 0.039222 | 9.211% | 90.789% | 0.0004468 | 0.0004906 |
| 4321 | 1 | 108 | 188 | 122 | 108 | 0.035022 | 7.407% | 92.593% | 0.0004149 | 0.0004324 |
| 4321 | 3 | 168 | 377 | 193 | 168 | 0.034815 | 7.143% | 92.857% | 0.0003457 | 0.0003647 |

门控没有变成“全部关闭”：绝大多数有效 NDP Batch 的 cosine 为非负并被保留；负向冲突只占约 7.1%～13.8%。这说明 Seed4321 的性能退化不能简单归因于所有 NDP 梯度都与主损失冲突。

### 梯度与 AMP

- Seed1234 NGATE diagnostic 前 20 个 Batch：2 个非零 NDP feat 梯度步，最大 L2 `0.010881`。
- Seed4321 NGATE diagnostic 前 20 个 Batch：3 个非零 NDP feat 梯度步，最大 L2 `0.010917`。
- 两个 seed 的 B0 NDP feat/probe 梯度均为零。
- B0/NGATE 四个正式任务均无 AMP skip、无 NaN/Inf。
- 正式 NGATE 的 NDP Loss 与梯度确实影响了训练；不是只做离线 cosine 计算。

Purity@10 与 B0 接近：Seed1234 节点 1/3 为 `0.307923/0.305742`，NGATE 为 `0.307957/0.305725`；Seed4321 节点 1/3 为 `0.305167/0.304015`，NGATE 为 `0.305121/0.304047`。未见明显大幅 Purity 损害，但这不能证明没有细粒度邻域副作用。

## Smoke、失败与修复记录

单元测试和编译：

```bash
/usr/local/bin/python3.10 -m py_compile scripts/ndp_v5_prcc_runner.py
/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_v5_b0_loss_contract.py
```

结果：`16 passed`，`py_compile` 通过。单测覆盖 B0 零 NDP、NGATE 权重、正 cosine 保留、负 cosine 拒绝、缺失/零范数梯度拒绝。

Smoke：

- B0 Seed1234/4321：20 步 PASS，NDP contribution 严格为 0。
- NGATE Seed1234/4321：20 步最终重试 PASS，均捕获非零 NDP 梯度和正 cosine 保留事件。
- 初次 NGATE Smoke 曾因 DDP rank 激活不同导致首个命中 Batch 卡住，已终止并保留目录 `smoke_ngate_seed1234_20`；这是实现同步问题，不计入性能结果。
- DDP 同步修复后的第一次重试在 batch 5 暴露 `unsupported NDP arm: NGATE`，原因是目标 Margin 分支遗漏 NGATE；目录 `smoke_ngate_seed1234_20_retry` 保留 `failure.json`。补齐 NGATE 复用 Top5 目标后，`retry2` PASS。

四个正式 `run_summary.json` 均为 `PASS`，失败 Smoke 没有被伪装为成功结果。

## 实际运行命令

共同环境：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
```

共同状态重建：

```bash
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_make_common_warmstart.py --seed {1234|4321} \
  --t1-model /home/DIFFER/log_NDP/ndp_dynamic_top5_prcc_20261026_115538/common_seed{seed}/model.pth \
  --run-dir /home/DIFFER/log_NDP/ndp_gradient_gate_prcc_20261026_133314/common_seed{seed} \
  --epoch 4 --global-step 2204
```

正式训练共同形式：

```bash
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train \
  --seed {1234|4321} --arm {B0|NGATE} \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --run-dir <new_arm_dir> \
  --resume-checkpoint <common_seed_state> \
  --epochs 3 --eval-epochs 1,3 \
  --save-updates 551,1102,1653 --diagnostic-steps 20
```

NGATE 额外传入同 seed 的 t1 feature 和上一轮 Top5 graph；Smoke 使用相同入口的 `--steps 20 --save-updates 20 --diagnostic-steps 20`。

## SHA256

- `scripts/ndp_v5_prcc_runner.py`：`a4c24989b91ba456b1b5ba3369564309ba7229e4ee4db03135d87646176242d1`
- `tests/test_ndp_v5_b0_loss_contract.py`：`b1ea9a842c06b228f517f501a908842bcdac40f5657dafb7ff98772c69df8470`
- `scripts/ndp_make_common_warmstart.py`：`efaee6e3206ab715e1fa8e433669b6e772fe246d006309ed96d86fbbb1f14c42`
- 固定 GPC `/data/DIFFER/scripts/gpc_v2_train.py`：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`

## 最终决策

1. **NGATE 是否真实参与训练？是。** 两种子均出现有效 NDP Loss、NDP→EVA02 非零梯度、正 cosine 保留和负 cosine 拒绝。
2. **是否比 Top5 更稳定？否。** NGATE 仍然 Seed1234 正向、Seed4321 负向；负 cosine 过滤没有消除 seed 依赖。
3. **是否证明梯度冲突是主要原因？否。** 冲突比例只有约 7%～14%，但 Seed4321 仍明显退化。
4. **是否继续 5 epoch？NO-GO。** 没有两个 seed 同时改善 Rank-1/mAP，不满足继续扩展条件。
5. **下一步建议：** 停止继续堆叠 Margin 保护和梯度开关变体，转向验证 NDP 关系是否提供原始 Triplet 之外的真实增量监督；若继续，应优先做关系级监督覆盖/互补性分析，而不是继续调 cosine 或 Margin 阈值。

PPU 完整 EVA02 A/A 约 0.4% 的既有数值差异仍需保留。由于本轮 Seed4321 的下降幅度明显超过该数量级，不能将 NGATE 解释为已有效方法。
