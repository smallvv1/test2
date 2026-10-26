# NDP-Dynamic（NDYN）PRCC 实验结果

## 结论摘要

本轮 NDYN 已在 PRCC 上完成双 seed 的 B0/NDYN 配对三 raw continuation epoch。NDYN 确实使用了 t0→t1 的真实 B0 退化关系、产生了非零 NDP Loss，并通过当前 anchor 特征向 EVA02 传播梯度；但官方 C→A 指标没有在两个 seed 上同时改善。因此本轮结果为：**可运行、可激活、无稳定性能增益，NO-GO for 继续延长训练或迁移 LTCC**。

NDYN 的动态筛选并未人为扩大覆盖：Seed1234 只选出 92 个负 PID，Seed4321 只选出 36 个负 PID。该覆盖很低，说明“一个 raw epoch 内出现且仍正确的局部退化”在当前轨迹中是稀疏事件。

## 实验身份与完整性

- 远端 main 最新提交核对结果：`c3480b787950c5749d4b77261d4d806f372adf47`。
- 新日志根目录：[ndp_dynamic_prcc_20261026_095249](/home/DIFFER/log_NDP/ndp_dynamic_prcc_20261026_095249)。
- 数据：PRCC train-only，17,896 张训练图；官方评估为 C→A。训练关系没有使用 query/gallery。
- 固定 run-id：`ndp_prcc_baseline_fix_20261025_130457`。
- GPC phase 未包含 arm 名称：
  - Seed1234：`ndp_v5_prcc_gpc_pair_ndp_prcc_baseline_fix_20261025_130457_seed1234`
  - Seed4321：`ndp_v5_prcc_gpc_pair_ndp_prcc_baseline_fix_20261025_130457_seed4321`
- 固定 GPC `/data/DIFFER/scripts/gpc_v2_train.py` SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`。
- NDYN runner SHA256：`da769d1e794b6a88918f2f48b6a13c396338cbcc92f6ff54339dd2f9485c7770`。
- 动态参考脚本 SHA256：`ab08b958633eb90579cc2f615dc9e65d541a252314daac9e85a77836d614d641`。
- 单测文件 SHA256：`fe11ac0e2bb5b06097e5ef28812537542c62a55a368023311d59e8688082155b`。
- 单元测试：`11 passed`；`py_compile` 通过。
- 固定参数：`lambda_ndp=0.05`、动态退化阈值 `delta=0.005`、当前 Margin 上限 `0.05`、GPC 权重 `0.1`。

### 起点说明

两 seed 先从已有 common recreated 完整状态执行 1 个 B0 raw epoch，分别生成 t1 模型。随后发现 prefix 节点的 rank state 没有可靠保留：Seed1234 的两个 rank state 已缺失，不能把它们伪称为完整续训。因此 B0/NDYN 正式三轮统一采用各自 seed 的 t1 `model.pth` 做 **model-only warm-start**，并由相同配置新建 optimizer、scheduler、GradScaler 和 RNG。这保证了每个 seed 内 B0/NDYN 的共同模型权重起点，但不是从 t1 的历史 optimizer/RNG 无缝续训。

两臂使用相同 raw batch 计划、相同 seed 和相同 GPC phase。Seed4321 的 NDYN 在 raw batch 871 发生一次 AMP skip，所以为 1653 raw / 1652 effective updates；这组不能称为成功 optimizer update 逐步严格同轨迹。

## 动态参考图

筛选公式实际写入图构建脚本：

```text
M0 > 0
M1 > 0
M1 < M0 - 0.005
M1 < 0.05
```

每条关系使用 t0/t1 中相同的 anchor、同身份 positive indices 和负 PID endpoint；脚本验证两份特征的 `image_index` 完全相等。NDYN 训练时冻结 t1 特征，当前 batch anchor 保留梯度，目标为 `max(M1 - 0.005, 0)`。

| Seed | t0→t1 特征 | t0 图关系数 | 动态 anchor | 动态负 PID | t0 历史正确 PID | 动态覆盖（PID） |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | update 0→1 | 10,311 | 85 | 92 | 36,881 | 0.249% |
| 4321 | update 0→1 | 10,313 | 33 | 36 | 37,080 | 0.097% |

关键 artifact SHA256：

- Seed1234 t0 features：`7d9cd273420b73f059117eb3658f52b192c3e4be3509d071f654b6288355ebfb`
- Seed1234 t1 features：`b40c4e92804b64612bed07ede0013ce194dd452c284479547f7a27ac92e8e9a0`
- Seed1234 dynamic graph：`13522443be098f6621c1679a75a62dae1433352acb2dfc8d960d5f74f028d6bd`
- Seed4321 t0 features：`ca7cb2f672d7ff3c7aed2c8cb92d15d1768266fcc73c1783cf37e8a8c59fbcfb`
- Seed4321 t1 features：`ea800bf4585243b54d218bd549a315d4402a4ef5f61f839ba74cebe175807517`
- Seed4321 dynamic graph：`3c5aeb60ad7b08ba552f871fc5d18e404148df2711a8fa71361fab77158bd9e1`

## PRCC C→A 结果

表中 Rank-1、mAP 为百分比；括号是 NDYN 相对同 seed、同 continuation 节点 B0 的百分点差。

| Seed | continuation raw epoch | B0 Rank-1 | B0 mAP | NDYN Rank-1 | NDYN mAP | NDYN−B0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 62.235 | 56.517 | 62.715 | 56.957 | +0.480 / +0.439 pp |
| 1234 | 3 | 64.014 | 59.421 | 63.872 | 59.466 | −0.141 / +0.045 pp |
| 4321 | 1 | 61.925 | 56.671 | 61.360 | 56.706 | −0.565 / +0.035 pp |
| 4321 | 3 | 63.872 | 59.256 | 63.534 | 58.851 | −0.339 / −0.404 pp |

仓库评估器本轮未暴露 mINP，因此没有用其他协议替代 mINP。

## 训练侧统计

`ndp_selected_relations` 是训练中关系访问次数，`ndp_active_relations` 是产生正 hinge 的关系访问次数；不是动态图中的 unique PID 数。`ndp_active_rate` 是全局 rank 汇总后的活跃 Batch 比例。

| Seed | 节点 | raw/effective updates | 选中关系访问 | 正 hinge 关系访问 | 活跃 Batch | NDP Loss | Purity@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 551 / 551 | 92 | 22 | 1.543% | 0.00007099 | 0.30799 |
| 1234 | 3 | 1653 / 1653 | 183 | 35 | 1.588% | 0.00008146 | 0.30571 |
| 4321 | 1 | 551 / 551 | 35 | 8 | 0.635% | 0.00001775 | 0.30513 |
| 4321 | 3 | 1653 / 1652 | 69 | 15 | 0.544% | 0.00004004 | 0.30400 |

B0 两 seed 所有节点均 `ndp=0`、`lambda_ndp=0`，没有加载或计算 NDP 关系损失。NDYN 正式训练的 NDP Loss 和正 hinge 均非零；没有 NaN/Inf。四个正式 run 的 `run_summary.json` 均为 `PASS`。

### 真实梯度 Smoke

默认正式日志只在前 20 个 batch 记录梯度探针，而动态关系首个正 hinge 出现在更晚 batch；因此补做的 40-step NDYN diagnostic smoke 只用于梯度验证，不用于性能比较。Seed1234 batch 25 捕获到：

- NDP Loss：`0.00436348`。
- 当前 feat[0] NDP gradient L2：`0.00373807`。
- EVA02 probe parameter gradient L2：`0.0149799`。
- 主损失 feat[0] gradient L2：`0.156047`。
- 总模型 gradient L2：`32.1871`。

这证明 NDYN 当前 anchor 没有被 detach，NDP 梯度确实回到了 EVA02。B0 的控制 Smoke 中 NDP contribution 和 NDP gradient 均为零。

## 实际运行命令

公共环境：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
```

t0/t1 导出使用：

```bash
/usr/local/bin/python3.10 scripts/ndp_v5_prcc_runner.py --mode prepare \
  --seed {1234|4321} --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --checkpoint <model.pth> --source-update {0|1} \
  --feature <t0_or_t1_features.npz> --graph <temporary_k5_graph.json>
```

动态图构建使用：

```bash
/usr/local/bin/python3.10 scripts/ndp_dynamic_reference.py \
  --t0 <t0_features.npz> --t1 <t1_features.npz> \
  --graph-t0 <t0_graph.json> \
  --output-graph <ndp_dynamic_graph.json> \
  --output-manifest <ndp_dynamic_manifest.json>
```

正式训练命令的共同形式：

```bash
/usr/local/bin/python3.10 -m torch.distributed.run --standalone \
  --nproc_per_node=2 scripts/ndp_v5_prcc_runner.py --mode train \
  --seed {1234|4321} --arm {B0|NDYN} \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --run-dir <new_arm_dir> \
  --resume-model <seed_specific_t1_model.pth> \
  --resume-epoch 4 --resume-global-step 2204 \
  --epochs 3 --eval-epochs 1,3 --diagnostic-steps 20
```

NDYN 额外传入对应 seed 的 t1 feature 和 dynamic graph。Smoke 运行同一入口的 `--steps 20`；梯度验证运行 `--steps 40 --diagnostic-steps 40`。

## 决策

1. **NDYN 是否真正参与训练？** 是。动态图来自 B0 update0→update1，正式日志有非零选中关系、正 hinge、NDP Loss；独立 diagnostic smoke 捕获了非零 feat 和 EVA02 参数梯度。
2. **是否比 B0 稳定提升？** 否。Seed1234 第 1 节点改善、第三节点 Rank-1 下降；Seed4321 两个节点 Rank-1 均下降，第三节点 mAP 也下降。
3. **是否能证明动态筛选优于静态 Margin？** 不能。本轮只证明动态筛选改变了保护集合，未证明性能增量。
4. **是否损害局部邻域？** Purity@10 的聚合变化很小，但 Seed4321 NDYN 从 0.30515 降到 0.30400，且 Rank-1/mAP 同时下降；不能把小 Purity 差异解释为安全。当前数据不足以证明无邻域附带损害。
5. **是否值得继续 5 epoch 或迁移 LTCC？** NO-GO。没有两个 seed 同方向提升，且 Seed4321 有 AMP skip 造成有效更新不完全匹配；不建议直接延长或迁移。
6. **下一步建议：** 暂停继续堆叠历史 Margin 保护变体。若继续研究，应先用无 AMP skip 的严格短程配对重新验证动态关系是否超出数值噪声；在此之前不要扫描阈值、lambda 或跑 60 epoch。

已知完整 EVA02 PPU A/A 约 0.4% 梯度差异仍然适用。因而本轮 Seed1234 第三节点约 `−0.141 pp` 和 Seed4321 第三节点约 `−0.339/−0.404 pp` 只能作为探索性结果，不能据此宣称机制有效；但两 seed 没有一致正向趋势，已经足以阻止下一阶段长程扩展。
