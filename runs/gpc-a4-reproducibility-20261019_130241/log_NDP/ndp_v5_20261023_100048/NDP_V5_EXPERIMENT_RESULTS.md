# NDP-V5 实际训练与 LTCC 结果

## 1. 结论

NDP-V5 已完成两个 seed、1788 个有效更新，并完成 update596、1192、1788 的固定节点评估。

V5 将原来的 `max(5 个负 PID)` 改为每个负 PID 独立的历史 Margin 保护，再在 anchor 内平均。结果显示：

- update1788：两个 seed 的 Rank-1、mAP、mINP 均超过同节点 B0。
- update1192：两个 seed 的 mAP 和 mINP 超过 B0，但 Seed4321 的 Rank-1 仍低于 B0。
- 相比 N2，V5 的 mAP 跨 seed 稳定性明显改善；Rank-1 在 3 epoch 节点也实现两个 seed 同时正增益。
- train-only 局部指标中，两个 seed 的 update1788 正确关系数、平均 Margin 和 Purity@10 均高于各自 update0 参考；但本轮没有为 B0/N2/N3/N4 在相同节点重新导出全部 train-only bank，因此不能把局部改善完全归因于 V5。

建议：`NDP-V5 可以进入一个严格匹配的 10 epoch 双 seed pilot`，暂不扩展 60 epoch，也不叠加 Hubness、双侧权重或新损失。

## 2. 固定协议

基础损失保持：

`L_base = CE + Triplet + clipBio + 0.1 * GPC-v1`

没有修改 EVA02、feat[0]、CE、Triplet、clipBio、GPC matched-budget 正样本选择、GPC Loss、优化器、采样器或 LTCC 评估协议。

V5 的关系定义为：

```text
M_ref(i,c) = mean(ref_positive_similarity) - ref_negative_similarity(i,c)
M_cur(i,c) = mean(current_positive_similarity) - current_negative_similarity(i,c)
L(i,c) = ReLU(M_ref(i,c) - M_cur(i,c) - 0.02)
```

仅使用 update0 历史 Margin 大于零的负 PID。一个 anchor 内先平均合法负 PID 的 hinge，再对 Batch 中有效 anchor 平均。参考特征 stop-gradient，当前 anchor 仍向 EVA02 反向传播。V5 使用固定 `lambda_ndp=0.05`。

候选图为同一份 2048-anchor、K=5 图；Seed1234 和 Seed4321 分别有 1947 个至少包含一个历史正确负 PID 的 anchor relation，合法历史负 PID 关系数分别为 9377 和 9297。

用户给定最新 GitHub commit：`b520a6c3baeef06082069ee193d91449e239d9ac`。本地 `/home/DIFFER` 不是 git worktree；固定工程 `/data/DIFFER` 的本地 HEAD 为 `573b0502a231568d509fe3a463664c1a3c42e258`。

## 3. 实际命令

Smoke 和完整训练使用同一入口，仅替换 seed 和输出目录：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_e60_minimal_pair_runner.py --mode full --arm N5 --seed 1234 \
  --max-effective-updates 1788 --save-updates 596,1192,1788 \
  --initial-lr 1e-6 \
  --reference-feature /home/DIFFER/outputs/ndp_true_multipid_20261021_103952/full_features/seed1234/update0_full_features_seed1234.npz \
  --graph /home/DIFFER/outputs/ndp_v2_20261023_061358/reference_graph/seed1234/multi_pid_reference_graph_v2.json \
  --run-dir /home/DIFFER/log_NDP/ndp_v5_20261023_100048/n5_seed1234
```

固定 LTCC 评估使用：

```bash
/usr/local/bin/python3.10 scripts/ndp_official_ltcc_nodes.py \
  --seed 1234 --config /data/DIFFER/diagnostics/reconstructed_ltcc_eval_config.yml \
  --device-index 0 --node N5 1192 <node_update1192/model.pth> \
  --node N5 1788 <node_update1788/model.pth>
```

局部指标使用 9576 张 LTCC train-only 图像导出 feat[0]，没有 optimizer.step，也没有使用官方 query/gallery：

```bash
/usr/local/bin/python3.10 scripts/ndp_v2_feature_export.py --seed 1234 --arm N5 --update 1788 ...
/usr/local/bin/python3.10 scripts/ndp_v5_local_metrics.py --seed 1234 --features <export.npz> ...
```

## 4. B0/N2/N3/N4/N5 固定节点对照

每个 continuation epoch 为 596 个有效更新；因此 update596、update1788 分别对应 1 epoch 和 3 epoch。N3 的 update596 与 N2 的 update596 是同一个恢复节点，N3 从该节点之后关闭 NDP。

数值为 Rank-1 / mAP / mINP；括号为相同 seed、相同 update 相对 B0 的 Rank-1 差值和 mAP 差值，单位为百分点。

### 1 epoch：update596

| Seed | 模型 | Rank-1 | mAP | mINP |
|---:|---|---:|---:|---:|
| 1234 | B0 | 0.533163 | 0.318329 | 0.086222 |
| 1234 | N2 | 0.543367 (+1.020 / +0.696) | 0.325284 | 0.089718 |
| 1234 | N3 | 0.543367 (+1.020 / +0.696) | 0.325284 | 0.089718 |
| 1234 | N4 | 0.525510 (-0.765 / +0.182) | 0.320149 | 0.086235 |
| 1234 | N5 | 0.520408 (-1.276 / -0.103) | 0.317296 | 0.086863 |
| 4321 | B0 | 0.512755 | 0.310752 | 0.081075 |
| 4321 | N2 | 0.517857 (+0.510 / +0.421) | 0.314960 | 0.083284 |
| 4321 | N3 | 0.517857 (+0.510 / +0.421) | 0.314960 | 0.083284 |
| 4321 | N4 | 0.522959 (+1.020 / +0.631) | 0.317059 | 0.083382 |
| 4321 | N5 | 0.502551 (-1.020 / +0.645) | 0.317204 | 0.085286 |

### 3 epoch：update1788

| Seed | 模型 | Rank-1 | mAP | mINP |
|---:|---|---:|---:|---:|
| 1234 | B0 | 0.505102 | 0.303841 | 0.083263 |
| 1234 | N2 | 0.500000 (-0.510 / -0.334) | 0.300502 | 0.079472 |
| 1234 | N3 | 0.538265 (+3.316 / +1.323) | 0.317067 | 0.084983 |
| 1234 | N4 | 0.515306 (+1.020 / +1.082) | 0.314662 | 0.088560 |
| 1234 | N5 | 0.543367 (+3.827 / +1.886) | 0.322699 | 0.086583 |
| 4321 | B0 | 0.507653 | 0.307395 | 0.085509 |
| 4321 | N2 | 0.515306 (+0.765 / +0.325) | 0.310650 | 0.087107 |
| 4321 | N3 | 0.497449 (-1.020 / +0.001) | 0.307403 | 0.087437 |
| 4321 | N4 | 0.477041 (-3.061 / -0.117) | 0.306226 | 0.087554 |
| 4321 | N5 | 0.512755 (+0.510 / +0.618) | 0.313570 | 0.088532 |

3 epoch 的平均相对 B0 差值：

| 模型 | 平均 Rank-1 差值 | 平均 mAP 差值 | Rank-1 两 seed 同号 |
|---|---:|---:|---|
| N2 | +0.128 pp | -0.004 pp | 否 |
| N3 | +1.148 pp | +0.662 pp | 否 |
| N4 | -1.020 pp | +0.483 pp | 否 |
| N5 | +2.168 pp | +1.252 pp | 是 |

## 5. N5 训练统计

以下为训练日志 rank0 观察值；selected/active 是负 PID 关系项计数，active batch 是至少有一个激活 PID 的 batch。

| Seed | 候选图关系 | 已选 PID 项 | 激活 PID 项 | 激活 batch 比例 | NDP 平均 loss | 有效更新 | AMP 跳步 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2048 | 17775 | 5573 | 62.64% | 0.007641 | 1788 | 0 |
| 4321 | 2048 | 17102 | 5164 | 60.54% | 0.007547 | 1788 | 1 |

20 步 smoke：

- Seed1234：20/20 有效更新，177 个已选 PID 项，60 个激活 PID 项，11/20 batch 激活，NDP 梯度非零。
- Seed4321：20/20 有效更新，189 个已选 PID 项，63 个激活 PID 项，13/20 batch 激活，NDP 梯度非零。

两个完整训练均保存 update596、1192、1788，并通过 checkpoint restore check。Seed4321 的单次 AMP 跳步没有导致参数非有限或训练中止。

## 6. train-only 局部 Margin、错误 PID、Purity@10

这些统计只使用 9576 张训练图像和固定 K=5 图，不是官方 query/gallery 指标。update0 是同一份参考特征；`new error PID` 为相对 update0 新增的 anchor-PID 错误。

| Seed | Update | 正确关系数 / 2048 | 平均 Margin | 新增错误 PID | Purity@10 |
|---:|---:|---:|---:|---:|---:|
| 1234 | 1192 | 1682 | 0.059172 | 149 | 0.970654 |
| 1234 | 1788 | 1704 | 0.064548 | 88 | 0.973340 |
| 4321 | 1192 | 1685 | 0.057793 | 57 | 0.969385 |
| 4321 | 1788 | 1683 | 0.056366 | 55 | 0.970850 |

与 update0 比较：Seed1234 的 update1788 正确关系数从1699升至1704、Purity@10 从0.972266升至0.973340；Seed4321 从1660升至1683、Purity@10 从0.967383升至0.970850。两 seed 的平均 Margin 也均高于 update0。与此同时仍存在新增错误 PID，不能声称 V5 完全没有邻域副作用。

这支持“多 PID 保护改善了整体局部结构”的方向性证据，但由于本轮没有对 B0/N2/N3/N4 在相同 update1192/1788 重新导出完整 train-only bank，不能严格声称 V5 已经因果上减少了相对所有基线的邻域损害。

## 7. 代码、数据与 checkpoint SHA256

| 对象 | SHA256 |
|---|---|
| NDP 训练入口 `ndp_e60_minimal_pair_runner.py` | `29629668f951029f16d886bc07169f0e44fdb7cffa10065aee386e70f8664f4c` |
| train-only 导出入口 | `2bae931e57f6686bc25584a8e238723127dbb679c0c927f920e9958b294e248c` |
| N5 局部指标脚本 | `8341795a7a42bd0f9ad948c010d0aee69ec63b68cd10739c41c10662b6cb7b46` |
| 固定 GPC engine | `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5` |
| Seed1234 V2 K=5 graph | `540ceb39f3ec44d734e020654e61b2a7688b26cb1e4ad14d61cdfdd67c2a6e9b` |
| Seed4321 V2 K=5 graph | `2b2556ea941fa0c19afa91b2bf3f0a3742d616faab5ad8580e1df455ad170355` |

N5 模型 checkpoint：

| Seed | update596 | update1192 | update1788 |
|---:|---|---|---|
| 1234 | `7b072fae7c6ad261265c105da46931f4f91ab3d8cbaf865d40a8085d88989185` | `67245c90afa92d338dc2924dc2dfae8ff6249dfddb0c814a56246a8051629bf9` | `015a7a18049955a8e1b2f2c49dcf4c9ead136952b10d174e9dc37c990935b96e` |
| 4321 | `37763d826c16c73bff7b6d6510345c8eb300f3183fc9746def1f96567aabf6c0` | `bd022880039303fbad7855360f143b1face1f1f272ffc59a7fd49cbb496971e4` | `9030c8c7ea34330363e1b3217475bae43340f1f168eea849f0d9c83093b3f124` |

所有官方评估 JSON 均记录 `distance_nan_count=0`、`distance_inf_count=0`，局部特征导出明确记录 `optimizer_step_executed=false`。

## 8. 直接回答关键问题

1. **V5 是否比 V2 更稳定？** mAP 是：N2 在 1/3 epoch 的 seed 方向不一致，而 V5 在 update1192 和 update1788 两个 seed 都相对 B0 为正。Rank-1 直到 update1788 才实现两个 seed 同时正增益。
2. **是否缓解 Seed4321 的负向结果？** 是，尤其是 3 epoch：V5 Rank-1 从 B0 的 0.507653 提升到 0.512755，mAP 从 0.307395 提升到 0.313570；但 1 epoch Rank-1 仍低于 B0。
3. **多负 PID 是否减少邻域身份损害？** train-only 结果显示两个 seed 在 3 epoch 的正确关系数、平均 Margin、Purity@10 均改善，且新增错误 PID 从 update1192 到 update1788 降低；这是支持性证据，不是严格的 B0/N2 matched-feature 因果证明。
4. **两个 seed 是否同时超过 B0？** 在 3 epoch update1788，Rank-1、mAP、mINP 三项均同时超过；在 1 epoch update596，Rank-1 没有同时超过。
5. **是否值得扩展到10 epoch？** 值得做一个仅 N5 的匹配 pilot。门禁应要求两个 seed 在预注册节点都不低于 B0，并继续检查 train-only 局部指标；在此之前不建议 60 epoch。

## 9. 限制与下一步

PPU 完整 EVA02 A/A 既有约 0.4% 相对梯度差仍然存在，本轮没有重复整套算子审计。Seed4321 发生一次 AMP 跳步，已保留在日志中。

下一步只建议：固定本版 N5 的 per-negative-PID 目标，运行匹配的 10 epoch 双 seed pilot，并在每个预注册节点导出 B0/N5 的同一 train-only 局部指标。不要在 10 epoch pilot 中同时改变 epsilon、lambda、图结构或 GPC。
