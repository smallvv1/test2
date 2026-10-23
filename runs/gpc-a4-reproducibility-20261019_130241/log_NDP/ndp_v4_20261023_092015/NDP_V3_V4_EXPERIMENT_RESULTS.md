# NDP-V3 / NDP-V4 实际训练结果

## 1. 结论摘要

本轮完成了 N3、N4 两种子的真实 1788 个有效更新和固定 LTCC 官方评估。

- N3 确实按要求从 N2 的 update596 恢复，并在第 2～3 epoch 关闭 NDP；它缓解了 Seed1234 的长期退化，但在 Seed4321 的两个固定节点均低于 B0，因此不是稳定改进。
- N4 从相同 E60 状态开始，将 `lambda_ndp` 从 0.05 按总 1788 步线性衰减到 0。Seed1234 两个节点优于 B0，Seed4321 两个节点低于 B0；没有跨 seed 的稳定收益。
- N4 相比 N3 不是一致改进：Seed1234 的 update1192 mAP 更高但 Rank-1 更低，update1788 的 Rank-1/mAP 均低于 N3；Seed4321 两个节点整体更差。
- NDP 训练路径、梯度和 checkpoint 均正常，但性能证据仍然存在明显 seed 依赖。建议停止 10/60 epoch 扩展，不把本轮结果写成 NDP 已有效。

最终状态：`NDP_MECHANISM = INCONCLUSIVE`，`CONTROLLED_SHORT_PILOT = NO_GO`。

## 2. 固定协议与数据来源

基础损失保持为：

`L_base = CE + Triplet + clipBio + 0.1 * GPC-v1`

本轮没有修改 EVA02、feat[0]、原始 CE/Triplet/Bio、GPC matched-budget 正样本选择、GPC 损失或采样规则。NDP 使用训练集 update0 特征和固定 2048-anchor、每 anchor 5 个不同负 PID 的图；参考端点 stop-gradient，只有当前 batch anchor 参与反向传播。

实验目录：

- N3：`/home/DIFFER/log_NDP/ndp_v3_20261023_084234/`
- N4：`/home/DIFFER/log_NDP/ndp_v4_20261023_092015/`
- N2/B0 复用：`/home/DIFFER/log_NDP/ndp_v2_20261023_070201/`
- 训练集参考图：`/home/DIFFER/outputs/ndp_v2_20261023_061358/reference_graph/`
- LTCC 官方评估配置：`/data/DIFFER/diagnostics/reconstructed_ltcc_eval_config.yml`

用户给定的最新已知 GitHub commit 为 `ebaced4c5da9ab5e5b2fce4157a7206cc40863e0`。本地 `/home/DIFFER` 不是 git worktree；固定 DIFFER+GPC 工程 `/data/DIFFER` 的本地 HEAD 为 `573b0502a231568d509fe3a463664c1a3c42e258`。

## 3. 实际运行命令

N3 Seed1234/4321 使用相同命令结构，仅替换 seed、参考图和恢复目录：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_e60_minimal_pair_runner.py --mode full --arm N3 --seed 1234 \
  --max-effective-updates 1788 --save-updates 1192,1788 \
  --resume-effective-updates 596 --resume-batch-index 596 \
  --resume-dir /home/DIFFER/log_NDP/ndp_v2_20261023_070201/epoch3_n2_seed1234/node_update0596
```

N4 Seed1234/4321 从 E60 初始状态运行：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_e60_minimal_pair_runner.py --mode full --arm N4 --seed 1234 \
  --max-effective-updates 1788 --save-updates 596,1192,1788 --initial-lr 1e-6 \
  --run-dir /home/DIFFER/log_NDP/ndp_v4_20261023_092015/n4_seed1234
```

固定节点评估实际调用：

```bash
/usr/local/bin/python3.10 scripts/ndp_official_ltcc_nodes.py \
  --seed 1234 --config /data/DIFFER/diagnostics/reconstructed_ltcc_eval_config.yml \
  --device-index 0 --node N4 1192 <node_update1192/model.pth> \
  --node N4 1788 <node_update1788/model.pth>
```

评估没有 optimizer.step、没有 query/gallery 选参；每个 JSON 都记录了 `distance_nan_count=0` 和 `distance_inf_count=0`。

## 4. LTCC 固定节点结果

Rank-1、mAP、mINP 均为官方固定 query/gallery 评估值；括号为相同 seed、相同 update 下相对于 B0 的差值，单位为百分点。

| Seed | 模型 | 有效更新 | Rank-1 | mAP | mINP |
|---:|---|---:|---:|---:|---:|
| 1234 | B0 | 1192 | 0.489796 | 0.306346 | 0.083148 |
| 1234 | N2 | 1192 | 0.520408 (+3.061) | 0.315039 (+0.869) | 0.085245 (+0.210) |
| 1234 | N3 | 1192 | 0.533163 (+4.337) | 0.314194 (+0.785) | 0.084844 (+0.170) |
| 1234 | N4 | 1192 | 0.528061 (+3.827) | 0.318536 (+1.219) | 0.090132 (+0.698) |
| 4321 | B0 | 1192 | 0.533163 | 0.314783 | 0.086440 |
| 4321 | N2 | 1192 | 0.515306 (-1.786) | 0.313974 (-0.081) | 0.085929 (-0.051) |
| 4321 | N3 | 1192 | 0.512755 (-2.041) | 0.312639 (-0.214) | 0.086057 (-0.038) |
| 4321 | N4 | 1192 | 0.505102 (-2.806) | 0.309571 (-0.521) | 0.084151 (-0.229) |
| 1234 | B0 | 1788 | 0.505102 | 0.303841 | 0.083263 |
| 1234 | N2 | 1788 | 0.500000 (-0.510) | 0.300502 (-0.334) | 0.079472 (-0.379) |
| 1234 | N3 | 1788 | 0.538265 (+3.316) | 0.317067 (+1.323) | 0.084983 (+0.172) |
| 1234 | N4 | 1788 | 0.515306 (+1.020) | 0.314662 (+1.082) | 0.088560 (+0.530) |
| 4321 | B0 | 1788 | 0.507653 | 0.307395 | 0.085509 |
| 4321 | N2 | 1788 | 0.515306 (+0.765) | 0.310650 (+0.325) | 0.087107 (+0.160) |
| 4321 | N3 | 1788 | 0.497449 (-1.020) | 0.307403 (+0.001) | 0.087437 (+0.193) |
| 4321 | N4 | 1788 | 0.477041 (-3.061) | 0.306226 (-0.117) | 0.087554 (+0.205) |

两 seed 的平均差值只能作为描述，不能掩盖方向冲突：

| 节点 | 模型 | 平均 Rank-1 差值 | 平均 mAP 差值 | 两 seed Rank-1 是否同号 |
|---:|---|---:|---:|---|
| 1192 | N3 | +1.148 pp | +0.285 pp | 否 |
| 1192 | N4 | +0.510 pp | +0.349 pp | 否 |
| 1788 | N3 | +1.148 pp | +0.662 pp | 否 |
| 1788 | N4 | -1.020 pp | +0.483 pp | 否 |

## 5. N3/N4 训练侧审计

### N3

N3 的两个恢复状态均完整包含 model、optimizer、scheduler、GradScaler 和 RNG。恢复状态的 `global_step=36356`，对应 `effective_update=596`；续训第一批进入 epoch62，没有重放 epoch61。恢复后 1192 个续训更新中，NDP selected/active 均为 0，基础损失仍正常更新。

第1 epoch 的 NDP 激活来自既有 N2 日志：

| Seed | 已选关系 | 激活关系 | 激活 batch 比例 | NDP 平均 loss |
|---:|---:|---:|---:|---:|
| 1234 | 3233 | 1262 | 46.34% | 0.009704 |
| 4321 | 3036 | 1105 | 40.75% | 0.008309 |

### N4

N4 的预注册权重为：

`lambda_ndp(update) = 0.05 * max(0, 1 - update / 1788)`

20 步 smoke 两 seed 均 PASS；N4 Seed1234 为 20/20 有效更新、NDP 激活 9 个 batch；Seed4321 为 20/20 有效更新、NDP 激活 9 个 batch。NDP 特征梯度和模型参数梯度均非零。

完整训练的 rank0 本地统计如下，候选图均为 2048 条；这不是跨 rank 的去重计数，而是训练日志中的 rank0 观察值。

| Seed | 历史正确关系 | 已选关系 | 激活关系 | 激活 batch 比例 | NDP 平均 loss | 有效更新 | AMP 跳步 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1699 | 3233 | 1307 | 47.23% | 0.010110 | 1788 | 1 |
| 4321 | 1660 | 3036 | 1131 | 40.64% | 0.008449 | 1788 | 1 |

两 seed 的最后一行 `step_executed=True`、参数有限；AMP scale 变化被保留在 `training_steps.csv` 和 `amp_events.csv`，没有因数值事件而覆盖或重跑结果。

## 6. 局部身份指标的边界

本轮直接完成的是训练日志激活统计和官方 LTCC 固定节点评估。N3/N4 没有再次导出 9576 张训练图像的完整 feature bank，因此不能把 N2 旧的 train-only local metrics 冒充成 N3/N4 的“正确身份关系数、错误 PID 或 Purity@10”。

可复用的既有 N2 update596 train-only 指标仅用于背景：

| Seed | 模型 | 正确关系率 | 新增错误 PID | Purity@10 |
|---:|---|---:|---:|---:|
| 1234 | B0 | 0.820801 | 97 | 0.972022 |
| 1234 | N2 | 0.814941 | 129 | 0.970703 |
| 4321 | B0 | 0.818848 | 52 | 0.968213 |
| 4321 | N2 | 0.819824 | 56 | 0.967627 |

这组数字不是 N3/N4 的结果，不能用来声称 N3/N4 改善或损害了局部关系。若继续研究，下一步应先对相同 update1192/1788 导出 N3/N4 的 train-only feature，再讨论局部身份机制，而不是继续增加训练 epoch。

## 7. 代码、配置和 checkpoint SHA256

| 对象 | SHA256 |
|---|---|
| NDP 训练入口 `scripts/ndp_e60_minimal_pair_runner.py` | `daaf08cdcaf1e391120ee32fa0e7d10c1f4015bd18a118d443ee9844aaf726cb` |
| 固定 GPC engine `/data/DIFFER/scripts/gpc_v2_train.py` | `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5` |
| 固定 LTCC eval config | `c7c9e967a5ae824bd9bfadfb27ea77bd1af645e4de009c18111609523d8d33e6` |
| Seed1234 V2 K=5 graph | `540ceb39f3ec44d734e020654e61b2a7688b26cb1e4ad14d61cdfdd67c2a6e9b` |
| Seed4321 V2 K=5 graph | `2b2556ea941fa0c19afa91b2bf3f0a3742d616faab5ad8580e1df455ad170355` |

N3 恢复源状态：

- Seed1234 rank0/rank1：`5806f9489b2588cce4f9a599abe92bdd3898d70667bce5a9bcd5119928e8ef6e` / `9cae973abb13e660b266e7fcc51734b15155d8d182aa843f1a0ed8ef1b2e7a9b`
- Seed4321 使用对应 N2 update596 状态，路径和字段审计同样 PASS。

N4 使用的 E60 初始状态 Seed1234 rank0/rank1 为：

- `e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`
- `9d6ff70981a594c0ad41f36c1d06b41926cc0f7f604fd787f5d4a00124651be0`

N3/N4 官方评估 JSON 保存了每个模型节点的 weight SHA256。N3 和 N4 的固定节点摘要如下：

| Seed | 模型 | update1192 | update1788 |
|---:|---|---|---|
| 1234 | N3 | `6fa22f414ea0c252daa1e45c82c355d06d02ff50273c95010ca63c5ba55b5270` | `bc1b2c02401049ccc1f11a191f88911f5d36ecec03b88f4e11fe59e56e12bccf` |
| 4321 | N3 | `c70614b8b2f3a7387fac71784efd4bad95f6da65c5fff46a02c678a071f78c38` | `94036f9e8e36c7c3d34a40f1c36bffb8cf774c4f797bdc5ab5d310a517da953d` |
| 1234 | N4 | `c0d4f8fc8675398e4b0052cf66d3910178aca978b063e4296afcf609617b0d0e` | `0d19e44b89baa91991fe2213c270d1da28d8330010538923ee1283709726961a` |
| 4321 | N4 | `460f05487d5b381e9ca4dac1d9ee6180a66af712432af2ce4ad0c889d7ac1fb9` | `f92f306c212e069bc3a24bc611aa4bacbc15a4017720205a18794b1061a5c8e9` |

## 8. 对用户问题的直接回答

1. **N3 是否缓解 N2 的 3 epoch 退化？** 对 Seed1234 是，N3 从 N2 的 Rank-1 0.500000 恢复到 0.538265；对 Seed4321 否，N3 为 0.497449，低于 B0 的 0.507653。因此只能说局部缓解，不能说稳定解决。
2. **N3/N4 哪个更好？** 没有统一赢家。N4 在 Seed1234 update1192 的 mAP 最好，但 N3 在该 seed update1788 的 Rank-1/mAP 都更高；Seed4321 N3/N4 均低于 B0，且 N4 更差。
3. **是否改善局部身份判别？** 本轮没有重新导出 N3/N4 train-only feature，因此没有足够证据回答局部关系数、错误 PID、Purity@10。LTCC 检索收益与局部机制之间不能直接等同。
4. **是否值得继续 10/60 epoch？** 不值得。两种子的方向冲突，N4 不能通过衰减消除 seed 依赖；当前建议 `NO_GO`。
5. **下一步最应该修改什么？** 先补齐 N3/N4 相同固定节点的 train-only 局部指标，定位 Seed4321 的负向原因（参考关系、邻域副作用或数值差异）；不应继续盲目延长训练或再叠加复杂 Loss。

## 9. 未完成事项与数值限制

- N3/N4 的 update1192/1788 train-only 9576-image feature export、正确关系数、错误 PID、Purity@10 本轮未执行，已明确标记，未编造结果。
- 既有完整 EVA02 A/A 相对梯度差约 0.4% 仍然存在。这里的 N3/N4 是探索性配对训练，微小收益不能视为已证实有效；本轮较大的 Seed1234 正向和 Seed4321 负向更像强 seed 依赖，而非稳定机制证据。
- 本轮没有修改 GPC 文件、没有推送 GitHub main、没有覆盖 N2/B0 历史目录。
