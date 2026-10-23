# NDP-V5 10 Epoch 配对实验结果

日期：2026-10-23
实验目录：`/home/DIFFER/log_NDP/ndp_v5_10epoch_20261023_111046/`
用户指定 GitHub 提交：`823c5a43829d2a4664471907abe42d9d8404c0ef`
固定 DIFFER 工程提交：`/data/DIFFER` HEAD = `573b0502a231568d509fe3a463664c1a3c42e258`

## 结论先行

NDP-V5 的 3 epoch 正向结果没有在 10 epoch 保持为稳定的 Rank-1 与 mAP 双指标提升。

- 3 epoch：两个 seed 的 N5 相对 B0 均有 Rank-1、mAP 提升，复用既有固定节点结果。
- 5 epoch：Seed1234 仍明显正向；Seed4321 Rank-1 持平、mAP 略低。
- 10 epoch：两个 seed 的 Rank-1 都低于 B0，但 mAP 都略高；因此不能称为稳定性能提升。
- train-only 局部关系在 10 epoch 出现 seed 依赖：Seed1234 改善，Seed4321 退化；这不能抵消 LTCC Rank-1 的长程退化。

决策：`NDP-V5 3 epoch = 保留为候选`；`NDP-V5 10 epoch = NO_GO`；不继续直接扩展到 60 epoch。下一步优先在固定 3 epoch 节点做独立 seed/PRCC 验证，不修改 NDP Loss，不设计 V6。

## 固定配置与方法

基础损失保持不变：

`L_base = CE + Triplet + clipBio + 0.1 * GPC`

本轮只增加既有 NDP-V5：

`L_total = L_base + 0.05 * L_NDP_V5`

`epsilon = 0.02`，2048 个历史 anchor，每个 anchor 最多 5 个负 PID；每个负 PID 独立计算历史/当前身份 Margin，再先在 anchor 内平均、再对有效 anchor 平均。历史参考端点冻结，当前 anchor 特征保留反向路径。B0 的 NDP 权重为 0。

没有修改 EVA02、GPC 正样本选择、GPC Loss、GPC 系数、CE、Triplet、clipBio、sampler 或评估协议。固定 GPC 文件 SHA256：

`/data/DIFFER/scripts/gpc_v2_train.py`
`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`

本轮研究代码 SHA256：

| 文件 | SHA256 |
|---|---|
| `scripts/ndp_e60_minimal_pair_runner.py` | `de7bf053ece0565bbc0e18da3b1b3b38f2fddb5a1c5b8961c38f5d398d907136` |
| `scripts/ndp_v2_feature_export.py` | `2bae931e57f6686bc25584a8e238723127dbb679c0c927f920e9958b294e248c` |
| `scripts/ndp_v5_local_metrics.py` | `8341795a7a42bd0f9ad948c010d0aee69ec63b68cd10739c41c10662b6cb7b46` |
| LTCC config | `c7c9e967a5ae824d9bfad9fb27ea77bd1af645e4de009c18111609523d8d33e6` |

## 真实执行命令

四个续训均使用以下入口，分别替换 `--arm`、`--seed`、`--resume-dir`、参考特征和图路径；保存 `2980,5960`：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_e60_minimal_pair_runner.py --mode full \
  --arm B0|N5 --seed 1234|4321 --max-effective-updates 5960 \
  --save-updates 2980,5960 --run-id <run_id> --run-dir <new_run_dir> \
  --reference-feature <seed_update0_full_features.npz> \
  --graph <multi_pid_reference_graph_v2.json> \
  --resume-dir <existing_update1788_dir> \
  --resume-effective-updates 1788 --resume-batch-index <1788|1789|1790> \
  --diagnostic-steps 20
```

固定评估命令：

```bash
/usr/local/bin/python3.10 scripts/ndp_official_ltcc_nodes.py \
  --seed 1234|4321 \
  --config /data/DIFFER/diagnostics/reconstructed_ltcc_eval_config.yml \
  --device-index 0 \
  --node B0 2980 <b0_seed/node_update2980/model.pth> \
  --node N5 2980 <n5_seed/node_update2980/model.pth> \
  --node B0 5960 <b0_seed/node_update5960/model.pth> \
  --node N5 5960 <n5_seed/node_update5960/model.pth> \
  --output <ltcc_seed_b0_n5_10e.json>
```

实际四个训练 run 的 `run_id` 与续训源如下：

| run_id | run_dir | resume source | resume batch index |
|---|---|---|---:|
| `ndp_v5_10epoch_20261023_111046_n5_s1234` | `.../n5_seed1234` | `ndp_v5_20261023_100048/n5_seed1234/node_update1788` | 1788 |
| `ndp_v5_10epoch_20261023_111046_b0_s1234` | `.../b0_seed1234` | `ndp_v2_20261023_070201/epoch3_b0_seed1234/node_update1788` | 1790 |
| `ndp_v5_10epoch_20261023_111046_n5_s4321` | `.../n5_seed4321` | `ndp_v5_20261023_100048/n5_seed4321/node_update1788` | 1789 |
| `ndp_v5_10epoch_20261023_111046_b0_s4321` | `.../b0_seed4321` | `ndp_v2_20261023_070201/epoch3_b0_seed4321/node_update1788` | 1790 |

四个命令均真实执行了 `--max-effective-updates 5960 --save-updates 2980,5960`；实际评估输出为：

- `ltcc_seed1234_b0_n5_10e.json`
- `ltcc_seed4321_b0_n5_10e.json`

实际导出的八个 train-only 特征文件和十六个局部指标 CSV/JSON 均位于本实验目录的 `local_features/` 与 `local_metrics/` 下。

局部指标命令使用 `scripts/ndp_v2_feature_export.py` 导出同一批 9576 张 train-only 图像，再使用 `scripts/ndp_v5_local_metrics.py`；导出和指标文件均记录 `optimizer_step_executed=false`、`official_query_gallery_used=false`。

## LTCC 固定节点结果

数值为 Rank-1、mAP、mINP；百分比列仅为便于阅读，原始 JSON 保留完整浮点值。

| Seed | 模型 | continuation epoch | update | Rank-1 | mAP | mINP |
|---:|---|---:|---:|---:|---:|---:|
| 1234 | B0 | 3 | 1788 | 50.51% | 30.38% | 8.33% |
| 1234 | N5 | 3 | 1788 | 54.34% | 32.27% | 8.66% |
| 1234 | B0 | 5 | 2980 | 50.51% | 30.25% | 8.26% |
| 1234 | N5 | 5 | 2980 | 52.81% | 32.02% | 8.53% |
| 1234 | B0 | 10 | 5960 | 51.28% | 30.18% | 7.69% |
| 1234 | N5 | 10 | 5960 | 50.77% | 30.97% | 7.65% |
| 4321 | B0 | 3 | 1788 | 50.77% | 30.74% | 8.55% |
| 4321 | N5 | 3 | 1788 | 51.28% | 31.36% | 8.85% |
| 4321 | B0 | 5 | 2980 | 52.04% | 30.82% | 8.02% |
| 4321 | N5 | 5 | 2980 | 52.04% | 30.74% | 7.83% |
| 4321 | B0 | 10 | 5960 | 51.53% | 30.72% | 8.07% |
| 4321 | N5 | 10 | 5960 | 51.02% | 31.05% | 8.25% |

相对 B0 的 N5 差值：

| Seed | 节点 | ΔRank-1 | ΔmAP | ΔmINP |
|---:|---:|---:|---:|---:|
| 1234 | 3 epoch | +3.83 pp | +1.89 pp | +0.33 pp |
| 1234 | 5 epoch | +2.30 pp | +1.76 pp | +0.27 pp |
| 1234 | 10 epoch | -0.51 pp | +0.79 pp | -0.03 pp |
| 4321 | 3 epoch | +0.51 pp | +0.62 pp | +0.30 pp |
| 4321 | 5 epoch | 0.00 pp | -0.08 pp | -0.19 pp |
| 4321 | 10 epoch | -0.51 pp | +0.33 pp | +0.18 pp |

因此 10 epoch 的 mAP 方向一致，但 Rank-1 方向在两个 seed 都相反；这不是满足“两个 seed 同时提升”的稳定结果。

## 训练状态、恢复和激活

四个 run 的 `effective_updates=5960`，`failure=null`，每个续训 CSV 有 4173 行训练记录，均有 4172 次实际 optimizer step。每个 run 出现 1 次 AMP `found_inf`，但非有限参数计数为 0，最终参数状态均为有限值。

| Seed | 模型 | NDP selected（rank0） | NDP active（rank0） | active batch rate | mean NDP loss | 最终 batch index |
|---:|---|---:|---:|---:|---:|---:|
| 1234 | B0 | 0 | 0 | 0.0000 | 0 | 5963 |
| 1234 | N5 | 41,801 | 11,816 | 0.6240 | 0.007521 | 5961 |
| 4321 | B0 | 0 | 0 | 0.0000 | 0 | 5963 |
| 4321 | N5 | 40,625 | 10,977 | 0.6000 | 0.007162 | 5962 |

N5 continuation 的历史正确关系数为 1947/2048（两 seed 相同）；B0 的该字段为诊断参考，不参与损失。N5 的有效激活率相对 3 epoch 既有结果（Seed1234 0.6264、Seed4321 0.6054）没有明显漂移。

四个最终节点的 rank0/rank1 状态均包含 `epoch`、`global_step`、`model`、`optimizer`、`scheduler`、`scaler`、`rng`、`phase`，恢复检查均为 `PASS`。由于原节点 AMP skip 不同，四组的绝对 batch index 存在小偏移；本轮按各自 checkpoint 的 sampler 位置续接，没有把跨模型 batch 序列伪称为逐 batch 完全一致。

## Train-only 局部身份指标

固定 2048 关系、同一 update0 参考特征和同一 K=5 图。`new_error_pid` 相对于该 seed 的 update0 关系错误集合计算；Purity@10 仅使用 train-only 9576 图像。

| Seed | 模型 | update | 正确关系 | 平均 Margin | 新增错误 PID | Purity@10 |
|---:|---|---:|---:|---:|---:|---:|
| 1234 | B0 | 2980 | 1610 | 0.05518 | 259 | 0.96445 |
| 1234 | N5 | 2980 | 1703 | 0.06581 | 85 | 0.97266 |
| 1234 | B0 | 5960 | 1695 | 0.06797 | 152 | 0.97168 |
| 1234 | N5 | 5960 | 1717 | 0.07308 | 107 | 0.97231 |
| 4321 | B0 | 2980 | 1679 | 0.05776 | 96 | 0.97051 |
| 4321 | N5 | 2980 | 1691 | 0.06172 | 89 | 0.96797 |
| 4321 | B0 | 5960 | 1722 | 0.06578 | 57 | 0.97227 |
| 4321 | N5 | 5960 | 1683 | 0.06302 | 95 | 0.96899 |

局部指标说明：Seed1234 在 5/10 epoch 均显示 N5 的关系保护效果；Seed4321 在 5 epoch 的关系数和平均 Margin 改善但 Purity 略低，10 epoch 则正确关系、平均 Margin、错误 PID 和 Purity 均劣于 B0。由此只能判定 N5 的局部保护存在明显 seed 依赖，不能声称没有附带损害，也不能据此推断 LTCC 全局 Rank-1 会长期改善。

## Checkpoint 与数据来源

本轮最终模型 SHA256：

| Seed | 模型 | update2980 | update5960 |
|---:|---|---|---|
| 1234 | B0 | `ba9baee6ec35053536e7e1888ff5e3ab5c3da28426eb36b878a110dbbea9c0fa` | `c4d7f858da635066871e691c7763531f77bd09f53f457751f4e55019a81d44fb` |
| 1234 | N5 | `9fe8e619c84dc6c97d4c4585338f8ea2b73e490e8f85b3f48c9cb1ff378b5017` | `1b6cb13eac5c8711dc611ee08573068c87489db66afad5cb82665961b5a60aec` |
| 4321 | B0 | `50c01187d8e4a4acb17c552890305ace42545639668693954126e55d4b60d1c3` | `a2439d3031267136859e1051417b0fff7f19c84ff8593d994ee42d3c7f57c610` |
| 4321 | N5 | `25626f3c51d121d0e9e76da03faed7f81a1ea04358f80e6e3acec9487a3862fc` | `06438ec192842932026571b1139f5612a5127ae5e5678d90e1457989ec86c6de` |

来源：LTCC train/query/gallery 使用原有数据与换装过滤协议；局部图使用 `/home/DIFFER/outputs/ndp_v2_20261023_061358/reference_graph/` 的 seed-specific K=5 图和既有 update0 train-only bank。没有使用测试集事件选择 mask、超参数或 checkpoint。

## A/A 与数值限制

此前完整 EVA02 PPU A/A 反向重复约有 0.4% relative L2 差异。本轮保留该限制，没有重新进行算子审计。训练日志中每个续训出现一次 AMP skip，但所有参数有限、训练完成且官方评估距离无 NaN/Inf。因此本轮可以报告为探索性长程结果，不能把小于该数值不确定性的变化视为已证实机制收益。

## 最终判断

1. **NDP-V5 是否随训练保持提升？** 否。3 epoch 两 seed 正向，10 epoch Rank-1 两 seed 均转负；mAP 仍小幅正向，但不是完整一致提升。
2. **是否存在明显局部身份损害？** 存在 seed 依赖：Seed1234 的 10 epoch 局部指标改善，Seed4321 的 10 epoch 局部指标退化；不能把 N5 视为无附带损害。
3. **是否值得继续 10/60 epoch？** 不建议继续 60 epoch，也不把当前 10 epoch 作为可推广配置。优先保留 3 epoch 结果，进行固定节点的独立 seed 与 PRCC 验证。
4. **下一步最应修改什么？** 本轮不修改 Loss。先确认 3 epoch 的收益能否在新 seed/PRCC 复现；若复现，再单独研究“训练阶段长度/关闭时机”，而不是继续提高训练时长。

结论标签：`NDP-V5_3E_CANDIDATE`；`NDP-V5_10EPOCH_NO_GO`；`NDP_MECHANISM = INCONCLUSIVE`。
