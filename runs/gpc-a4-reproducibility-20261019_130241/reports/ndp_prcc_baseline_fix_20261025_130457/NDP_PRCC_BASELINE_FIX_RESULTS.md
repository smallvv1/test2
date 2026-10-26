# NDP PRCC B0 基线修复与 NDP-V5 配对实验结果

## 结论先行

本轮已完成修复后的 PRCC B0 重新训练、训练后参考图重建，以及两个 seed 的 B0/N5 配对 continuation。修复后的 B0 不再包含 NDP 项，N5 的 NDP-V5 可以正常反向传播，但在这组公平程度受 AMP skip 影响的 5-raw-epoch 对照中，没有得到两个 seed 一致的 Rank-1、mAP 提升。

因此本轮结论是：

**PRCC 上 NDP-V5 真实参与训练，但修复 B0 后尚未证明具有稳定性能增量；暂不进入 EPS005、Floor0、V7 或更长训练。**

## 实验身份与可复现信息

- 远端 `test2/main` 核对结果：`f5bed19afaf27c6ff831424608a3939410e9f9b8`。
- 本轮 run：`ndp_prcc_baseline_fix_20261025_130457`。
- 日志根目录：`/home/DIFFER/log_NDP/ndp_prcc_baseline_fix_20261025_130457/`。
- 使用的训练入口 SHA256：`e3a44cccc87dfda419bbf56bbe84d37113e574fa50ce47d5be4ad7a858ac669c`。
- GPC 源文件 SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`。
- B0 契约测试 SHA256：`c176f8889e3abb3bf41f7cefb86faf3ef9aa916211ba914f0ac404d2b9e8a95f`。
- 本地 `/data/DIFFER` 工作树含大量既有研究修改，且其 Git HEAD 为 `573b0502a231568d509fe3a463664c1a3c42e258`；本轮以 `/home/DIFFER/scripts/ndp_v5_prcc_runner.py` 的实际 SHA 和上述远端 commit 记录实验身份，没有覆盖或提交 GitHub。

固定项没有改变：EVA02、CE、Triplet、clipBio、GPC-v1 matched-budget 选择、GPC 损失、GPC 系数 `0.1`、优化器、sampler 和 PRCC C→A 评估流程。

## 1. B0 Loss bug 修复

旧逻辑在 B0 分支仍把 `ndp_loss` 以系数 1 加到总损失：

```python
total = main_total + (LAMBDA_NDP * ndp_loss if ndp_enabled else ndp_loss)
```

这使得旧 B0 实际上不是固定的 `CE + Triplet + clipBio + 0.1 * GPC`，而是额外加入了未乘 lambda 的 NDP 项。修复后使用显式 arm 合成：

```python
if arm == "B0":
    total = main_total
else:
    total = main_total + LAMBDA_NDP * ndp_loss
```

同时，B0 不再加载、遍历或计算 NDP 参考关系；它只执行原始基础损失和 GPC。N5 才加载参考图并计算逐负 PID NDP-V5。代码位置为 `scripts/ndp_v5_prcc_runner.py:367`、`:450` 和 `:578` 附近。

验证命令：

```bash
PYTHONPATH=/home/DIFFER/scripts:/data/DIFFER/scripts:/data/DIFFER \
  /usr/local/bin/python3.10 -m pytest -q \
  /home/DIFFER/tests/test_ndp_v5_b0_loss_contract.py
/usr/local/bin/python3.10 -m py_compile \
  /home/DIFFER/scripts/ndp_v5_prcc_runner.py
```

结果：`2 passed`，`py_compile` 通过。测试覆盖：B0 的 total 与 main_total 为同一数值且 NDP 梯度为零；N5 的 total 为 `main_total + 0.05 * ndp_loss` 且 NDP 梯度非零。

## 2. 正确 B0 三 raw epoch 预训练

旧的 `b0_cont_seed1234_5raw` 和 `b0_cont_seed4321_5raw` 命令漏传 `--resume-checkpoint`，实际从 common initialization 开始，因此保留但从本报告的配对比较中排除。它们没有被伪装成正确 B0。

本轮从 PRCC Bio-only common initialization 重新训练 B0，未使用 query/gallery，也未使用旧的错误 B0 checkpoint：

| Seed | raw epoch 3 Rank-1 | raw epoch 3 mAP | raw batches | effective updates | NDP contribution |
|---|---:|---:|---:|---:|---:|
| 1234 | 58.1428% | 51.6384% | 1653 | 1652 | 0 |
| 4321 | 57.4937% | 51.5071% | 1653 | 1652 | 0 |

checkpoint：

- Seed1234：`b0_seed1234_3raw/node_rawepoch03_update1652/`，model SHA256 `b803504065e60185ced22654bdbb74f47bc4f404b4a3b8bf1e34a36c7ce70b18`。
- Seed4321：`b0_seed4321_3raw/node_rawepoch03_update1652/`，model SHA256 `9c1bbfcd4830cd2f109a5c78a1ee6d59986af798d8c0f651e729b46e81d03cac`。

每个 checkpoint 目录均保存 rank0/rank1 state，包含 model、optimizer、scheduler、GradScaler 和 RNG；run summary 状态为 `PASS`。预训练阶段 seed1234、4321 各有一个 AMP skip，故有效更新为 1652，raw batch 为 1653。

## 3. 训练后 PRCC train-only NDP 参考图

参考图从各自修复后 B0 的 update1652 checkpoint 导出，使用 PRCC train-only 的 17896 张训练图像；没有使用 query/gallery。每个 anchor 最多 5 个负 PID，关系只保留历史 update1652 中 margin 正确的负 PID。

| Seed | feature shape | anchor 数 | K=5 relation 数 | 历史正确 PID 数 |
|---|---|---:|---:|---:|
| 1234 | 17896 × 1024 | 17896 | 10311 | 36881 |
| 4321 | 17896 × 1024 | 17896 | 10313 | 37080 |

参考文件及 SHA256：

- Seed1234 feature：`reference_seed1234_update1652/prcc_update1652_features.npz`，`a1562766aad797e598e3b7fa658c6c5b64d72f3a61261acfb0ec81db30a69da7`。
- Seed1234 graph：`reference_seed1234_update1652/prcc_v5_k5_graph.json`，`b7fd9492bdf9315a2db2b4b9c56674ae496b8c6e0ef4ac77ee67173642efb930`。
- Seed4321 feature：`reference_seed4321_update1652/prcc_update1652_features.npz`，`464e170e346d8b227761af9e6457a7dab9e934422f75699358ecba64f9885d35`。
- Seed4321 graph：`reference_seed4321_update1652/prcc_v5_k5_graph.json`，`6c153dcb1cfe2d5ded851ef8a70125c83475dcfb744d2d32158374e8be64826e`。

## 4. Smoke Test

### B0 smoke

- 20 个有效 optimizer updates：PASS。
- `ndp=-0.0` 占位零值，NDP selected/active 均为 0，lambda 为 0。
- `found_inf=0`，参数更新正常。

### N5 smoke

- 从 seed1234 修复后 B0 checkpoint 启动，20 个有效 updates：PASS。
- NDP selected relations：1271；active relations：27；active batch rate：25.0%。
- NDP mean loss：`1.47024e-4`。
- 前 20 步 NDP feature gradient max：`1.06392e-3`。
- 前 20 步 NDP probe parameter gradient max：`3.57929e-3`。
- `found_inf=0`，20/20 optimizer steps 执行。

这确认 B0 不产生 NDP 梯度，N5 的当前 anchor 仍保留向 EVA02 的梯度路径；参考端点没有反向更新。

## 5. 修复后 B0/N5 五 raw continuation 对照

两臂均从各自同一个修复后 B0 update1652 checkpoint 分支，使用同一 seed、same GPC pair phase、same raw batch continuation。N5 的 `lambda_ndp=0.05`、`epsilon=0.02`，其余训练设置固定。报告中的 epoch 3/5 是 continuation segment raw epoch；对应绝对训练 epoch 为 6/8。

### Seed 1234

| continuation raw epoch | 模型 | Rank-1 | mAP | 相对匹配 B0 Rank-1 | 相对匹配 B0 mAP |
|---:|---|---:|---:|---:|---:|
| 3 | B0 | 63.3644% | 58.3306% | — | — |
| 3 | N5 | 62.6588% | 57.7206% | -0.7056 pp | -0.6100 pp |
| 5 | B0 | 62.9410% | 59.1704% | — | — |
| 5 | N5 | 61.9249% | 58.6693% | -1.0161 pp | -0.5011 pp |

### Seed 4321

| continuation raw epoch | 模型 | Rank-1 | mAP | 相对匹配 B0 Rank-1 | 相对匹配 B0 mAP |
|---:|---|---:|---:|---:|---:|
| 3 | B0 | 63.2797% | 57.8626% | — | — |
| 3 | N5 | 63.5620% | 58.0502% | +0.2822 pp | +0.1877 pp |
| 5 | B0 | 63.7313% | 59.6883% | — | — |
| 5 | N5 | 63.8442% | 59.6111% | +0.1129 pp | -0.0772 pp |

评估器没有暴露 mINP，所有本轮 mINP 均记为 `not exposed by repository R1_mAP_eval`，没有用其他指标冒充 mINP。

## 6. N5 训练侧统计

| Seed | continuation epoch | NDP active batch rate | selected PID relations | active PID relations | active/selected | NDP mean loss | mean current margin | Purity@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 3 | 27.98% | 108925 | 1910 | 1.754% | 1.2578e-4 | 0.010341 | 0.306500 |
| 1234 | 5 | 25.91% | 72592 | 1172 | 1.615% | 1.1210e-4 | 0.012512 | 0.306596 |
| 4321 | 3 | 28.86% | 109678 | 2082 | 1.898% | 1.5156e-4 | 0.010856 | 0.305133 |
| 4321 | 5 | 25.73% | 72969 | 1231 | 1.688% | 1.2985e-4 | 0.013153 | 0.305209 |

前 20 个 diagnostic steps 的未加 lambda NDP-loss feature gradient：

- Seed1234：max `1.0639e-3`，mean `1.4658e-4`；main-loss feature gradient max `0.4048`，mean `0.2579`。
- Seed4321：max `1.4162e-3`，mean `2.8320e-4`；main-loss feature gradient max `0.4536`，mean `0.2756`。

这是非零但很小的 NDP 梯度信号；N5 实际总损失中的 NDP 梯度还要乘 `0.05`。N5 两 seed 均 2755/2755 optimizer steps，无 AMP skip；B0 seed1234 有 1 个 AMP skip（2754/2755 有效 updates），B0 seed4321 无 skip。故 seed1234 的配对不是逐个成功 optimizer update 完全同轨迹，只能称为相同 raw continuation 计划下的配对。

GPC 对照的训练配置使用同一个 seed-specific pair phase，并且匹配节点的 `gpc_selected_relations` 相同：

- seed1234：continuation epoch 3 为 32742，epoch 5 为 21876。
- seed4321：continuation epoch 3 为 32608，epoch 5 为 21688。

本轮没有生成单独的 `gpc_pair_check.json` 文件，因此不能把“选择数量相同”夸大为已归档的前100个 batch 全部关系 ID 逐项审计；代码已固定用 `selected_v2(..., "B0", ...)` 和相同 pair phase，且 B0/N5 配置中的 GPC SHA 相同。

## 7. 训练命令与数据约束

修复后 B0 三 epoch 的命令形式：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train --seed 1234 --arm B0 \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --run-dir /home/DIFFER/log_NDP/ndp_prcc_baseline_fix_20261025_130457/b0_seed1234_3raw \
  --epochs 3 --eval-epochs 3
```

N5 和修复后 B0 continuation 都使用 `--resume-checkpoint` 指向对应 `b0_seed*_3raw/node_rawepoch03_update1652`，并运行 `--epochs 5 --eval-epochs 3,5`。N5 额外传入对应 seed 的 train-only feature/graph 和 `--ndp-epsilon 0.02`。

没有读取未来 checkpoint 来选择训练关系；参考图只读取 B0 update1652 的 train-only 特征。PRCC query/gallery 只用于固定节点评估，没有用于训练关系选择或调参。

## 8. 最终判断

1. **B0 bug 是否修复？** 是。B0 不加载 NDP 图、不计算关系损失，total 严格等于 main_total；单元测试和 20 步 smoke 通过。
2. **NDP 是否真实参与训练？** 是。N5 20 步和正式训练都有非零 NDP loss、active relations、NDP feature/probe gradients；参数更新和 AMP 正常。
3. **修复后 NDP-V5 是否在两个 seed 上提升 Rank-1/mAP？** 否。Seed4321 有小幅 Rank-1 提升，但 mAP 在第5节点略降；Seed1234 两个节点均下降。
4. **旧结果是否仍可直接作为 PRCC B0 证据？** 否。旧 B0 受 bug 影响，且本轮发现的漏传 resume 命令结果也已排除。
5. **是否可以声称 NDP 已在 PRCC 有稳定收益？** 不可以。当前结果具有 seed 依赖，且 seed1234 还有 AMP skip 导致的 raw-plan/有效 update 轨迹差异。

## 决策

本轮 **NO-GO for immediate longer training**：不跑 EPS005、不改 Floor0、不设计 V7、不直接延长 60 epoch。保留本轮全部 checkpoint 和日志作为正确 B0 基线与 N5 负/混合结果。

若后续继续，最值得做的不是再改 Loss，而是先在现有 runner 中生成并保存 B0/N5 前100个 raw batch 的逐项 GPC relation-ID 对照，同时用一个不含 AMP skip 的严格匹配短程重复确认 seed4321 的微小增益是否超过数值噪声；在该证据之前，不应把 NDP-V5 作为 PRCC 已验证改进。

PPU 已知的约 0.4% 级 A/A 非确定性仍然适用。本轮的零点几 pp 变化，尤其 seed4321 的 +0.1129 pp Rank-1 / -0.0772 pp mAP，不能视为可靠增益。
