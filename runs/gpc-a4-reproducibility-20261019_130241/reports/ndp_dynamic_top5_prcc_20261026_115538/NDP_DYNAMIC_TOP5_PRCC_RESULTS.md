# NDP-Dynamic-TOP5 PRCC 实验结果

## 结论摘要

本轮已完成 PRCC 双 seed、B0/NDYN-TOP5 四组配对训练，各 3 个 raw continuation epoch，并在 continuation 1、3 固定节点完成官方 C→A 评估。

结论：**NDYN-TOP5 已真实参与训练并显著扩大了动态关系覆盖，但没有在两个 seed 上形成稳定的 Rank-1/mAP 增益，因此不通过继续延长到 5 epoch 的门禁。**

Seed1234 在第 3 节点获得 `+0.677 pp Rank-1 / +0.283 pp mAP`；Seed4321 同节点为 `-0.508 pp / +0.017 pp`。第 1 节点两个 seed 的 Rank-1 和 mAP 均略低于各自 B0。该结果支持“可激活、覆盖更高”，不支持“稳定有效”。

## 实验身份与数据完整性

- 用户指定最新提交：`500599f337389e032cec2aa63f11ad674cd83016`。
- 服务器训练目录不是 Git 工作树；该提交在 `/home/DIFFER/github_uploads/test2_20261021_recovery` 中核对为 HEAD。训练修改以本报告列出的脚本 SHA256 为准，未推送 GitHub main。
- 新实验目录：`/home/DIFFER/log_NDP/ndp_dynamic_top5_prcc_20261026_115538/`。
- 数据：PRCC train-only；参考图只使用训练图像和训练 PID 关系，没有使用 query/gallery。
- 官方评估：PRCC C→A。当前仓库评估器没有暴露 mINP，因此没有用其他协议替代。
- 固定 GPC run-id：`ndp_prcc_baseline_fix_20261025_130457`。
- Seed 1234 GPC pair phase：`ndp_v5_prcc_gpc_pair_ndp_prcc_baseline_fix_20261025_130457_seed1234`。
- Seed 4321 GPC pair phase：`ndp_v5_prcc_gpc_pair_ndp_prcc_baseline_fix_20261025_130457_seed4321`。
- GPC 源 `/data/DIFFER/scripts/gpc_v2_train.py` SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`。

每个 seed 的 B0 与 NDYN-TOP5 都从同一份 t1 模型权重、同一新建 optimizer/scheduler/GradScaler/RNG 状态分支。这是严格的 seed 内共同 model-only warm-start 新片段，不是 t1 历史 optimizer 状态的无缝续训。四个正式任务使用相同 raw batch 计划和 GPC phase；由于 AMP skip，effective update 数仍需单独报告。

## NDYN-TOP5 定义

对 t0/t1 中相同 image index、相同正样本和负 PID endpoint 计算：

```text
M0 > 0
M1 > 0
M0 - M1 > 0
risk_score = (M0 - M1) / (M1 + 0.02)
```

所有候选按 `(-risk_score, relation_index, pid_index)` 确定性排序，保留候选关系的最高 5%。训练时使用 t1 冻结特征作为参考，目标为：

```text
target = M1 + min(0.005, 0.5 * (M0 - M1))
L_NDP = mean_anchor(mean_pid(ReLU(target - M_current)))
L_total = L_base + 0.05 * L_NDP
L_base = CE + Triplet + clipBio + 0.1 * GPC
```

B0 分支不计算 NDP 损失；B0 的 NDP contribution 和 NDP 梯度为零。

### 参考图覆盖

| Seed | t0→t1 图关系数 | 合法动态候选 PID 关系 | Top5 PID 关系 | Top5 anchor 数 | 图 SHA256 |
|---:|---:|---:|---:|---:|---|
| 1234 | 10,311 | 5,111 | 256 | 215 | `aea741cf955d6ebf33e82eea7c0781e5b15b42295d0686068dbf4d8152c38862` |
| 4321 | 10,313 | 3,831 | 192 | 178 | `6253226f373cfeac40346e4a3384d6656502c8b66b9fddcd50315af961cb673c` |

复用的 t0/t1 特征 SHA256：

- Seed1234 t0：`7d9cd273420b73f059117eb3658f52b192c3e4be3509d071f654b6288355ebfb`
- Seed1234 t1：`b40c4e92804b64612bed07ede0013ce194dd452c284479547f7a27ac92e8e9a0`
- Seed4321 t0：`ca7cb2f672d7ff3c7aed2c8cb92d15d1768266fcc73c1783cf37e8a8c59fbcfb`
- Seed4321 t1：`ea800bf4585243b54d218bd549a315d4402a4ef5f61f839ba74cebe175807517`

相比原 NDYN 的 unique 动态 PID 数，Top5 分别从 92/36 扩大到 256/192；这确实改变并扩大了保护集合，而不是只改变日志名称。

## PRCC C→A 结果

Rank-1 和 mAP 以百分比报告；括号为 NDYN-TOP5 减去同 seed、同 continuation 节点 B0 的百分点差。

| Seed | continuation raw epoch | B0 Rank-1 | B0 mAP | NDYN-TOP5 Rank-1 | NDYN-TOP5 mAP | 差值 Rank-1 / mAP |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 62.997 | 56.981 | 62.885 | 56.881 | -0.113 / -0.099 pp |
| 1234 | 3 | 63.534 | 58.912 | 64.211 | 59.195 | +0.677 / +0.283 pp |
| 4321 | 1 | 62.631 | 56.943 | 61.840 | 56.736 | -0.790 / -0.207 pp |
| 4321 | 3 | 63.590 | 58.830 | 63.082 | 58.847 | -0.508 / +0.017 pp |

跨 seed 的平均差值为：

- continuation 1：Rank-1 `-0.452 pp`，mAP `-0.153 pp`；
- continuation 3：Rank-1 `+0.085 pp`，mAP `+0.150 pp`。

平均值不能掩盖 seed 方向不一致，因此不能称为稳定提升。

## 训练侧统计

`ndp_selected_relations` 是训练期间访问到的关系次数，`ndp_active_relations` 是产生正 hinge 的关系访问次数，不是 Top5 图中的 unique PID 数。`ndp_active_rate` 使用 runner 的全局 rank 汇总定义。

| Seed | 节点 | raw/effective updates | 活跃 Batch 数 | 选中关系访问 | 正 hinge 关系访问 | 活跃关系比例 | 活跃 Batch rate | NDP Loss | Purity@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 551 / 551 | 113 | 253 | 141 | 55.73% | 10.254% | 0.0004827 | 0.30798 |
| 1234 | 3 | 1653 / 1653 | 226 | 509 | 270 | 53.05% | 10.254% | 0.0010087 | 0.30571 |
| 4321 | 1 | 551 / 551 | 107 | 188 | 123 | 65.43% | 9.710% | 0.0004330 | 0.30513 |
| 4321 | 3 | 1653 / 1653 | 166 | 377 | 192 | 50.93% | 7.532% | 0.0007308 | 0.30402 |

B0 对应节点 NDP 均为 0，未计算 NDP 关系损失。正式训练统计没有 NaN/Inf。

### AMP 与有效更新

- B0 Seed1234：1 次 AMP skip，raw/effective 为 `551/550`、`1653/1652`。
- NDYN-TOP5 Seed1234：0 次 AMP skip，raw/effective 为 `551/551`、`1653/1653`。
- B0 Seed4321：0 次 AMP skip，raw/effective 为 `551/551`、`1653/1653`。
- NDYN-TOP5 Seed4321：0 次 AMP skip，raw/effective 为 `551/551`、`1653/1653`。

因此 seed1234 的两臂 raw batch 计划一致，但 effective optimizer update 不完全一致；这组结果不能夸大为无数值扰动的严格 optimizer-step 对照。

### 梯度验证

20 步 Smoke 中已经捕获两种子 NDYN-TOP5 的正 hinge 和非零梯度。正式日志前 20 个 diagnostic step 的最大值为：

| Seed | NDYN-TOP5 非零 NDP feat 梯度步数 | 最大 NDP feat 梯度 L2 | 非零 EVA02 probe 梯度步数 | 最大 probe 梯度 L2 |
|---:|---:|---:|---:|---:|
| 1234 | 2 | 0.010881 | 2 | 0.030559 |
| 4321 | 4 | 0.010917 | 4 | 0.033316 |

B0 两个 seed 均为 NDP feat/probe 梯度零。由此确认当前 anchor 没有被 detach，NDP 梯度确实进入 EVA02；本实验包含真实 optimizer.step。

## 与原始 NDYN 的探索性比较

原始 NDYN 结果来自 `/home/DIFFER/NDP_DYNAMIC_PRCC_RESULTS.md`，仅作探索性参照，起点和保护集合不同，不能视作严格同批消融。

| Seed | 节点 | 原 NDYN ΔRank-1 / ΔmAP | NDYN-TOP5 ΔRank-1 / ΔmAP |
|---:|---:|---:|---:|
| 1234 | 1 | +0.480 / +0.439 pp | -0.113 / -0.099 pp |
| 1234 | 3 | -0.141 / +0.045 pp | +0.677 / +0.283 pp |
| 4321 | 1 | -0.565 / +0.035 pp | -0.790 / -0.207 pp |
| 4321 | 3 | -0.339 / -0.404 pp | -0.508 / +0.017 pp |

Top5 的覆盖和激活明显高于原 NDYN，但性能方向仍不一致；因此“增加动态保护覆盖”本身没有证明为有效机制。

## 实际运行命令

公共环境：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
```

单元测试和编译检查：

```bash
/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_v5_b0_loss_contract.py
/usr/local/bin/python3.10 -m py_compile \
  scripts/ndp_v5_prcc_runner.py \
  scripts/ndp_dynamic_top5_reference.py \
  scripts/ndp_make_common_warmstart.py
```

结果：`13 passed`，`py_compile` 通过。

Top5 图构建命令的实际入口：

```bash
/usr/local/bin/python3.10 scripts/ndp_dynamic_top5_reference.py \
  --t0 <existing_t0_features.npz> \
  --t1 <existing_t1_features.npz> \
  --graph-t0 <existing_k5_graph.json> \
  --output-graph <new_top5_graph.json> \
  --output-manifest <new_top5_manifest.json>
```

Smoke 和正式训练均使用：

```bash
/usr/local/bin/python3.10 -m torch.distributed.run --standalone \
  --nproc_per_node=2 scripts/ndp_v5_prcc_runner.py --mode train \
  --seed {1234|4321} --arm {B0|NDYN_TOP5} \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --run-dir <new_run_dir> \
  --resume-checkpoint <common_seed_state> \
  --epochs 3 --eval-epochs 1,3 \
  --save-updates 551,1102,1653 --diagnostic-steps 20
```

NDYN-TOP5 额外传入对应 seed 的 t1 feature 和新 Top5 graph；Smoke 使用同一入口的 `--steps 20 --save-updates 20 --diagnostic-steps 20`。

## 代码与状态 SHA256

- `scripts/ndp_v5_prcc_runner.py`：`0de1179800e49dd9ae571f6df8fd815ac51d08cfaa75373bcd167ecd6c42f8eb`
- `scripts/ndp_dynamic_top5_reference.py`：`d3b46d92767f8e1029473543496c9d792ff83d8adb38ec86a423469376b78617`
- `scripts/ndp_make_common_warmstart.py`：`efaee6e3206ab715e1fa8e433669b6e772fe246d006309ed96d86fbbb1f14c42`
- `tests/test_ndp_v5_b0_loss_contract.py`：`6bf33417095a3042a33db80e42f6c3f94566095220a3d16a44b8069821dd3dec`
- Seed1234 common t1 model：`2cb7e822fdef9133aa250ac81fe2ab21295ad8af98f2e4c5811f8f56af4da98f`
- Seed4321 common t1 model：`3155a876f0b5b067ccf4c3f79a67ff31a93e81f0210ff693dd844f632956861b`

四个正式 `run_summary.json` 均为 `PASS`，对应模型、rank state、训练 CSV、AMP CSV 和评估日志均保存在新实验目录，未覆盖历史实验。

## 最终决策

1. **是否真正参与训练：是。** 两种子都出现正 hinge、非零 NDP Loss 和 NDP→EVA02 梯度；B0 对照为零。
2. **是否提高激活覆盖：是。** 相比原 NDYN，Top5 unique 关系和训练中活跃 Batch/正 hinge 访问明显增加。
3. **是否稳定改善 PRCC：否。** Seed1234 第 3 节点改善，但 Seed4321 Rank-1 下降；第 1 节点两个 seed 均不优于 B0。
4. **是否存在明显 Purity@10 损害：未见明显大幅变化。** 但 Purity 接近不变不等于没有局部关系损害，不能用它替代身份检索结论。
5. **是否继续 5 epoch：NO-GO。** 未满足两个 seed 同时正向的预设条件，不建议延长本配置。
6. **下一步：停止继续堆叠相似的历史 Margin 保护变体。** 如果继续研究，应转向验证动态退化关系的预测可靠性，以及它相对原始 Triplet 是否提供真正新的监督；应先解决 AMP/PPU 数值配对限制，再做独立 seed 的小规模验证。

已知完整 EVA02 PPU A/A 约 0.4% 梯度差异仍然适用。因此本轮小幅差值只能视为探索性观察，不能据此宣称 NDP-Dynamic-TOP5 有效；但覆盖扩大后仍没有两个 seed 一致收益，已经足以阻止下一阶段长程扩展。
