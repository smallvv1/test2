# NDP-V5 PRCC 5-epoch 配对实验结果

实验日期：2026-10-24
实验目录：`/home/DIFFER/log_NDP/ndp_prcc_5epoch_20261024_125141/`
远端 GitHub 主线核对提交：`e6ddd921cae2c63ab09db12dcf6b34a97a16cbc4`

## 结论先行

四个正式 run 均完成且 `run_summary.json` 为 `PASS`。本轮严格运行了 5 个
raw sampler epochs，并在 raw epoch 3、5 固定评估 PRCC C→A。

- 5 raw epochs 时两个 seed 的 N5 Rank-1 都高于匹配 B0：+0.79 pp、+0.14 pp。
- mAP 只有 Seed1234 提升 +0.88 pp；Seed4321 下降 -0.27 pp。
- 因此不能称为两个 seed、两个指标均稳定提升。当前证据支持“Rank-1 有一致但很小的正向方向”，不支持继续直接扩展到 60 epoch。
- NDP-V5 确实激活并参与训练；NDP loss、当前特征梯度和模型更新均为有限非零值。
- AMP skip 的位置在 B0/N5 之间不同，所以虽然 raw epoch 计划一致，不能把四臂称为严格逐 optimizer-update 配对。

## 1. 固定配置与数据边界

基础损失保持：

```text
L_base = CE + Triplet + clipBio + 0.1 * original GPC-v1
L_total = L_base + 0.05 * NDP-V5
epsilon = 0.02
```

固定项：EVA02、原始 CE/Triplet/Bio、GPC matched-budget 正样本选择、GPC
损失和权重 0.1、PRCC sampler 及 C→A 评估协议均未修改。N5 复用了两个
seed 对应的训练后 warm-start 参考图；参考特征 stop-gradient，当前 batch
的 `feat[0]` 保留反向路径。没有使用 query/gallery 构建训练参考图。

参考文件：

```text
Seed1234 feature: /home/DIFFER/log_NDP/ndp_v5_prcc_warmstart_20261024_073719/reference_seed1234_update1653/prcc_update1653_features.npz
Seed1234 graph:   /home/DIFFER/log_NDP/ndp_v5_prcc_warmstart_20261024_073719/reference_seed1234_update1653/prcc_v5_k5_graph.json
Seed4321 feature: /home/DIFFER/log_NDP/ndp_v5_prcc_warmstart_20261024_073719/reference_seed4321_update1653/prcc_update1653_features.npz
Seed4321 graph:   /home/DIFFER/log_NDP/ndp_v5_prcc_warmstart_20261024_073719/reference_seed4321_update1653/ndp_v5_k5_graph.json
```

源 checkpoint 是两个 seed 各自的 B0 `update1653` 完整状态，恢复 model、
optimizer、scheduler、GradScaler、RNG 后分出 B0/N5 两臂。每个 raw epoch
调用 `sampler.set_epoch(absolute_epoch)`，没有因 AMP skip 补跑额外 epoch。

## 2. 实际命令

Smoke Test 使用 2 PPU、20 个有效 update：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train --seed 1234 --arm N5 \
  --run-id ndp_prcc_5epoch_20261024_125141 \
  --run-dir /home/DIFFER/log_NDP/ndp_prcc_5epoch_20261024_125141/n5_seed1234_smoke \
  --feature <seed1234_feature.npz> --graph <seed1234_graph.json> \
  --resume-checkpoint <seed1234_b0_update1653> --steps 20 --diagnostic-steps 20
```

四个正式 run 使用相同命令模板，仅替换 seed、arm、run-dir 和 seed-specific
参考图：

```bash
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train --seed {1234|4321} --arm {B0|N5} \
  --run-id ndp_prcc_5epoch_20261024_125141 \
  --run-dir /home/DIFFER/log_NDP/ndp_prcc_5epoch_20261024_125141/{arm}_seed{seed}_5raw \
  --feature <seed-specific-feature.npz> --graph <seed-specific-graph.json> \
  --resume-checkpoint <matching-B0-update1653> \
  --epochs 5 --eval-epochs 3,5 --save-updates 1653,2755 --diagnostic-steps 20
```

Smoke Test PASS：20 个有效 update，NDP selected relations 1357，active
relations 21，active Batch rate 20.0%，NDP mean loss
`9.9686e-05`。前 20 步中 NDP feature gradient 最大值
`4.3299e-04`，NDP probe parameter gradient 最大值
`2.9186e-03`，总模型梯度有限且非零。Seed4321 没有重复单独 smoke；此前
同一 warm-start NDP-V5/参考图的 Seed4321 smoke 已 PASS，本轮正式 run 也
正常启动并完成。

## 3. PRCC C→A 官方指标

仓库评估器能输出 Rank-1、mAP，但不输出 mINP；因此 mINP 记为 N/A，未做
任何估计或替代填充。

| Seed | 模型 | Raw epoch | Raw batches | 成功 updates | Rank-1 | mAP | N5-B0 Rank-1 | N5-B0 mAP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1234 | B0 | 3 | 1653 | 1653 | 64.69% | 59.26% | — | — |
| 1234 | N5 | 3 | 1653 | 1652 | 64.10% | 58.70% | -0.59 pp | -0.56 pp |
| 1234 | B0 | 5 | 2755 | 2754 | 62.94% | 58.82% | — | — |
| 1234 | N5 | 5 | 2755 | 2754 | 63.73% | 59.71% | +0.79 pp | +0.88 pp |
| 4321 | B0 | 3 | 1653 | 1652 | 62.38% | 58.31% | — | — |
| 4321 | N5 | 3 | 1653 | 1652 | 62.40% | 58.04% | +0.03 pp | -0.26 pp |
| 4321 | B0 | 5 | 2755 | 2754 | 63.00% | 59.85% | — | — |
| 4321 | N5 | 5 | 2755 | 2754 | 63.14% | 59.58% | +0.14 pp | -0.27 pp |

第 3 epoch 是新 runner 的固定 raw-epoch 节点；它与上一轮“达到 1653
有效 updates 即停止”的 3 epoch 结果不是完全同协议，不能直接合并平均。

### 与此前 warm-start 3-epoch 结果比较

| Seed | 旧 3E B0 Rank-1 | 旧 3E N5 Rank-1 | 旧 3E B0 mAP | 旧 3E N5 mAP |
|---:|---:|---:|---:|---:|
| 1234 | 63.90% | 64.01% | 58.45% | 59.24% |
| 4321 | 62.09% | 62.26% | 57.61% | 57.75% |

旧结果来自 `ndp_v5_prcc_warmstart_20261024_073719`，旧 runner 按有效
optimizer updates 停止且遇到 skip 时会消耗额外 raw batch；本轮按固定 raw
epoch 停止。因此这两张表用于阶段趋势参考，不作为无条件的同轨迹因果对照。

## 4. NDP 激活与局部训练指标

下表是 N5 在各固定节点的双 rank 汇总。`active rate` 为 runner 记录的
active Batch 比例；`active/selected` 是实际激活负 PID 关系占已选关系的
比例。

| Seed | Raw epoch | Active batches | Active rate | Selected relations | Active relations | Active/selected | 当前错误 PID | 当前正确 PID | 平均当前 Margin | NDP mean loss | Purity@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 3 | 841 | 25.44% | 105176 | 1744 | 1.66% | 13102 | 92074 | 0.010683 | 0.0001247 | 0.30619 |
| 1234 | 5 | 516 | 23.41% | 70129 | 1029 | 1.47% | 7242 | 62887 | 0.012966 | 0.0001039 | 0.30734 |
| 4321 | 3 | 954 | 28.86% | 109419 | 2040 | 1.86% | 14142 | 95277 | 0.010797 | 0.0001481 | 0.30514 |
| 4321 | 5 | 583 | 26.45% | 72815 | 1178 | 1.62% | 7756 | 65059 | 0.013044 | 0.0001242 | 0.30518 |

B0 的 NDP 关系统计为 0（这是设计行为），用于局部 purity 对照：

- Seed1234：B0 purity 0.30616 → 0.30731，N5 0.30619 → 0.30734。
- Seed4321：B0 purity 0.30513 → 0.30518，N5 0.30514 → 0.30518。

这些 batch-local Purity@10 没有显示明显损害，但它不是完整的逐关系邻域
反事实审计，不能据此证明 NDP 没有伤害其他身份。

## 5. Raw batches、AMP skip 与配对限制

四臂都运行了完全相同的 raw batch 总数：epoch 3 为 1653，epoch 5 为
2755。实际 AMP 事件如下：

| Seed | B0 skip | N5 skip | epoch 3 成功 updates | epoch 5 成功 updates |
|---:|---|---|---:|---:|
| 1234 | batch 2161, epoch 8 | batch 1418, epoch 7 | 1653 / 1652 | 2754 / 2754 |
| 4321 | batch 699, epoch 5 | batch 1363, epoch 6 | 1652 / 1652 | 2754 / 2754 |

每臂均发生 1 次 `found_inf` skip，随后 GradScaler 按日志缩放。由于 skip
位置不同，成功 optimizer update、scheduler/AMP 状态以及后续数值轨迹不再
完全一致；本轮结果应称为“相同 raw epoch 预算的配对实验”，而不是严格的
逐 update 同轨迹实验。所有事件保存在各臂 `amp_events.csv`、
`training_steps.csv` 和 `run_summary.json`。

此前 GPC 配对验证在固定、不含 arm 的 phase 下，前 100 Batch、两个 rank
均为 0 mismatch；旧 arm-dependent phase 为 200 mismatch。新 runner 正式
训练使用：

```text
gpc_v2_train.PHASE = ndp_v5_prcc_gpc_pair_{run_id}_seed{seed}
```

因此 GPC 选择上下文不再因 B0/N5 arm 改变。

## 6. 代码、GPC、参考图和 checkpoint 哈希

- 新 runner：`scripts/ndp_v5_prcc_runner.py`
  `4797169a735a37394f9f3aee4a17f967c7f1e05db84010092490c6b6222a20`
- 固定 GPC `/data/DIFFER/scripts/gpc_v2_train.py`
  `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`
- 源 B0 update1653 checkpoint：Seed1234
  `2e63a254db95221580011d0b49ba4f00620f2c89d3ecfa2de4b92e39dada9bba`；
  Seed4321
  `937e560dde233efa484ed1f5ac915c4653de587bee3a746130dade7c4e1b0298`
- Seed1234 reference feature：
  `cfa90437f80f917bf4911501c4ebf7b1ddb2f147e087a9df51aac3a8db520955`
- Seed1234 graph：
  `475596ab2e47ffeb4905d11e3fbb059ff0882d7752c93441fcbff2e6b4a84a4c`
- Seed4321 reference feature：
  `fea248efa7499d05b94d4b3b0b05ea687ff2a1d0346552d70125e2b24f59b8b7`
- Seed4321 graph：
  `577fd28f9398941738fca37c96f38ddd0946266ef27b06634d07747335b4db98`

5 raw epochs 最终模型 checkpoint：

| Seed | B0 model SHA256 | N5 model SHA256 |
|---:|---|---|
| 1234 | `7c4dca27c6d487a6036a5c19f81d7044ffabe5a9c4169e0b84fad93aa0053b14` | `7ebfbc980f68f9b9673f2835d17b15f14b2a4d3ca1fc480a2389fe9c89a52272` |
| 4321 | `acfa951835c8ca43a174de6b4c2e5337f3eba2b23b2e6c4180745c71dcae6aa1` | `945db19db06e9efa9438362bb17140a4b94bf85572a3e0871dde48bf462bf868` |

## 7. 未完成事项与最终决策

- PRCC evaluator 当前未提供 mINP，故 mINP 未完成，不编造。
- 本轮未新增独立 seed，未进行 10/60 epoch 延长。
- PPU 完整 EVA02 A/A 约 0.4% 的既有非确定性仍然适用；本轮没有重新做整套算子审计。
- NDP 活跃关系只占约 1.47%～1.86% 的已选负 PID，说明大量历史关系在当前 margin 下零梯度；这可能限制长期收益。

**决策：暂不进入 60 epoch，也不设计 V7。** NDP-V5 在 PRCC 5 raw epochs
显示 Rank-1 两个 seed 同向小幅改善，但 mAP 只有一个 seed 改善，故尚不足以
宣称稳定性能提升。下一步若继续，优先做一次预注册的独立 seed/重复配对，
保持同一 GPC phase、同一 raw-batch 预算和同一 V5 损失；同时专门记录 NDP
梯度/激活覆盖，而不是继续延长当前配置。
