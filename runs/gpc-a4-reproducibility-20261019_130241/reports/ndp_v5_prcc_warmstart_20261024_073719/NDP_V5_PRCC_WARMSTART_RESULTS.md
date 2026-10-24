# NDP-V5 PRCC Warm-start 配对实验结果

实验日期：2026-10-24
实验目录：`/home/DIFFER/log_NDP/ndp_v5_prcc_warmstart_20261024_073719/`
远端 `main` 核对提交：`e2f8a28d746cb1f8436518c60804dc208bd18452`

## 结论先行

本轮修复了 GPC 配对上下文，并从两个 seed 各自的 B0 update1653 checkpoint 重建了 PRCC train-only NDP 参考图。20 步 Smoke 和四个正式 warm-start run 均完成。

在固定 update1653 节点上，N5 相对匹配 B0 的 C→A 结果为：

| Seed | B0 Rank-1 | N5 Rank-1 | 差值 | B0 mAP | N5 mAP | 差值 |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 63.90% | 64.01% | +0.11 pp | 58.45% | 59.24% | +0.78 pp |
| 4321 | 62.09% | 62.26% | +0.17 pp | 57.61% | 57.75% | +0.13 pp |

两个 seed 的方向一致，但 Rank-1 增益很小，且两臂发生了不同位置的 AMP skip。因此这是有条件的正向 warm-start 证据，不是已经排除数值混杂的最终结论。建议下一步只做一次同协议 5 epoch 验证，不进入 60 epoch 或设计 V7。

## 1. GPC 配对修复与验证

问题来源是旧 runner 将 `arm` 写入 `gpc.p7.PHASE`。固定 GPC 的 `stable_selection()` 会把 `PHASE`、seed、epoch、rank、iteration 和 pair index 一起哈希，因此 B0/N5 的 arm 名称会改变 GPC 关系顺序。

本轮改为：

```text
gpc.p7.PHASE = ndp_v5_prcc_gpc_pair_{run_id}_seed{seed}
```

不再包含 B0/N5 arm；GPC 选择算法、matched-budget、GPC 损失和权重 0.1 均未修改。

前 100 个 Batch、两个 rank 的验证结果：

| Seed | 比较 batch 数 | 固定 phase mismatch | 旧 arm-dependent phase mismatch |
|---:|---:|---:|---:|
| 1234 | 200 rank-batch | 0 | 200 |
| 4321 | 200 rank-batch | 0 | 200 |

固定 phase 验证文件：

- `gpc_pair_check_seed1234/gpc_pair_check.json`
- `gpc_pair_check_seed4321/gpc_pair_check.json`

这确认原配对问题真实存在，也确认本轮训练使用的 GPC 上下文已经匹配。

## 2. Warm-start 状态与 AMP 数据顺序

每个 seed 的 B0 checkpoint 同时作为该 seed 的 B0/N5 初始状态，恢复了：

- model
- optimizer
- scheduler
- GradScaler
- rank-specific RNG
- checkpoint 中的历史 epoch/global step

新的训练片段显式调用 `sampler.set_epoch(absolute_epoch)`，不是从历史 epoch1 重放。每组都运行 1653 个有效更新，并保存 update551、1102、1653。

AMP 事件如下：

| Seed | B0 | N5 | 影响 |
|---:|---|---|---|
| 1234 | 无 skip | batch757，epoch6 | N5 多消耗一个 replacement raw batch，结束于 absolute epoch8 |
| 4321 | batch777，epoch5 | 无 skip | B0 多消耗一个 replacement raw batch，结束于 absolute epoch7 |

因此两臂的有效更新预算相同，但在各自 skip 之后 raw batch 轨迹不再完全相同。这是 AMP 数值路径造成的实验限制，不能把本轮结果描述为逐 raw-batch 严格同步。所有 skip、scale 变化、global step 和实际 batch index 均记录在各 run 的 `amp_events.csv`、`training_steps.csv` 和 `run_summary.json` 中。

Seed1234 初始 checkpoint 的历史状态为 epoch4/global_step1654；Seed4321 为 epoch3/global_step1653。这是基于上一轮 update1653 节点的明确新 continuation 片段，不伪称为原训练无缝延续。

## 3. 训练后参考图

参考特征分别从对应 B0 update1653 模型导出，使用 PRCC 全部 train-only 图像的 1024 维身份特征。没有使用 query/gallery，也没有使用未来 checkpoint。

| Seed | 特征形状 | graph row | 历史正确负 PID | 相对上一轮 common-init graph row |
|---:|---|---:|---:|---:|
| 1234 | 17896×1024 | 10040 | 35613 | 2351 → 10040 |
| 4321 | 17896×1024 | 10369 | 36990 | 2351 → 10369 |

每个 anchor 最多 5 个不同负 PID，历史 gate 使用对应 B0 checkpoint 的正确 Margin。参考端在 NDP loss 中 stop-gradient，当前 batch 的 `feat[0]` 保留梯度。

## 4. Smoke Test

两个 seed 的 N5 warm-start Smoke 均 PASS，各 20 个有效 optimizer updates：

| Seed | selected relations | active relations | active rate | NDP loss | NDP feature grad max | probe parameter grad max | NaN/Inf |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1234 | 1357 | 21 | 20.0% | 0.0001001 | 0.0004330 | 0.0016982 | 无 |
| 4321 | 1349 | 48 | 42.5% | 0.0003293 | 0.0014812 | 0.0043474 | 无 |

两次 Smoke 都确认 NDP 真的参与反向传播，而不是只计算了 detached reference。

过程记录：Seed1234 Smoke 的第一次启动因命令误写了不存在的 graph 文件名而退出，错误发生在读取参考文件阶段，没有执行 optimizer step；修正为实际生成的 `prcc_v5_k5_graph.json` 后通过。该失败目录与修正后的 PASS 目录均保留。

## 5. 正式训练与 PRCC C→A 评估

### 5.1 官方指标

| Seed | 模型 | continuation 有效更新 | C→A Rank-1 | C→A mAP | mINP |
|---:|---|---:|---:|---:|---|
| 1234 | B0 | 1653 | 63.90% | 58.45% | N/A |
| 1234 | N5 | 1653 | 64.01% | 59.24% | N/A |
| 4321 | B0 | 1653 | 62.09% | 57.61% | N/A |
| 4321 | N5 | 1653 | 62.26% | 57.75% | N/A |

仓库评估器 `R1_mAP_eval` 不输出 mINP，因此没有填入估计值。

### 5.2 NDP 激活与局部指标

下表为正式片段的主要训练周期聚合统计，selected/active 等计数为双 rank 汇总：

| Seed | active batches | active rate | selected relations | active relations | 当前错误 PID | 当前正确 PID | 平均当前 Margin | NDP mean loss | Purity@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 826 | 24.98% | 105176 | 1700 | 13116 | 92060 | 0.010724 | 0.0001239 | 0.30619 |
| 4321 | 946 | 28.61% | 109419 | 2013 | 13949 | 95470 | 0.010832 | 0.0001462 | 0.30512 |

B0 不计算 NDP 关系统计。Batch-local Purity@10 对比为：

- Seed1234：B0 0.30513，N5 0.30619。
- Seed4321：B0 0.30520，N5 0.30512。

目前没有看到明显的聚合 Purity@10 损害，但这不能替代逐邻域反事实分析。

## 6. 实际命令与哈希

配对检查命令形式：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone \
  --nproc_per_node=2 scripts/ndp_v5_prcc_runner.py \
  --mode verify_gpc_pair --seed 1234 \
  --run-id ndp_v5_prcc_warmstart_20261024_073719 \
  --run-dir /home/DIFFER/log_NDP/ndp_v5_prcc_warmstart_20261024_073719/gpc_pair_check_seed1234 \
  --check-epoch 5
```

正式 warm-start 命令在四个 run 目录的启动记录中保留，核心参数为：

```text
--arm B0 或 N5
--epochs 3
--save-updates 551,1102,1653
--resume-checkpoint .../ndp_v5_prcc_20261024_042844/b0_seed{seed}_3e/node_update1653
```

代码与数据哈希：

- 新 runner：`ca068555936990e203c4de5b4300bfa32993535273534a43cec7119ffb2c7ed8`。
- 固定 GPC `/data/DIFFER/scripts/gpc_v2_train.py`：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`。
- B0 源 checkpoint：Seed1234 `2e63a254db95221580011d0b49ba4f00620f2c89d3ecfa2de4b92e39dada9bba`；Seed4321 `937e560dde233efa484ed1f5ac915c4653de587bee3a746130dade7c4e1b0298`。
- Seed1234 warm reference features：`cfa90437f80f917bf4911501c4ebf7b1ddb2f147e087a9df51aac3a8db520955`。
- Seed1234 warm graph：`475596ab2e47ffeb4905d11e3fbb059ff0882d7752c93441fcbff2e6b4a84a4c`。
- Seed4321 warm reference features：`fea248efa7499d05b94d4b3b0b05ea687ff2a1d0346552d70125e2b24f59b8b7`。
- Seed4321 warm graph：`577fd28f9398941738fca37c96f38ddd0946266ef27b06634d07747335b4db98`。

最终模型 checkpoint SHA256：

- B0/1234：`4728d7fc81a0e2651535c0961b11c418aa1aee14643cf05ad591c278178c9610`。
- N5/1234：`e6709ececafb33ef4afa6e9beb8dbbea478fe7625d1047819f455412f93882bb`。
- B0/4321：`472cd64d40ce7e6361fbb26781bc40f41dae5f9c092333237b10e73cd5a48c9c`。
- N5/4321：`7eac024b2ef2eb0d43fc68ab29c0e9e4b44857d6d3169d1f6e7f795a5ef61161`。

## 7. 决策

1. **GPC 对照是否已匹配？** 是。固定上下文前 100 Batch、两个 seed、两个 rank 均 0 mismatch；旧 arm-dependent 上下文全部产生 mismatch。
2. **训练后参考图是否增加覆盖？** 是。graph row 从 2351 提升到 10040/10369，历史正确负 PID 从 6892 提升到 35613/36990。
3. **NDP 是否真正参与训练？** 是。两个 Smoke 和两个正式 N5 run 都有 selected/active 关系、非零 NDP loss 及当前特征到 EVA02 的梯度。
4. **Rank-1/mAP 是否提高？** 本轮两个 seed 的两个指标均提高，但 Rank-1 只有 +0.11/+0.17 pp；同时 AMP skip 位置不同，需谨慎解释。
5. **是否值得继续？** 值得进行一次预注册的 5 epoch warm-start 验证，前提是继续保留固定 GPC phase、同一 B0 初始 state、同一 NDP 图和完整 AMP/数据顺序记录。当前不建议直接扩展到 60 epoch、修改 NDP loss 或设计 V7。

此前完整 EVA02 PPU A/A 约 0.4% 的数值非确定性仍然存在。本轮没有重复整套算子审计；因此小幅 Rank-1 增益不能被视为最终稳定收益。
