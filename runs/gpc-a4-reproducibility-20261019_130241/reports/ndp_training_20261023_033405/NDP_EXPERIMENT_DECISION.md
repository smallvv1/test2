# NDP 训练实验报告（2026-10-23）

代码版本：`164d7b1a3242157c2e898caffd03d65445f3ab35`（远端 HEAD 与 main 核对一致）。本轮报告目录：`/home/DIFFER/reports/ndp_training_20261023_033405`。

固定约束：EVA02、feat[0] 1024 维、原始 CE/Triplet/clipBio、GPC-v1 matched-budget 选择及 `0.1` 系数均由现有训练引擎提供；本轮只新增最小 NDP。NDP 使用 update0/E60 的 train-only K=5 固定图，历史正确条件为 `M_ref(update0)>0`，当前项为 live anchor 与冻结端点的同一 `mean-positive cosine - max-negative cosine` Margin，`epsilon=0.02`、`lambda_ndp=0.05`。训练关系选择没有读取未来节点、query/gallery 或测试信息。

核心命令：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_e60_minimal_pair_runner.py --mode smoke|full --seed 1234|4321 \
  --run-id <timestamp> --run-dir <new-output-dir> --diagnostic-steps 20
```

新代码和 checkpoint 的 SHA256 见 [`NDP_CHECKPOINT_AND_CODE_SHA256.csv`](NDP_CHECKPOINT_AND_CODE_SHA256.csv)；逐关系 train-only 结果见两个 `local_relation_metrics_seed*.csv`，节点汇总见 `local_node_summary_seed*.csv`。这些 CSV 未使用官方 query/gallery。

## 结论摘要

| 问题 | 证据结论 |
|---|---|
| NDP 是否真正参与训练 | 是。两 seed 均有真实激活、非零 live feature/parameter gradient、20/400 有效 optimizer steps，且 checkpoint 保存/恢复 PASS。 |
| 有多少有效激活 | 400 步 rank0 统计：seed1234 为 54/400 batch、135 条关系；seed4321 为 42/400 batch、98 条关系。候选固定为 2560，历史正确分别 1197/1158。 |
| 是否改善局部身份判别 | 有限且不一致。400 步正确关系 +4/+10；平均 Margin +0.001174/−0.000059；Purity@10 −0.000977/−0.001172。 |
| LTCC Rank-1/mAP/mINP | Rank-1 两 seed 均下降；mAP 与 mINP 两 seed 均只有约 0.00007–0.00149 的小幅增加，不能视为可靠收益。 |
| 是否损害附近其他身份 | 固定 K=5 邻域的整体 Purity@10 两 seed 均轻微下降；这提示存在微弱局部代价，但不能仅由该 proxy 断言结构性伤害。逐关系证据在 CSV，且没有执行网络一步后的反事实邻域隔离实验。 |
| 两 seed 是否一致 | 只有“Rank-1 不升”一致；局部平均 Margin、Purity 和其它检索指标不形成一致正向模式。 |
| 是否值得继续 10/60 epoch | 当前 NO-GO。先不扩展 10 或 60 epoch，不增加复杂双侧/多 PID损失；应先修正/验证训练时关系覆盖和邻域 collateral 的独立因果诊断。 |
| 下一步最值得修改哪一处 | 不是阈值。优先记录每个 NDP active relation 的 anchor/negative PID，并在同一 batch/同一冻结状态下做目标关系与同 anchor 其他 4 个 PID 的梯度预算匹配；只有确认 collateral 不是由 endpoint 冻结近似或低覆盖造成，才考虑新的候选。 |

## 门禁判断

- NDP_MECHANISM：`INCONCLUSIVE`。
- A/A_REPRODUCIBILITY：`FAIL`（沿用既有完整模型约 0.4% A/A 限制，本轮未重做 A/A）。
- CONTROLLED_SHORT_PILOT：`NO_GO` 作为进一步正式扩展门禁；本轮 400 步本身是已授权的探索性 pilot，已完成且不应包装为正向证明。

## 因果边界

本轮证明了“最小 NDP 能以固定历史 bank 进入真实 EVA02 optimizer.step”，没有证明“选择性保护带来 LTCC 泛化收益”。N1 的局部正确关系增加可能来自关系重排，而非稳定识别结构；官方 Rank-1 在两个 seed 均下降，足以阻止继续堆叠复杂模块。mAP/mINP 的微小增加小于目前可由 PPU A/A 与两 seed 波动解释的范围，不能据此宣称收益。

邻域审计的 update400 配对 proxy 显示：seed1234 其它历史可靠负 PID 的平均 ΔMargin 为 +0.000789、变差比例 49.45%；seed4321 为 −0.000591、变差比例 52.42%。这不是单侧 NDP 的独立因果证明，但已足以要求下一轮先做逐 active relation 的邻域梯度预算审计。

失败/限制如实记录：首次官方评估尝试分别遇到 YACS 类型和错误 caption 路径，随后使用训练侧 Bio-only 构造及服务器实际 sidecar 路径修复；未加载错误模型、未产生训练结果。首次 seed1234 smoke 的诊断字段 NaN 已修复并以新目录重跑；失败目录保留。
