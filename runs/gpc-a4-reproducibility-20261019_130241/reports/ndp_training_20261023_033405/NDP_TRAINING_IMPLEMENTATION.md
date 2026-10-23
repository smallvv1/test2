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

## 1. 实现与 Smoke Test

入口：[`scripts/ndp_e60_minimal_pair_runner.py`](/home/DIFFER/scripts/ndp_e60_minimal_pair_runner.py)。每个 batch 计算原始 base loss 和原 GPC loss，然后仅对当前 batch 中出现的 anchor 查 frozen reference bank；positive/negative endpoint 不需要同 batch 出现，只有 live anchor 特征保留 autograd。参考 Margin 用 `float(...)` 固定，当前 Margin 从 `feat[0]` 计算，NDP 梯度再与主损失相加后执行一次原训练 optimizer/scaler step。

Smoke 的真实 PPU 结果：

| seed | updates | 历史正确/候选 | selected | active | active batch | NDP mean | NDP feature grad | main feature grad | probe param grad | restore |
|---|---|---|---|---|---|---|---|---|---|---|
| 1234 | 20 | 1197/2560 | 2 | 1 | 5.00% | 0.00033830 | 0.01605 | 0.04284 | 0.09853 | PASS/PASS |
| 4321 | 20 | 1158/2560 | 15 | 4 | 10.00% | 0.00082362 | 0.00469, 0.00959 | 0.02286, 0.02237 | 0.04296, 0.14968 | PASS/PASS |

Smoke 中两 seed 均 20/20 `step_executed=True`、LR 为 `2e-7`、AMP scale 未下降、参数有限且无 NaN/Inf。seed1234 第12步 `ndp=0.006766`、`lambda_ndp*ndp≈0.000338`，总损失高于 main loss；seed4321 第2/7步均观察到非零 NDP feature/probe 参数梯度。因此 NDP 确实经过当前 EVA02 反向并参与 optimizer.step，而不是只记录日志。

新增 focused unit test：`tests/test_ndp_minimal_pair_math.py`，`2 passed`。它验证当前改善零惩罚、退化产生当前分支梯度，以及参考/当前共用同一 Margin 定义；未重复旧的 38 项测试。

结论：Smoke PASS。此前一次 seed1234 smoke 的诊断 NaN 是诊断窗口误用绝对 global step 的代码问题，未覆盖训练结果；修正后以新时间戳 `20261023_032320` 重跑通过。
