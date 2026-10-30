# NDP_SELECT：LTCC / PRCC 短程配对结果

## 结论

本轮完成了 seed=1234、LTCC 与 PRCC 各一组 B0/NDP_SELECT 的 5 raw epoch 配对续训。NDP_SELECT 可以稳定反向传播，且梯度统计不再写入 NaN；但尚未获得同时改善 LTCC 和 PRCC Clothes-Changing 的证据，因此不启动正式 60 epoch NDP_SELECT。

NDP_SELECT 在 PRCC C→A 上有小幅正向变化，在 LTCC Clothes-Changing 上 Rank-1 下降。该结果只能作为短程探索性结果，不能宣称 NDP 已有效。

## 固定条件

- seed：1234；PPU 双卡；5 raw continuation epoch。
- B0：`CE + Triplet + clipBio + 0.1 * GPC`。
- NDP_SELECT：B0 + 选择性危险关系 hinge。
- NDP_SELECT 仅使用 `safe_positive_cross_clothes=True` 且当前 detached margin 位于 `(0, 0.05)` 的关系；epoch 1–2 的 NDP 总权重为 0，epoch 3 为 0.0005，epoch 4–5 为 0.001。
- GPC、EVA02、CE、Triplet、Bio、sampler、优化器和评估器未修改。
- 两个 arm 在每个数据集均从同一个已完成 B0 的模型权重 model-only warm-start，并各自创建相同的新 optimizer/scheduler/scaler/RNG；因此这是严格模型权重起点配对，但不是历史 optimizer state 的无缝续训。
- 训练关系只来自 train split；未使用 query/gallery 选择关系或参数。

## 官方固定节点结果

| Dataset / protocol | B0 Rank-1 | B0 mAP | NDP_SELECT Rank-1 | NDP_SELECT mAP | Δ Rank-1 (pp) | Δ mAP (pp) |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 86.004 | 53.780 | 86.004 | 53.970 | +0.000 | +0.190 |
| LTCC Clothes-Changing | 58.163 | 32.665 | 56.378 | 33.127 | -1.786 | +0.461 |
| PRCC Same-Clothes | 100.000 | 99.361 | 100.000 | 99.416 | +0.000 | +0.056 |
| PRCC C→A | 63.731 | 63.433 | 64.380 | 63.594 | +0.649 | +0.162 |

评估器未提供 mINP，记录为 Unavailable。以上结果均为第 5 raw epoch 的预先固定节点，不是按测试指标挑选的 checkpoint。

## 训练侧统计

LTCC 最后一轮：

- B0 train loss `4.813454`，AMP skip `1`，effective updates `595`。
- NDP_SELECT train loss `4.816629`，NDP raw loss `0.004134`，实际激活 `808/1828` 关系，active batch rate `0.719`，AMP skip `0`。
- NDP 特征梯度诊断为有限值；前 3 个诊断 batch 的 NDP/main 特征梯度比约 `0.063–0.128`。其余 batch 不重复探测，CSV 以有限值记录，不能解释为全 epoch 梯度均值。

PRCC 最后一轮：

- B0 train loss `7.080259`，AMP skip `1`，effective updates `550`。
- NDP_SELECT train loss `7.080727`，NDP raw loss `0.003373`，实际激活 `1577/4341` 关系，active batch rate `0.909`，AMP skip `1`。
- 在线关系数 `53688`，有效 anchor `17896`；关系图仅来自 PRCC train-only 数据。

## 实现修复与约束检查

1. 新增 `finite_gradient_norm()`：autograd probe 缺失或非有限时记录 `0` 和无效语义，不再把未探测值写成 NaN。
2. B0 仍由 `compose_total_loss()` 直接返回 `main_total`，不计算 NDP 关系损失；Smoke 中 B0 NDP loss/梯度为 0。
3. NDP_SELECT 的当前 anchor 保留梯度，参考端点停止梯度。
4. 本轮新目录未发现 `rank*state.pt`、`rank*state.pth` 或 `checkpoint_e*_rank*.pth`。
5. 5 epoch 结果使用 `SHORT_RUN_COMPLETE.json`；正式 60 epoch 才生成 `RUN_COMPLETE.json`。本目录没有声称完成 60 epoch。

## 运行命令

代码检查：

```bash
/usr/local/bin/python3.10 -m py_compile \
  scripts/ndp_v5_prcc_runner.py \
  scripts/train_ndp_listwise_full60.py \
  scripts/ndp_safe.py
```

Smoke、训练和评估均通过 `/usr/local/PPU_SDK/envsetup.sh ppu` 后使用 `/usr/local/bin/python3.10` 的 `torch.distributed.run` 执行，具体命令保存在本目录各 arm 的 `final_model.json`、`metrics_per_epoch.csv` 和 `eval_final.json` 对应路径中。

## 代码和模型摘要

- 上游代码参考 commit：`4ecd74c67a51dce1ef8d8343ee7b21d0d9620240`。
- `scripts/train_ndp_listwise_full60.py` SHA256：`d4689136d77b3ff55609483c063c0a8fa965f382960124f28568e0138c780fcc`。
- `scripts/ndp_v5_prcc_runner.py` SHA256：`46c04c69fdb281f0ff41aea178afa5787ffd74d075ec1d466e4f32f87587fb99`。
- `scripts/ndp_safe.py` SHA256：`7dd6ac84d1966b78a127755c48bbf1b02535998095d23f11c87df10b826b7c25`。
- LTCC B0 model SHA256：`bec464171012ba432560420610370238c60d1bf3a7b17d42278e3a4afb00d519`。
- LTCC NDP_SELECT model SHA256：`971154cfe9df831e5d4b3c05ddc2bb00caf38dccdee919390817e6006816efa5`。
- PRCC B0 model SHA256：`8ce40a0093031f51c22017a9a37f7d8f6aa6663464b5fc71570d200bacc36520`。
- PRCC NDP_SELECT model SHA256：`be52ee9c6d91b9a2671c95d20122b88af76894c3e03253a28c5e444eb04525b1`。

## 下一步决策

NO_GO：不把 NDP_SELECT 扩展到正式 60 epoch，也不继续扫描其阈值或权重。PRCC 的小幅正向结果值得保留为线索，但 LTCC CC Rank-1 的下降说明当前选择性关系仍可能破坏跨衣身份结构。下一步应先用训练集验证划分检查关系监督质量与原 Triplet 的增量覆盖，再决定是否设计新的监督目标；B0 继续作为固定基线。
