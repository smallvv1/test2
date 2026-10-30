# NDP_V2：LTCC / PRCC 短程配对实验结果

## 结论

本轮完成 seed=1234 的 LTCC、PRCC 各 5 个 raw epoch 训练和官方固定协议评估。NDP_V2 可以正常反向传播，训练无 NaN/Inf，且没有生成逐 Rank checkpoint；但没有形成足以启动正式 60 epoch 的稳定增益：LTCC Clothes-Changing Rank-1 下降 2.296 pp，PRCC C→A 仅上升 0.141 pp。因此本轮对 NDP_V2 做 **NO-GO：不扩展正式 60 epoch**。

PRCC 首次正式启动曾因 V2 跨衣正样本逐候选同步选择过慢而在首个 epoch 前停止。随后将其改为分块张量相似度计算，并用同一配置重新运行；该次重新运行完整完成 5 epoch，首次失败目录保留未覆盖。

## 方法与固定条件

- B0：`CE + Triplet + clipBio + 0.1 * GPC`。
- NDP_V2：B0 加选择性 margin hinge；只保留历史图中具有两个不同衣物正参考的关系，当前 detached margin 在 `(-0.05, 0.05)` 时才选中；负 margin 使用目标 `0.005`，正近边界 margin 使用目标 `0.02`。
- NDP 总权重：epoch 1–2 为 0，epoch 3 为 0.0005，epoch 4–5 为 0.001。
- 训练参考只来自 train split；未使用 query/gallery。
- seed=1234，PPU 双卡，模型权重 warm-start 后为两臂分别新建相同 optimizer/scheduler/scaler/RNG。该配对是相同模型权重起点，不是旧完整 optimizer state 的无缝续训。
- GPC、EVA02、CE、Triplet、Bio、sampler、学习率协议和评估器均未改动；GPC context 为 `phase7i_relation_specificity`，GPC 权重固定为 0.1。

## 官方评估结果

百分比单位；B0 复用同一模型权重 warm-start 的 5-epoch 配对结果，NDP_V2 为本轮重新训练结果。

| 数据集 / 协议 | B0 Rank-1 | B0 mAP | NDP_V2 Rank-1 | NDP_V2 mAP | Δ Rank-1 (pp) | Δ mAP (pp) |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 86.004 | 53.780 | 85.396 | 53.770 | -0.609 | -0.010 |
| LTCC Clothes-Changing | 58.163 | 32.665 | 55.867 | 32.984 | -2.296 | +0.319 |
| PRCC Same-Clothes | 100.000 | 99.361 | 100.000 | 99.360 | +0.000 | -0.001 |
| PRCC C→A | 63.731 | 63.433 | 63.872 | 63.477 | +0.141 | +0.044 |

评估器未提供 mINP，四个协议均记录为 `Unavailable`。PRCC C→A 的小幅提升不足以抵消 LTCC 的 Rank-1 下降，也未在独立 seed 上验证。

## 训练侧统计

| 数据集 | epoch | NDP raw loss | lambda | selected relations | active relations | active batches | AMP skip | effective updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LTCC | 1 | 0.010008 | 0 | 3786 | 2139 | 91.86% | 1 | 595 |
| LTCC | 3 | 0.009909 | 0.0005 | 3673 | 2118 | 90.94% | 0 | 596 |
| LTCC | 5 | 0.010102 | 0.001 | 3620 | 2090 | 91.19% | 0 | 596 |
| PRCC | 1 | 0.004126 | 0 | 822 | 335 | 38.29% | 0 | 551 |
| PRCC | 3 | 0.004339 | 0.0005 | 832 | 362 | 41.02% | 0 | 551 |
| PRCC | 5 | 0.004313 | 0.001 | 834 | 364 | 42.11% | 0 | 551 |

最后一轮 NDP 特征梯度 / 主特征梯度比约为 LTCC `0.0465`、PRCC `0.1472`；未观察到 NDP 梯度长期压过主损失。在线图规模为 LTCC 28,728 条关系、PRCC 53,688 条关系，均为 train-only。

## 实际运行命令

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m py_compile \
  scripts/ndp_safe.py scripts/ndp_v5_prcc_runner.py \
  scripts/train_ndp_listwise_full60.py

# Smoke（LTCC/PRCC 各 20 步）
torchrun --standalone --nproc_per_node=2 \
  scripts/train_ndp_listwise_full60.py --mode smoke \
  --dataset ltcc --seed 1234 --arm NDP_V2 --epochs 1 --smoke-steps 20 \
  --init-model /home/DIFFER/log_NDP/NDP_SAFE_FULL60_SEED1234_20261029_143733/LTCC/model_final.pth \
  --output-dir /home/DIFFER/log_NDP/ndp_v2_short_20261030_173957/LTCC_NDP_V2

torchrun --standalone --nproc_per_node=2 \
  scripts/train_ndp_listwise_full60.py --mode smoke \
  --dataset prcc --seed 1234 --arm NDP_V2 --epochs 1 --smoke-steps 20 \
  --init-model /home/DIFFER/log_NDP/NDP_SAFE_FULL60_SEED1234_20261029_143733/PRCC/model_final.pth \
  --output-dir /home/DIFFER/log_NDP/ndp_v2_short_20261030_173957/PRCC_NDP_V2_VEC_SMOKE

# 正式短程训练：LTCC 5 轮；PRCC 向量化修复后 5 轮
torchrun --standalone --nproc_per_node=2 scripts/train_ndp_listwise_full60.py \
  --mode train --dataset ltcc --seed 1234 --arm NDP_V2 --epochs 5 \
  --init-model /home/DIFFER/log_NDP/NDP_SAFE_FULL60_SEED1234_20261029_143733/LTCC/model_final.pth \
  --output-dir /home/DIFFER/log_NDP/ndp_v2_short_20261030_173957/LTCC_NDP_V2

torchrun --standalone --nproc_per_node=2 scripts/train_ndp_listwise_full60.py \
  --mode train --dataset prcc --seed 1234 --arm NDP_V2 --epochs 5 \
  --init-model /home/DIFFER/log_NDP/NDP_SAFE_FULL60_SEED1234_20261029_143733/PRCC/model_final.pth \
  --output-dir /home/DIFFER/log_NDP/ndp_v2_short_20261030_180855/PRCC_NDP_V2

# 固定官方评估
torchrun --standalone --nproc_per_node=1 scripts/train_ndp_listwise_full60.py \
  --mode eval --dataset ltcc --seed 1234 --arm NDP_V2 \
  --output-dir /home/DIFFER/log_NDP/ndp_v2_short_20261030_173957/LTCC_NDP_V2
torchrun --standalone --nproc_per_node=1 scripts/train_ndp_listwise_full60.py \
  --mode eval --dataset prcc --seed 1234 --arm NDP_V2 \
  --output-dir /home/DIFFER/log_NDP/ndp_v2_short_20261030_180855/PRCC_NDP_V2
```

## 文件与可复现性

- LTCC 训练完成标记：`LTCC_NDP_V2/LTCC/SHORT_RUN_COMPLETE.json`。
- PRCC 训练完成标记：`PRCC_NDP_V2/PRCC/SHORT_RUN_COMPLETE.json`。
- 正式 60 epoch 未执行，因此没有 `RUN_COMPLETE.json`。
- LTCC 最终模型：`LTCC_NDP_V2/LTCC/model_final.pth`，SHA256 `a7154b1dd906b671d64b1b203ab56524904e731284f634315b65c4cfaa840c28`。
- PRCC 最终模型：`PRCC_NDP_V2/PRCC/model_final.pth`，SHA256 `2d777892b072c28ef036de951b952d463078e3cfebe860229446244cf12bcd4a`。
- `scripts/ndp_safe.py` SHA256 `65d96ed9415db9c3c210410b21d6ee398446d91eaf02f437b5124fc8230f8f09`。
- `scripts/ndp_v5_prcc_runner.py` SHA256 `7cda8469de2b40ebc15fa103e79a380ff27c7ad8f3c1343f53d2a25c673fe896`。
- `scripts/train_ndp_listwise_full60.py` SHA256 `c0474c1fce99cce5da70642370bcea2ef3a9134fb8a75af0ea09ca50d93d6f68`。
- 新实验目录扫描结果：未发现 `rank*state.pt`、`rank*state.pth` 或其他逐 Rank checkpoint。
- 所有参考特征和邻域均从 train-only 生成；没有 query/gallery 训练或调参。

## 下一步

当前不值得直接投入正式 60 epoch。应保留 B0 为固定基线，保留 NDP_V2 的完整失败/混合结果；下一步若继续，优先在训练集验证划分上检查跨衣正样本可靠性和局部关系增量覆盖，而不是继续扫描 margin、lambda 或扩大训练轮数。
