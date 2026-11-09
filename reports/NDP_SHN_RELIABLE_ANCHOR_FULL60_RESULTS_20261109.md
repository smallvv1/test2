# NDP_SHN_RELIABLE_ANCHOR：正式 60 epoch 实验报告

日期：2026-11-09  
仓库：`smallvv1/test2`  
正式 seed：`1234`  
设备：2 × PPU-ZW810E  
GPC：代码、正样本选择和权重保持不变，`lambda_gpc=0.1`

## 1. 本轮方法

本轮以 `NDP_SHN_RELIABLE_RISK`/`NDP_SHN_RELIABLE` 为基础，修改的只有 train-only bank 辅助关系筛选，命名为 `NDP_SHN_RELIABLE_ANCHOR`：

- 保留实时 `SHN_MP` 主损失、身份中心、Top-3 困难负身份、8 Anchor、原有 batch 采样和 `lambda_ndp=0.001`。
- 保留跨衣正样本、身份中心支持度下限 `0.55` 和 bank 正端点 detach；bank 负身份不替代实时负身份。
- 风险统一改为 Anchor 视角的局部判别 margin：
  `sim(Anchor, bank-positive) - max sim(Anchor, current-batch negative identity center)`。
- 每个 Anchor 至少有 2 个可靠候选时，选择其按 margin 排序的第二低候选，跳过只有一个极端困难正样本的 Anchor；随后按局部 margin 和可靠性排序，最多保留 2 个且 Anchor/PID 均不重复。
- bank 辅助仍只乘固定 `0.25`，不引入新网络、分类器、历史最大 margin、跨 epoch 目标或测试集信息。

Smoke Test、3 epoch train-only 验证和新增选择器梯度测试均通过。选择器产生稀疏 bank 辅助关系，且 loss 对学生特征有有限、非零梯度。

## 2. 三阶段正式结果

数值为统一官方评估器下的 Rank-1 / mAP，差值单位为百分点（pp）。`NDP` 列为本轮 `NDP_SHN_RELIABLE_ANCHOR`。

| 协议 | Bio | Bio+GPC | Bio+GPC+NDP | GPC 独立提升 | NDP 独立提升 | 总提升 |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 83.57 / 52.48 | 84.38 / 54.19 | 84.18 / 52.14 | +0.81 / +1.71 | -0.20 / -2.05 | +0.61 / -0.34 |
| LTCC Clothes-Changing | 48.98 / 29.17 | 55.36 / 33.94 | 50.51 / 28.50 | +6.38 / +4.77 | -4.85 / -5.44 | +1.53 / -0.67 |
| PRCC Same-Clothes | 100.00 / 99.21 | 100.00 / 99.48 | 100.00 / 99.55 | +0.00 / +0.27 | +0.00 / +0.07 | +0.00 / +0.34 |
| PRCC Clothes-Changing C→A | 64.61 / 63.00 | 64.10 / 63.41 | 63.93 / 63.76 | -0.51 / +0.41 | -0.17 / +0.35 | -0.68 / +0.76 |

精确评估值：

- LTCC General：Bio `83.5699797/52.4799969`，Bio+GPC `84.3813360/54.1862303`，NDP `84.1785014/52.1410017`。
- LTCC CC：Bio `48.9795923/29.1698879`，Bio+GPC `55.3571403/33.9366870`，NDP `50.5102038/28.4991864`。
- PRCC Same-Clothes：Bio `100.0000000/99.2068555`，Bio+GPC `100.0000000/99.4786598`，NDP `100.0000000/99.5480596`。
- PRCC C→A：Bio `64.6062672/63.0032319`，Bio+GPC `64.0982211/63.4143003`，NDP `63.9288723/63.7603229`。

## 3. 完整性与训练状态

- LTCC、PRCC 均完成 `60/60` epoch，分别为 35,760 和 33,060 global steps；正式训练统一 seed `1234`。
- 两个正式运行均复用历史 Bio common initialization；没有重新训练 Bio 或 Bio+GPC，GPC 代码、采样和权重均未改变。
- LTCC 最后 epoch：实时 SHN 4,736 条、bank 辅助 1,184 条；NDP raw loss `0.447348`，学生 NDP/main feature-grad ratio `0.2510`，AMP skip `0`。
- PRCC 最后 epoch：实时 SHN 4,408 条、bank 辅助 1,102 条；NDP raw loss `0.278345`，学生 NDP/main feature-grad ratio `0.4301`，AMP skip `0`。
- 训练和 bank 刷新只使用 train split；`query_gallery_used_during_training=false`。
- 正式目录没有任何 `rank*state.pt`、`rank*state.pth` 或逐 Rank checkpoint；完成目录只保留单文件 `model_final.pth` 及必要日志/JSON。
- LTCC 模型 SHA256：`f17483a8bf109f1d7a7330aa37521d200241407d2fbcd0b703232f2662a1ba78`。
- PRCC 模型 SHA256：`4bdd06e92a68f29801fee6295998b38356cdd89534bc3f2007e03a54f1deadc1`。

## 4. 结论与论文建议

本轮虽然修正了 Anchor 视角风险定义并使选择器实际激活，但没有获得可信的跨数据集增益：LTCC CC 相对 Bio+GPC 的 Rank-1/mAP 分别下降 `4.85/5.44` pp；PRCC C→A 只有 mAP `+0.35` pp，同时 Rank-1 `-0.17` pp。因此，不能将本轮提升归因于新增 NDP，也不建议继续堆叠同类 bank/Margin 变体。

历史 `NDP_SHN_MP` 仍作为 NDP 对照保留（LTCC CC `57.40/33.41`、PRCC C→A `64.15/63.78`）；本轮只作为风险筛选修正的负结果证据。后续论文应以固定 Bio+GPC、SHN_MP、RELIABLE、RISK 及本轮 Anchor-view 对照为主，明确说明 NDP 的增益不稳定，并转向整理 GPC 的主要实验结论。

## 5. 复现路径

- 正式日志根目录：`/home/DIFFER/log_NDP/ndp_shn_reliable_anchor_full60_seed1234_20261109_restart/`
- LTCC：上述目录下 `LTCC/`，官方结果为 `LTCC/eval_final.json`。
- PRCC：上述目录下 `PRCC/`，官方结果为 `PRCC/eval_final.json`。
- 之前一次因性能优化而中止的早期运行保留在：`/home/DIFFER/log_NDP/ndp_shn_reliable_anchor_full60_seed1234_20261109/`；本报告只采用 restart 目录的完整 60 epoch 结果。
- Bio 与 Bio+GPC 的基线 JSON 沿用已有统一评估结果，没有重复训练。
