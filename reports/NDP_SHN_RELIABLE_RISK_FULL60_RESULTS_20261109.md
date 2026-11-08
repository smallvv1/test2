# NDP_SHN_RELIABLE_RISK：正式 60 epoch 实验报告

日期：2026-11-09  
仓库：`smallvv1/test2`  
正式 seed：`1234`  
训练设备：2 × PPU-ZW810E  
GPC：原代码、样本选择与权重保持不变，`lambda_gpc=0.1`

## 1. 本轮改动

本轮以 `NDP_SHN_RELIABLE` 为基础，只修改 train-only bank 辅助关系的筛选方式，保留实时 `SHN_MP` 主损失、身份中心、Top-3 困难负身份、8 Anchor、采样方式和 `lambda_ndp=0.001`。

- 候选跨 Batch 同身份正样本先通过原可靠性门控（身份中心支持度下限 0.55）。
- 对每个候选计算局部风险 margin：当前 Anchor 与 bank 正样本的相似度，减去当前学生实时负身份中心中的最大相似度。
- 以候选风险 margin 的固定 40% 分位点作为“适度风险”目标，选择接近该目标且可靠的关系，避免只选最简单正样本或最困难正样本。
- 贪心约束最多 2 个辅助关系，且 Anchor 和 PID 均不重复，修复旧版 Top-2 覆盖重复的问题。
- bank 正样本端点保持 detach，辅助项仍只乘固定 0.25；负身份始终来自学生当前 batch，bank 负中心不替代实时负样本。
- bank 按 epoch 从 train split 刷新；不使用历史最大 Margin、测试 query/gallery 或固定跨 epoch 目标。

该分支命名为 `NDP_SHN_RELIABLE_RISK`，GPC 实现及其正样本选择器未修改。

## 2. 三阶段正式结果

数值为换装协议的 Rank-1 / mAP，所有差值为百分点（pp）。`NDP` 列为本轮 `NDP_SHN_RELIABLE_RISK`。

| 协议 | Bio | Bio+GPC | Bio+GPC+NDP | GPC 独立提升 | NDP 独立提升 | 总提升 |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 83.57 / 52.48 | 84.38 / 54.19 | 84.58 / 54.59 | +0.81 / +1.71 | +0.20 / +0.40 | +1.01 / +2.11 |
| LTCC Clothes-Changing | 48.98 / 29.17 | 55.36 / 33.94 | 55.61 / 34.22 | +6.38 / +4.77 | +0.26 / +0.28 | +6.63 / +5.05 |
| PRCC Same-Clothes | 100.00 / 99.21 | 100.00 / 99.48 | 100.00 / 99.57 | +0.00 / +0.27 | +0.00 / +0.09 | +0.00 / +0.36 |
| PRCC Clothes-Changing C→A | 64.61 / 63.00 | 64.10 / 63.41 | 64.04 / 63.88 | -0.51 / +0.41 | -0.06 / +0.47 | -0.56 / +0.88 |

精确值（便于复核）：

- LTCC General：Bio `83.5699797/52.4799969`，Bio+GPC `84.3813360/54.1862303`，NDP `84.5841765/54.5899803`。
- LTCC CC：Bio `48.9795923/29.1698879`，Bio+GPC `55.3571403/33.9366870`，NDP `55.6122422/34.2189425`。
- PRCC Same-Clothes：Bio `100.0000000/99.2068555`，Bio+GPC `100.0000000/99.4786598`，NDP `100.0000000/99.5667005`。
- PRCC C→A：Bio `64.6062672/63.0032319`，Bio+GPC `64.0982211/63.4143003`，NDP `64.0417755/63.8848221`。

## 3. 训练、验证和完整性状态

- LTCC、PRCC 均完成 60/60 epoch，seed=1234；LTCC 35,760 steps，PRCC 33,060 steps。
- 两个正式运行均复用历史共同 Bio 初始化，未重新训练 Bio 或 Bio+GPC；`lambda_gpc=0.1`、`lambda_ndp=0.001`。
- Smoke Test 和 3 epoch train-only 验证通过；新选择器与旧可靠性排序产生不同候选，梯度有限且非零，CE/Triplet 无异常。
- 正式训练和 bank 刷新只使用 train split；训练期间未读取官方 query/gallery，官方评估仅在训练结束后执行。
- LTCC 最后 epoch：实时 SHN 4,736 条关系、bank 辅助 1,184 条关系；PRCC：实时 SHN 4,408 条、bank 辅助 1,102 条。bank 辅助可靠性仍为有限值（LTCC 约 0.771，PRCC 约 0.823），说明新筛选器没有退化成简单正样本排序。
- CE、Triplet、GPC、NDP 均保持有限值；AMP 只有少量正常 skip，没有中断、异常恢复或改变优化协议。
- 正式目录只保留单文件 `model_final.pth`、日志、CSV 和 JSON；未生成任何 `rank*state.pt`、`rank*state.pth` 或逐 Rank checkpoint。正式完成后训练器不再保留 `resume_latest.pth`。

## 4. 结果分析与论文建议

本轮在 LTCC CC 上相对固定 Bio+GPC 同时提升 Rank-1 `+0.26` pp 和 mAP `+0.28` pp；LTCC General 也同时提升。在 PRCC C→A 上 mAP 提升 `+0.47` pp，但 Rank-1 下降 `-0.06` pp；PRCC Same-Clothes 仅 mAP 有 `+0.09` pp，Rank-1 已饱和。因此，本轮相比旧版 reliable bank 更好地改善了 LTCC mAP，但不能宣称所有协议和指标都稳定提升。

作为对照，历史 `NDP_SHN_MP` 的 LTCC CC 为 `57.40/33.41`，仍有更高 Rank-1；本轮 `RISK` 为 `55.61/34.22`，mAP 更高但没有超过 SHN_MP 的 Rank-1。综合证据支持“实时 SHN 为主、可靠 bank 为弱辅助”这一低干扰方向，但不支持继续无止境堆叠相似 bank 或 Margin 变体。论文中建议保留固定 Bio+GPC、SHN_MP、SHN_RELIABLE 和本轮 RISK 的单 seed 对照，并如实报告 PRCC Rank-1 的轻微回落。

## 5. 复现路径

- LTCC 完整日志：`/home/DIFFER/log_NDP/ndp_shn_reliable_risk_full60_seed1234_20261108/LTCC/`
- PRCC 完整日志：`/home/DIFFER/log_NDP/ndp_shn_reliable_risk_full60_seed1234_20261108/PRCC/`
- LTCC 评估 JSON：上述目录下 `LTCC/eval_final.json`
- PRCC 评估 JSON：上述目录下 `PRCC/eval_final.json`
- 最终模型 SHA256：LTCC `b559d91efec4dde9ba965260e72e05e25a3880b6d78b81b3c369190913918e28`；PRCC `98da5b2d56de411a508864a521e959f4403f2d7977de56581d293ce8a34f53ae`
- Bio-only 与固定 Bio+GPC 的三阶段基线 JSON 沿用历史统一评估结果，未重新训练。
