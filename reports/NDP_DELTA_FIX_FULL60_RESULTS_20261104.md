# NDP_DELTA_FIX 正式实验报告（seed=1234）

## 结论

NDP_DELTA_FIX 已完成 LTCC、PRCC 各 60 epoch 和官方评估。该版本修复了 DELTA 的关系绑定和目标累积问题，并且在 PRCC 上取得小幅提升；但 LTCC 明显下降，因此不能作为跨数据集的可靠增量方法。按用户要求，保留真实结果，不继续堆叠相似 NDP Margin 变体，后续优先整理固定 GPC 论文实验。

## 方法与公平性

- GPC 代码、正样本选择、匹配预算和权重 `0.1` 未修改；正式 GPC 源 SHA-256 为 `69ce0b4883df501d4f85f8732afb80ca2f8a7faa3f9046d40d941e1813268156`。
- 共同初始化来自既有 Bio 初始化；LTCC state hash 为 `c593b713f631fcc78087095d160b62de9ef3196cb7cefd0495d0daa62cc83202`，PRCC state hash 为 `6567bd0d54e6919be355757c5a2d7fb2e5d1c2128368357e51b7047f5ded97fc`。
- 风险 key 绑定 `GPC state_key + negative PID`，margin target 固定为该关系首次真实下降前的值；关系只消费一次，每批最多 1 条、每 epoch 最多 16 个保护批次。风险探测、邻域和评估协议均不读取官方测试集。
- Smoke 两套通过；正式训练未出现非有限 loss。LTCC 共 4,395 次探测/923 次下降/1 个保护批次；PRCC 共 326,750 次探测/92,682 次下降/873 个保护批次。PRCC 的 NDP 梯度持续非零；LTCC 仅零星激活。
- 训练只使用两张 PPU。正式目录只保留 `model_final.pth`（未上传归档）及运行期间的单文件 `resume_latest.pth`；完成后按协议清理 resume。未生成任何 `rank*state.pt/pth` 文件。

## 官方结果

百分比单位；括号内为相对固定 Bio+GPC 的百分点变化。

| 数据集 / 协议 | 固定 Bio+GPC Rank-1 | 固定 Bio+GPC mAP | NDP_DELTA_FIX Rank-1 | NDP_DELTA_FIX mAP |
|---|---:|---:|---:|---:|
| LTCC General | 84.381 | 54.186 | 84.179 (-0.202) | 52.407 (-1.779) |
| LTCC CC | 55.357 | 33.937 | 50.000 (-5.357) | 29.050 (-4.887) |
| PRCC Same-Clothes | 100.000 | 99.479 | 100.000 (+0.000) | 99.566 (+0.088) |
| PRCC C→A | 64.098 | 63.414 | 64.409 (+0.310) | 64.192 (+0.778) |

## 归因与后续决定

PRCC 的提升与真实激活的 exact-risk 保护同时出现，但 LTCC 的大幅下降说明该机制没有得到跨数据集验证；不能将 PRCC 单独的小增益写成稳定方法增益。尤其 LTCC 的 NDP 介入极少，无法解释其下降，说明当前版本仍不适合作为论文主方法。保留本实验作为失败/边界证据，停止继续增加同类 NDP Margin 约束，转向固定 GPC 的论文实验、复现和机制分析。

## 归档

- 正式小日志：`runs/ndp-delta-fix-full60-seed1234-20261104/`
- Smoke/短程日志：`runs/ndp-delta-fix-validation-20261104/`
- 完整模型仍保留在服务器：`/home/DIFFER/log_NDP/ndp_delta_fix_full60_20261104/{LTCC,PRCC}/model_final.pth`。
