# NDP_SR LTCC / PRCC 正式 60 Epoch 结果

本报告对应 NDP_SR 正式实验：seed=1234，LTCC 与 PRCC 均从项目既有共同初始化开始训练 60 epoch。训练和参考特征库只使用 train split；query/gallery 只在最终评估阶段使用。输出目录没有生成任何 `rank*state.pt`、`rank*state.pth` 或其他逐 Rank checkpoint。

## 方法与固定条件

基础损失、采样和 GPC 保持 Bio+GPC B0 不变：`CE + Triplet + clipBio + 0.1 * GPC-v1`，`lambda_gpc=0.1`，GPC context 为 `phase7i_relation_specificity`。共同初始化和冻结教师均复用既有正式文件，没有重新训练 B0。

NDP_SR 的关系和损失定义如下：

- 冻结 Bio+GPC 教师只负责选择可靠的 train-only anchor、正样本和负样本索引；教师 margin gate 为 `[0.02, 0.10]`。
- 学生端使用自己的当前 anchor、正端点和负端点计算局部排序 margin；学生参考特征库每 5 个 epoch 用当前学生重新提取一次。
- 每个 anchor 只选择当前学生 margin 最小的一条可靠关系，施加 `ReLU(0.02 - M_student)`，`lambda_ndp=0.005`。
- 正负端点和参考库均处于学生特征空间；只有当前 live anchor 保留梯度，修复 NDP_RANK 中学生 anchor 与冻结教师端点不在同一特征空间的问题。

固定输入：

- LTCC common init：`/data/DIFFER/outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth`
- PRCC common init：`/data/DIFFER/outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT.pth`
- LTCC B0 teacher：`/home/DIFFER/log_NDP/ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/model_final.pth`
- PRCC B0 teacher：`/home/DIFFER/log_NDP/ndp_nteach_current_b0_pair_20261031_125146/PRCC_B0_full/PRCC/model_final.pth`

代码 SHA256：

- `scripts/ndp_safe.py`: `800ff25a2106b128c10d7ce968d520151742ea87f934d7f980854b771d1fcaa7`
- `scripts/ndp_v5_prcc_runner.py`: `4008b9de0a15f7c5752fdcb662c5bcb01417b77c6ac0065ee411d9c8c1687fb9`
- `scripts/train_ndp_listwise_full60.py`: `c3a48ff8bbc2b99c80238fd6de442b23647153f0fda7d8d0f9630438cfe6515f`

## 官方评估结果

差值为 NDP_SR 减 Bio+GPC B0，单位为百分点。mINP 为仓库评估器不支持。

| 数据集 / 协议 | Bio+GPC B0 Rank-1 | NDP_SR Rank-1 | ΔRank-1 | Bio+GPC B0 mAP | NDP_SR mAP | ΔmAP |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 84.381 | 83.976 | -0.405 | 54.186 | 53.676 | -0.510 |
| LTCC Clothes-Changing | 55.357 | 54.847 | -0.510 | 33.937 | 32.143 | -1.794 |
| PRCC Same-Clothes | 100.000 | 100.000 | +0.000 | 99.479 | 99.502 | +0.023 |
| PRCC C→A | 64.098 | 62.913 | -1.185 | 63.414 | 63.567 | +0.153 |

官方评估的 Rank-5/Rank-10：LTCC General `90.669/92.698`，LTCC Clothes-Changing `69.643/75.000`；PRCC Same-Clothes `100.000/100.000`，PRCC C→A `71.578/75.162`。

最终模型 SHA256：LTCC `390c09a291d34203e3bcc86c4e9bf681f7f9ab988ba97657f0b4b68ea56973ae`；PRCC `da1e18d759f976fe0abd778b900791f03f7809b56251d9afd819930ebf2abf49`。

## 训练有效性与结论

两套正式训练均完成 60/60 epoch。最终 epoch 的 NDP/main feature-grad ratio：LTCC `0.180214`，PRCC `0.034105`；全程 AMP skipped steps：LTCC `15`，PRCC `14`。最终 epoch 的 active batches：LTCC `474/596`，PRCC `91/551`；NDP loss 分别为 `0.027839` 和 `0.005762`。NDP 在两套数据上均实际激活并产生梯度，没有出现全程关闭或异常 NaN。

NDP_SR 修复了特征空间一致性和静态关系使用问题，但本轮没有得到 LTCC 与 PRCC 同时具有说服力的增益：LTCC General/CC 均下降，PRCC C→A Rank-1 下降，只有 PRCC mAP 小幅上升。因此不继续堆叠相似 NDP 模块，后续优先完成固定 GPC 的论文实验和复现。

正式服务器日志目录：`/home/DIFFER/log_NDP/ndp_sr_full60_seed1234_20261101_010500/`。GitHub 中只归档小体积日志和评估 JSON，不提交约 1.2 GB 的最终模型文件。
