# NDP-V2 最终实验决策

## 决策

**对 10 epoch 的匹配 B0/N2 扩展：GO（条件性探索性 GO）。** 两 seed 在固定 update596 官方 LTCC 上均取得正的 Rank-1/mAP/mINP 差值，且 NDP 真正参与了训练；继续 10 epoch 有足够的实证价值。

**对直接 60 epoch 或论文级机制定论：NO_GO。** 当前只有两个 seed、一个 continuation epoch；局部 train-only relation/Purity 没有一致改善，seed1234 有明显局部退化；PPU A/A 约0.4%差异和一次 AMP skip 也要求更长程结果谨慎解释。

## 对用户关心问题的直接回答

1. **NDP 是否真正参与训练？** 是。Smoke 和正式 N2 都有非零 NDP loss/feature gradient；当前 live anchor 未 detach；模型参数执行了 optimizer step。N2 seed1234/4321 正式 epoch 分别有 423/386 条 active relation，激活 batch 率约45.9%/42.4%。
2. **NDP-V2 是否比 N1 更容易激活？** 是。相对 N1 约10.5%–13.5%的 active batch，V2 为约42.4%–45.9%；这是 2048 anchor 覆盖扩大后的直接结果。
3. **是否改善 LTCC Rank-1/mAP？** 在本轮匹配控制下，是：seed1234 Rank-1/mAP `+0.010204/+0.006955`；seed4321 `+0.005102/+0.004208`。两 seed 方向一致，但样本数不足以证明普遍有效。
4. **是否改善局部身份判别？** 不能下统一肯定结论。seed4321 的 aggregate relation rate 和当前错误 PID略好，seed1234 相反；两 seed Purity@10 都略降。NDP 可能改善官方检索所见结构，同时破坏部分固定邻域关系。
5. **是否存在明显邻近身份损害？** 有轻度但真实的 train-only 信号：seed1234 新增错误 PID 129 vs B0 97、Purity@10 下降约0.00132；seed4321 Purity@10 下降约0.00059。尚未达到停止探索的证据，但足以禁止直接扩到60 epoch。
6. **哪个 seed 更好？** seed1234 的官方增益更大（Rank-1 +1.02 pp、mAP +0.70 pp）；seed4321 增益较小但仍同方向（+0.51 pp、+0.42 pp）。
7. **是否值得继续10–60 epoch？** 值得继续 10 epoch 的配对探索；不建议直接跑60 epoch。10 epoch 必须继续使用同一 B0/N2、同 LR/采样和固定评估节点，并把局部 collateral 作为硬监测指标。
8. **下一次最该改哪里？** 不要继续扩大图、增加 Hubness/双侧分支或调 GPC。先针对 seed1234 的局部附带损害做“梯度预算/每 anchor 关系预算”诊断：比较 NDP 对目标负 PID和其他四个负 PID的一阶影响，优先验证是否是一个 anchor 的五 PID max 监督把梯度集中到错误方向。只有该因果证据成立，才考虑预注册的关系上限或局部安全门控。

## 状态标签

- `NDP_TRAINING_PARTICIPATION = YES`
- `NDP_V2_LTCC_EXPLORATORY_SIGNAL = POSITIVE_ON_2_SEEDS`
- `NDP_LOCAL_MECHANISM = INCONCLUSIVE`
- `NDP_MECHANISM = INCONCLUSIVE`
- `A/A_REPRODUCIBILITY = FAIL`（继承已有约0.4%差异，本轮未重新宣称修复）
- `CONTROLLED_10_EPOCH_PILOT = GO_CONDITIONAL`
- `CONTROLLED_60_EPOCH_TRAINING = NO_GO`

## 复现与证据路径

- 实验日志：[NDP-V2 日志目录](/home/DIFFER/log_NDP/ndp_v2_20261023_061358)
- 训练实现报告：`NDP_V2_IMPLEMENTATION.md`
- 对照报告：`NDP_V2_B0_COMPARISON.md`
- 逐关系 CSV：[seed1234](/home/DIFFER/log_NDP/ndp_v2_20261023_061358/local_metrics/seed1234/local_relation_metrics_seed1234.csv)、[seed4321](/home/DIFFER/log_NDP/ndp_v2_20261023_061358/local_metrics/seed4321/local_relation_metrics_seed4321.csv)
- 官方结果 JSON：[seed1234](/home/DIFFER/log_NDP/ndp_v2_20261023_061358/ltcc_seed1234_update596.json)、[seed4321](/home/DIFFER/log_NDP/ndp_v2_20261023_061358/ltcc_seed4321_update596.json)

本报告没有把未完成的 10/60 epoch 写成已完成，也没有把局部冲突证据隐藏在平均检索指标之后。
