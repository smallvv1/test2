# NDP-V2 最终决策（含3 epoch结果）

## 最终结论

**不建议继续当前 NDP-V2 到 10–60 epoch；`CONTROLLED_10_EPOCH_PILOT = NO_GO`。** 1 epoch 的配对 LTCC 信号是正的，但 3 epoch 只在 seed4321保留、seed1234转负，均值接近零/略负；同时 update596 train-only 局部指标已有邻近身份附带损害。继续堆训练时长没有足够因果依据。

这不等于 NDP 已被证伪：它确实能训练且短程可能改善检索；但当前“2048 anchor + 5 PID max reactive protection”还不是可安全扩展的版本。

## 八个问题的直接回答

1. **NDP 是否真正参与训练？** 是。N2 smoke 和 1/3 epoch均有 live anchor 梯度、NDP loss 和参数更新。
2. **V2 是否比 N1 更容易激活？** 是。N1约10.5%–13.5% active batch；V2 1 epoch约42.4%–45.9%，3 epoch约40.75%–46.34%。
3. **是否改善 Rank-1/mAP？** 只在 1 epoch 两 seed 同向：seed1234 `+1.02/+0.70 pp`，seed4321 `+0.51/+0.42 pp`。3 epoch seed1234为 `-0.51/-0.33 pp`，seed4321为 `+0.77/+0.33 pp`，均值 Rank-1约`+0.13 pp`、mAP约`-0.004 pp`，不能称稳定改善。
4. **是否改善局部身份判别？** 未证明。update596 局部 margin/rate/Purity跨 seed不一致，Purity两 seed均略降；3 epoch没有相应全量局部导出，因此不做过度外推。
5. **是否损害附近身份？** 有真实轻度信号。seed1234 update596 N2 新增错误 PID=`129` vs B0=`97`、Purity=`0.970703` vs `0.972021`；seed4321 Purity=`0.967627` vs `0.968213`。这足以阻止当前版本长程扩展。
6. **哪个 seed 更好？** 短程 seed1234增益更大；3 epoch seed4321更好，seed1234变差，说明 seed依赖明显。
7. **是否值得继续10–60 epoch？** 当前版本不值得直接继续；10/60均 NO_GO。先做一个小范围、无网络训练的局部梯度预算/附带影响诊断，再决定新版本是否重新开短程训练。
8. **下一处具体改进？** 不是再扩大 anchor 或增加复杂分支。先按 anchor 分解五个负 PID 的 NDP 梯度和邻域一阶影响，确认 max-negative 监督是否把一个 anchor 的预算集中到错误方向；以此为因果证据，预注册每-anchor关系预算/局部 collateral gate，再做新的匹配 B0/N3短程实验。

## 状态标签

- `NDP_TRAINING_PARTICIPATION = YES`
- `NDP_V2_SHORT_EPOCH_SIGNAL = POSITIVE_ON_2_SEEDS`
- `NDP_V2_3_EPOCH_SUSTAINABILITY = FAIL_INCONSISTENT`
- `NDP_LOCAL_MECHANISM = INCONCLUSIVE`
- `NDP_MECHANISM = INCONCLUSIVE`
- `A/A_REPRODUCIBILITY = FAIL`
- `CONTROLLED_10_EPOCH_PILOT = NO_GO`
- `CONTROLLED_60_EPOCH_TRAINING = NO_GO`

## 证据路径

- 最终日志：[3-epoch日志](/home/DIFFER/log_NDP/ndp_v2_20261023_070201)
- 1 epoch日志：[短程日志](/home/DIFFER/log_NDP/ndp_v2_20261023_061358)
- 实现报告：[NDP_V2_IMPLEMENTATION.md](/home/DIFFER/reports/ndp_v2_20261023_070201/NDP_V2_IMPLEMENTATION.md)
- 对照报告：[NDP_V2_B0_COMPARISON.md](/home/DIFFER/reports/ndp_v2_20261023_070201/NDP_V2_B0_COMPARISON.md)
- 官方结果：[/home/DIFFER/log_NDP/ndp_v2_20261023_070201/ltcc_seed1234_update1788.json](/home/DIFFER/log_NDP/ndp_v2_20261023_070201/ltcc_seed1234_update1788.json)、[/home/DIFFER/log_NDP/ndp_v2_20261023_070201/ltcc_seed4321_update1788.json](/home/DIFFER/log_NDP/ndp_v2_20261023_070201/ltcc_seed4321_update1788.json)

本结论没有把未完成的10/60 epoch写成完成，也没有用1 epoch正结果掩盖3 epoch seed1234的负结果。
