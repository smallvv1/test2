# GPC 下一轮改进决策

审计 RUN_TIME：`20261019_130241`。

## 最终决策

1. **新版评估源码是否归档完整？** 是。修复源码、测试、逐文件 hash 对照、六模型 checkpoint/config/manifest hash、评估协议/Query/Gallery hash 和配对 JSON/CSV 已归档；不含密钥、权重、数据集。
2. **有效验证 PID 数量？** 11。693 Query、1256 Gallery；统计证据不足。
3. **A4 两种子的配对 95% CI？** 见 `GPC_A4_PAIRED_PID_CONFIDENCE_REPORT.md`：seed1234 Image ΔR1 `[-2.6287,+1.3598] pp`、Image ΔmAP `[-1.3786,+2.1554] pp`；PID ΔR1 `[-4.9612,+0.7137] pp`、PID ΔmAP `[-1.0351,+2.0837] pp`。seed4321 Image ΔR1 `[-2.6762,+4.4560] pp`、Image ΔmAP `[-3.0052,-0.2814] pp`；PID ΔR1 `[-4.0244,+2.8673] pp`、PID ΔmAP `[-2.7644,+0.1217] pp`。
4. **B0 旧新为什么变化？** 已确认 B0 也漂移；旧/新 wrapper、Phase7I phase hash、未记录 sampler/增强/DDP/各 rank RNG 状态不完全一致。不能指定单一根因，状态 `REPRODUCIBILITY_UNRESOLVED`。
5. **A4 旧新为什么变化？** 旧 batch-local selector 与新全局累计 selector 不同，且 phase hash/关系流不同；叠加 B0 基线漂移。不能单独归因，状态 `REPRODUCIBILITY_UNRESOLVED`。
6. **PID 均衡是否生效？** 训练 relation 暴露层面是：A4 CV/Gini 明显下降，pair budget 和 active rate 未增加；检索身份效果均衡未证明。
7. **A4 是否稳定收益？** 否。两个 seed 的 Rank-1/mAP 方向不一致或区间跨 0；A4 保留为 `A4_MIXED`，不进入正式 LTCC。
8. **A5 是否比 A4 更值得后续研究？** 仅作为探索性候选：已有 A5 两 seed/60 epoch 开发收益小且需要重新核对饱和一致性，当前仍无稳定证据，不能直接启动训练。
9. **是否需要新 GPC 机制？** 不需要。先保留原始 GPC-v1；现有 A4 机制梯度数据不完整，不能据此提出新机制；本阶段不加入 NDP。
10. **当前是否允许启动下一轮训练？** **不允许**。本轮只完成诊断；在补齐可重复性协议和预先冻结独立科学问题/控制组前，不启动新的 60 epoch 或正式 LTCC。

## 状态标签

`A4_MIXED`；`REPRODUCIBILITY_UNRESOLVED`；`MECHANISM_INCONCLUSIVE`；`NEXT_TRAINING_BLOCKED_BY_PROTOCOL`。
