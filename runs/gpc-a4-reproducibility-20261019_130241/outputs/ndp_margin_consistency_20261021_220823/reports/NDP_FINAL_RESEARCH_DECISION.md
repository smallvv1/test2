# NDP 当前阶段最终研究决策

## 已确认

1. 参考/当前 Margin 定义已统一，A/B/C 三种聚合严格分开；两个 seed 都有 512 anchor、2560 条真实 train-only K=5 多 PID 固定关系。
2. mean 主定义下 update0→200 反应式激活为 seed1234 `33/2560`、seed4321 `9/2560`，冻结特征梯度有限。
3. 双侧身份关系存在非对称变化：A单侧/B单侧/双侧恶化分别为 seed1234 `9/18/25`、seed4321 `13/16/18`。
4. 原始欧氏 SoftMargin Triplet 与反应式余弦候选的冻结特征梯度方向不同，但可靠+观察退化和可靠 active 集合等价，不构成两个机制。
5. 独立固定输入 LayerNorm probe 稳定；完整 EVA02 AMP A/A 仍有约0.00412相对梯度差，根因未唯一定位。

## 未证明

- 尚无历史真实 Batch-Hard 逐步 miner 记录；当前多层覆盖含 reconstructed sampler。
- Reliability 与 Vulnerability 的当前阈值存在逻辑互斥，不能支持“可靠且脆弱”的独立门控价值。
- 没有 NDP optimizer.step 与 B0 的因果比较，不能声称 Rank-1/mAP 或真实局部保护收益。
- 双侧样本仅 80/70 条转移，不能证明双侧训练增益。

## 门禁

`NDP_MECHANISM = INCONCLUSIVE`  
`A/A_REPRODUCIBILITY = FAIL`  
`CONTROLLED_SHORT_PILOT = NO_GO`

本轮没有修改 GPC-v1、Bio、CE、Triplet、EVA02、采样器或官方评估协议，也没有执行 optimizer.step。当前最小可保留方向是“统一 Margin 定义 + 固定多 PID 关系诊断”；反应式保护暂只保留为数学/冻结特征候选。下一步应先获取真实 miner 覆盖、拆开可靠性/脆弱性定义并继续定位完整模型 A/A，再申请任何受控短程训练。
