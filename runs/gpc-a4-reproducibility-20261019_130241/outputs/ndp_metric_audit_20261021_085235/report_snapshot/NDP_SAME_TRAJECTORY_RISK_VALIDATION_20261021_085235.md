# 同轨迹风险验证

本轮复用了 `/home/DIFFER/outputs/ndp_overnight_recovery_20261021_024248/` 的固定 Bio+GPC continuation。两个 seed 均有 update 0/20/50/100/200/400；阶段3在 PPU 上 eval/no_grad 重新编码固定64组，没有执行 optimizer.step，关系集合和参考状态固定。

## 预更新预测

对每个 target 节点，只使用前一个或此前节点的优势下降，target 节点只用于生成评价标签。每个 seed 320 个 transition：0→20、20→50、50→100、100→200、200→400。

| seed | target 激活数/320 | 激活率 | 上一节点 drop AUC/AP | 历史最大 drop AUC/AP | drop slope AUC/AP |
|---:|---:|---:|---:|---:|---:|
|1234|114|35.625%|0.7263/0.6697|0.7554/0.6461|0.6451/0.5915|
|4321|91|28.438%|0.7880/0.6596|0.7680/0.6318|0.6677/0.4840|

阶段3节点激活数为 seed1234：0、22、22、20、13、37；seed4321：0、11、9、18、27、26。相同 checkpoint 重复提取此前最大绝对特征差为0、最小 cosine约0.999999821，说明节点间变化不是简单重复编码噪声。

## 限制

这是真正同一 continuation 的预更新时序证据，但只覆盖固定64组，不是9576张训练图像的全库风险预测；标签是 `advantage_drop>0.02`，尚未等同于独立的检索身份翻转标签。它说明局部优势阈值事件具有一定时间持续性，但没有证明 NDP 约束能减少身份损害，也没有产生 optimizer-step 因果证据。

结构化结果：`/home/DIFFER/outputs/ndp_metric_audit_20261021_085235/same_trajectory/`。
