# NDP 短程更新试验状态

本 RUN_TIME 未启动新的 NDP PPU 短程更新。原因是离线晋级门禁未通过：局部 margin 变化在两个 seed 方向相反，且已有历史 PPU NDP 原型只有 4 个候选/4 步，报告为结构接口可激活但训练证据不足；现有 A/A AMP 数值基线仍为 `AA_ROOT_CAUSE_UNRESOLVED`。

历史证据：

- 旧 NDP 开发报告的动态风险为 `NDP_RISK_PREDICTION_WEAK`；完整重复噪声未被单关系信号稳定超过。
- 旧结构原型的 N1/N2 接口可产生有限梯度，但真实 4 步中激活不足，不能判断局部保护效果。

本轮没有用“挑选有利 batch”“放宽 AMP 门禁”或额外训练来制造有效样本。短程 PPU 试验仅在以下条件同时满足后再申请：至少两个 seed、多个独立状态、固定更新数、A/A 噪声可接受、C1/C2/C3 关系预算匹配。

当前状态：`NDP_SHORT_UPDATE_NOT_AUTHORIZED`。

