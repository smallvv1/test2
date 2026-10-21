# NDP 下一阶段决策

## 状态

- `P3_METRIC_FIX=PASS`
- `P2_P5_DEFINITION_ALIGNMENT=FAIL_AS_SAME_METRIC`：两者已分别实现并审计，不能混称。
- `R3_HISTORY=NOT_ONLINE_VALID`
- `SAME_TRAJECTORY_RISK=PRELIMINARY_SUPPORTED_ON_64_GROUPS`
- `A/A_REPRODUCIBILITY=FAIL`
- `NDP_MECHANISM=INCONCLUSIVE`
- `CONTROLLED_SHORT_PILOT=NO_GO`

## 主要判断

1. 修复 ROC 后 R0/R1 的回顾性 endpoint 标签 AUC 很高，但它是用边界下降构造的事后标签，不能证明在线预测；R2–R5 没有稳定超过 R0。
2. P2 Soft Intrusion 会因安全负 PID 稀释、正样本 Q25 变化、sigmoid 饱和和单代表抽样漏掉关键边界翻转。P5 与 P2 数学不同，也不能直接继承 P2 结果。
3. R3 的独立终点 Jaccard 不具备时间历史资格；R4 seed 隔离后仍只是采样 hubness；R5 显示明显的方向不对称，但没有稳定方法收益。
4. 同轨迹64组的前一状态优势下降对下一节点激活有中等预测信号，但覆盖小、标签仍是优势阈值事件，不足以设计正式在线选择器。
5. PPU A/A 反向分叉仍在 block22 反向链，A/A 门禁未解除。

## 下一步

保留固定 Bio+GPC-v1；停止 R3/R4/R5 堆叠和 P2/P5 混合损失。若继续，只能先完成 PPU 反向算子根因或预注册可接受噪声边界，并扩大同轨迹的独立身份损害标签；在此之前不运行 NDP optimizer-step、30 epoch 或60 epoch。当前最小可保留机制是“关键局部边界的离线诊断”，不是已验证的 NDP 训练模块。
