# NDP 下一阶段决策

## 状态

`NDP_ACTIVATION_VERIFIED` 仅指真实 Bio+GPC-v1 continuation 节点上的自然局部优势激活与无更新 PPU 参数梯度。

同时：

`NDP_NUMERICAL_REPRODUCIBILITY_BLOCKED`

`NDP_MECHANISM_INCONCLUSIVE`

`READY_FOR_CONTROLLED_SHORT_PILOT`：`NO`

## 当前结论

1. 阶段3故障已由变量名错误修复，两个 seed 均成功完成；没有重训。
2. 真实轨迹中可靠激活存在，update=400 为26/64和17/64；梯度有限且到达选定 EVA02 参数。
3. C1/C3 与自适应 margin Triplet 等价，不能作为独立 NDP 创新证据。
4. C2 detached cache 与 C3 live neighbor 梯度路径不同，但 C2 激活不稳定，尚无 optimizer-step 或检索效用证据。
5. 历史 AMP A/A 失败率仍使 B0/NDP 配对更新不可解释。

因此本轮不启动30/60 epoch，也不启动短程 optimizer-step。下一步应先做固定关系、固定缓存时点、固定 lambda 的 A/A 数值复核；若 A/A 门禁通过，再以 C2 或真正不同于 adaptive Triplet 的关系分布候选做小规模控制实验。
