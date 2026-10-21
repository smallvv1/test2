# NDP 下一阶段决策

## 三项最终结论

- `A/A_REPRODUCIBILITY: FAIL`
- `NDP_INDEPENDENT_MECHANISM: INCONCLUSIVE`
- `CONTROLLED_SHORT_PILOT: NO_GO`

## 依据

1. 两个 seed 的严格 A/A 都满足相同物化输入、相同 GPC 关系、相同初始完整状态、
   20/20 有效更新和 0 skip，但第1步出现梯度/optimizer/参数状态差异；因此不能把
   NDP 的小幅更新结果与数值噪声分开。
2. 固定 B0 continuation 已观察到 update20/50/100/200/400 的自然可靠激活，且
   update400 同状态重复重编码稳定；这支持“存在可观察训练状态”，不等于“加入 NDP
   后改善身份学习”。
3. LRD 候选已具备独立于单自适应 margin 的多邻居概率结构和可反驳的单测反例，
   但尚无 optimizer-step 后的 NDP 对照证据，也没有证明它优于 Triplet/SupCon。

## 当前允许与禁止

允许：继续修复 PPU backward/A-A 数值路径；在不更新模型的条件下做 LRD 数学和
PPU 梯度探针；冻结 C0-C4 的关系、预算和控制协议。

禁止：在 A/A 门禁前执行 NDP optimizer-step 效果比较；启动 30/60 epoch；读取官方
LTCC query/gallery 选择参数；修改 GPC-v1。

下一步最值得做的是定位 PPU backward 首个非确定算子或建立可接受的共同精度控制，
然后仅做少量、完整记录的 C0/C4 配对更新。若 A/A 仍 FAIL，应维持
`NDP_INDEPENDENT_MECHANISM=INCONCLUSIVE`，不进入正式训练。

