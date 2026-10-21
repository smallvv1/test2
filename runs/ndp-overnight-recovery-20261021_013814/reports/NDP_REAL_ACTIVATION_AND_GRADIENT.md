# NDP 真实激活与梯度报告

既有无 optimizer.step 的 PPU 探针已通过：EVA02 forward/backward 可用，人工构造的局部优势退化可产生有限非零 NDP 梯度；参考分支停止梯度。

本轮没有进入真实 continuation 节点的 NDP forward/backward，因此：

- 真实自然 NDP 激活：`NOT_VERIFIED`；
- 真实 NDP 参数梯度：`NOT_MEASURED`；
- 人工退化梯度：不能替代自然激活证据；
- B0/NDP严格配对：未执行；
- 局部身份保护证据：没有。

由于没有自然激活集合，不能比较缓存 NDP、实时邻居 NDP、参考 margin Triplet 或 SupCon 控制，也不能评估跨衣学习损害。
