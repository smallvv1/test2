# A/A block22 反向路径复核

seed4321 的 block22 内部 AMP 双运行均完成，`found_inf=0`，无 AMP skip，GPC 首批关系数为 0。

首个可观测内部差异仍在 `blocks.22.norm1` / `blocks.22.attn.q_proj` 反向输出：

- `blocks.22.norm1` 最大绝对差约 `7.63e-6`；
- `blocks.22.attn.q_proj` 最大绝对差约 `9.54e-7`；
- block22 顶层输出梯度相同；
- 差异向早期 blocks 传播。

这与 seed1234 的路径一致，但仍未定位到 norm、q/k/v、attention matmul 或归约中的单个底层 PPU kernel。当前 A/A 仍 FAIL；不允许将后续 NDP 小幅差异解释为因果收益。

结果目录：`/home/DIFFER/outputs/ndp_aa_operator_audit_20261021_041000/`。
