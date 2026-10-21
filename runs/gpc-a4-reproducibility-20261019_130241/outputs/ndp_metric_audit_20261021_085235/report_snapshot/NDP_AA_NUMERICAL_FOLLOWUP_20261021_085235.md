# A/A 数值问题复核

本轮没有为追求 PASS 重启训练，也没有改变正式 AMP、GPC 或原始损失配置。复用并核对既有 PPU 算子级证据：同一第1步 forward/loss 相同，found_inf=0、无 AMP skip；第1步未缩放梯度已经不同，第2步 forward 开始分叉。

两个 seed 的首个可观测内部差异都在 EVA02 `blocks.22.norm1` / `blocks.22.attn.q_proj` 反向链。seed4321 复核中 norm1 最大绝对差约`7.63e-6`、q_proj约`9.54e-7`；另一份 seed1234 记录中 q_proj 输出梯度最大绝对差约`3.05e-5`。rank-local 梯度在 all-reduce 前已经不同，因此 DDP 不是首要方向性根因；底层 norm、Q/K/V、attention matmul 或归约 kernel 尚未精确定位。

旧梯度余弦大于1的问题来自低精度点积/范数计算；float64 重算后最大超界约`5.35e-14`，属于计算舍入，不能掩盖真实的逐层梯度差异。

结论：`A/A_REPRODUCIBILITY=FAIL`。目前没有可预注册的全训练噪声界限，不能做 NDP optimizer-step 因果比较；离线同轨迹无梯度审计不受此结论影响。
