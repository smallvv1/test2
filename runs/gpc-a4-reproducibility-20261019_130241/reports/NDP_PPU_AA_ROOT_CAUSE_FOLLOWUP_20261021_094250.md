# NDP PPU A/A 数值问题跟进

## 本轮动作

本轮未启动新的模型计算、反向或 optimizer-step；只读复核已有 PPU 算子级审计和 DDP JSON。原因是用户门禁明确禁止在 A/A 未通过时做 NDP 因果更新，本轮任务核心是安全的同轨迹离线标签重构。已有真实证据路径：

- `/home/DIFFER/reports/NDP_AA_OPERATOR_LEVEL_AUDIT_20261021_034200.md`
- `/home/DIFFER/reports/NDP_AA_OPERATOR_RECHECK_20261021_041500.md`
- `/home/DIFFER/outputs/ndp_aa_operator_audit_20261021_033800/`
- `/home/DIFFER/outputs/ndp_aa_operator_audit_20261021_041000/`
- `/home/DIFFER/outputs/ndp_ddp_gradient_audit_20261021_033042/`

## 已确认事实

1. 同 checkpoint、同物化 batch 的第1步 forward/loss一致。
2. 第1步未缩放参数梯度已不同；两次 `found_inf=0`、无 AMP skip，optimizer step均执行。
3. 首个可观测反向差异在 EVA02 `blocks.22.norm1` / `blocks.22.attn.q_proj` 链路；尚未定位到单个 norm、Q/K/V、attention matmul 或归约 kernel。
4. seed1234 选定参数相对 L2 差异约 `3.83e-3`，seed4321约 `4.53e-3`；余弦接近1但不等于逐元素复现。
5. rank-local 梯度在 all-reduce 前已不同；现有证据不支持把 DDP all-reduce称为首因。
6. 之前 float64 余弦审计已将大于1的问题缩小为约 `5.35e-14` 的浮点舍入，不代表 A/A 通过。

## 结论

`A/A_REPRODUCIBILITY=FAIL`。当前差异已经足以使小幅 NDP optimizer-step 效果无法归因。下一步若要解除门禁，必须在相同上游梯度下进一步隔离 norm、Q/K/V、attention matmul、softmax/归约的 PPU kernel；或预注册一个以参数/梯度误差为核心的数值噪声界，但在此之前不能进行 NDP 更新效果比较。

