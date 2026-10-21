# NDP PPU 反向 Kernel 隔离状态

## 已有证据复核

本轮执行的是只读复核命令：`cat outputs/ndp_future_boundary_20261021_094250/ppu_aa_followup_summary.json` 并读取两个历史算子审计报告；没有新增 PPU backward 进程，避免把禁止的效果试验混入本轮。

`A/A_REPRODUCIBILITY=FAIL` 仍然成立。两个 seed 的第 1 步 forward/loss 一致，第 1 步未缩放 backward 已分叉，found_inf=0、AMP skip=0；差异在 EVA02 `blocks.22.norm1` / `blocks.22.attn.q_proj` 反向路径首次可观测，尚未缩小到 norm、Q/K/V、attention matmul、softmax 或某个归约 kernel。

已有最大差异约为：

- norm1 输出梯度绝对差 `7.63e-6`；
- q_proj 输出梯度绝对差约 `3.05e-5`（另一复核约 `9.54e-7`）；
- 选定参数未缩放梯度相对 L2：seed1234 `3.83e-3`，seed4321 `4.53e-3`；
- 双 PPU DDP 的 rank-local gradient 也不同，但没有证据显示 all-reduce 创造了方向错误。

本轮补做了无 optimizer.step 的安全探针：两个 seed 均分别完成 AMP 与隔离 FP32 的 forward/backward，`found_inf=0`，四次 `optimizer_step_executed=false`。这是跨精度对照，不是同一精度 A/A 重复；它显示 AMP 与 FP32 的梯度存在明显差别，但不能单独定位底层 kernel，也不能替代同精度 A/A。

输出：`outputs/ndp_future_boundary_20261021_094250/ppu_aa_no_step/seed{1234,4321}/{amp,fp32}.json` 及对应梯度张量。seed1234 选定 `blocks.0.attn.q_proj.weight` 的 AMP/FP32 未缩放梯度相对差约 0.182，seed4321 约 0.494；这说明精度路径会改变梯度，但不证明两次 AMP A/A 的差异来源。用户门禁仍要求同精度重复性，因此 A/A 继续 FAIL。

证据报告：

- `reports/NDP_AA_OPERATOR_LEVEL_AUDIT_20261021_034200.md`；
- `reports/NDP_AA_OPERATOR_RECHECK_20261021_041500.md`；
- `outputs/ndp_future_boundary_20261021_094250/ppu_aa_followup_summary.json`。

## 门禁

`A/A_REPRODUCIBILITY=FAIL`。在未定位底层算子或预注册可接受噪声界限前，不允许将反应式 NDP 接入 optimizer，也不能进行短程因果收益比较。
