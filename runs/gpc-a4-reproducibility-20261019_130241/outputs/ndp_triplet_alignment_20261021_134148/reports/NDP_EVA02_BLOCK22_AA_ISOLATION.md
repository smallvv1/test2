# NDP EVA02 完整/局部 A/A 数值复核

## 本轮固定输入探针

修复后的 PPU 探针在循环外一次性物化输入、上游梯度、参数，并逐次核对 SHA256。独立 LayerNorm 与真实 `EVA02 block22.norm1` 的 AMP/FP32、输入梯度和 affine 参数梯度重复均为 `max_abs=0`、`relative_l2=0`，参数未改变；旧的 LayerNorm 巨大差异标记为 `INVALID_UPSTREAM_CONTROL`，不能继续作为 kernel 不稳定证据。

本轮探针状态：`PASS_FIXED_INPUT_UPSTREAM_PROBE`，PPU `cuda:0`，torch `2.6.0`，重复 `3`。

## 门禁结论

这并没有解决完整 EVA02 训练图 A/A。历史完整模型同精度 AMP A/A 仍有 seed1234 相对 L2 约 `0.00410`、seed4321 约 `0.00471` 的参数梯度差异；该差异来自完整模块组合路径，尚未在固定上游梯度下缩小到 q/k/v、attention score/softmax、残差或 DDP 归约的唯一 kernel。因此：

`A/A_REPRODUCIBILITY = FAIL`

本轮没有执行 optimizer.step，也没有把局部 norm 稳定误报成完整模型通过。下一步必须继续做真实 block22 物化输入/上游梯度的组合路径隔离；在此之前不允许 NDP 效果比较。

## 可审计命令

`/usr/local/bin/python3.10 -u scripts/ndp_ppu_same_precision_kernel_probe.py --output outputs/ndp_triplet_alignment_20261021_134148/ppu_same_precision_kernel_probe.json --repeats 3 --include-eva-norm`
