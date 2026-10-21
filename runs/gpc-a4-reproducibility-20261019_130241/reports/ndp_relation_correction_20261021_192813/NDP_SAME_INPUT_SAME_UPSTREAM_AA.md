# 同输入、同上游的 A/A 复核

## 可接受的局部结论

两种子每个探针组都满足：输入哈希相同、上游梯度哈希相同、stride/shape 相同、参数初始哈希相同；比较状态为 `VALID_COMPARISON`。因此本轮不能再把随机上游输入造成的差异称为 kernel 非确定性。

## 尚未解除的门禁

该测试覆盖的是 LayerNorm 局部路径，而不是完整训练图。历史完整模型同 AMP A/A 仍有约 `relative L2 0.00410/0.00471` 的梯度差异，且此前差异涉及 block22/attention 链；本轮没有完成 q/k/v、attention matmul、softmax、DDP 归约的同精度闭环定位。因此：

`A/A_REPRODUCIBILITY=FAIL`

不能进行 NDP optimizer-step 因果效果比较。

## 证据

- Seed1234/4321 probe JSON: [`ndp_layernorm_probe.json`](/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/ndp_layernorm_probe.json)、[`ndp_layernorm_probe_seed4321.json`](/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/ndp_layernorm_probe_seed4321.json)
- 旧完整模型 A/A 对照仍在历史目录，未覆盖：`/home/DIFFER/outputs/ndp_multipid_reconstruction_20261021_101900/ppu_same_precision_aa/`
- 回归测试: [`pytest_ndp_regression.log`](/home/DIFFER/log_NDP/ndp_relation_correction_20261021_192813/pytest_ndp_regression.log)，23 passed。

本批次的实际命令保存在 `/home/DIFFER/log_NDP/ndp_relation_correction_20261021_192813/`；所有模型探针使用 `/usr/local/bin/python3.10`，PPU 环境为 `PPU_SDK=/usr/local/PPU_SDK`，离线梯度审计不执行 optimizer.step。
