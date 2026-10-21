# LayerNorm 探针随机性修正

## 结论

上一轮 `relative L2≈1.4` 不能作为 LayerNorm kernel 非确定性证据：旧代码的 `up=torch.randn_like(x)` 没有使用固定 generator。该旧结果已标记为 `INVALID_UPSTREAM_CONTROL`。

本批次在循环外物化 input、upstream、weight、bias，每次重复新建计算图并清空梯度，同时核对输入/上游哈希。Seed1234 和 Seed4321 的独立 LayerNorm、实际 EVA02 `blocks.22.norm1`，均在 PPU 的 AMP/FP32 路径通过固定输入比较；输入梯度和参数梯度均有限且重复差异为 0（报告中的 cosine 仅受浮点显示精度影响）。

- Seed1234 probe: `PASS_FIXED_INPUT_UPSTREAM_PROBE`，最大 forward 差 `0.0`，最大 input-gradient 差 `0.0`。
- Seed4321 probe: `PASS_FIXED_INPUT_UPSTREAM_PROBE`，最大 forward 差 `0.0`，最大 input-gradient 差 `0.0`。
- CPU 参考重复也为 0 差异。

这只撤回“LayerNorm 独立探针已证明不稳定”的旧判断，不能撤回完整模型 A/A FAIL；完整 EVA02 backward 的首个分叉仍未被本探针定位。

## 复现与证据

本批次的实际命令保存在 `/home/DIFFER/log_NDP/ndp_relation_correction_20261021_192813/`；所有模型探针使用 `/usr/local/bin/python3.10`，PPU 环境为 `PPU_SDK=/usr/local/PPU_SDK`，离线梯度审计不执行 optimizer.step。

- JSON: [`ndp_layernorm_probe.json`](/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/ndp_layernorm_probe.json)
- JSON: [`ndp_layernorm_probe_seed4321.json`](/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/ndp_layernorm_probe_seed4321.json)
- 原始日志: [`ndp_layernorm_probe.log`](/home/DIFFER/log_NDP/ndp_relation_correction_20261021_192813/ndp_layernorm_probe.log)

## 源码 SHA256

- `/home/DIFFER/scripts/ndp_ppu_same_precision_kernel_probe.py`: `cde27be76d28097a699ed8b1edba085a1a78c2fe4f0df36d3fbf60ba1203ce22`
- `/home/DIFFER/scripts/ndp_multi_pid_reference_graph.py`: `d1d56bb690001f992bb1889d28769234633f888f362a9a74977e1c48659b34ec`
- `/home/DIFFER/scripts/ndp_multi_pid_trajectory_eval.py`: `13c185ef98963e9abdce5af9ca75310c99a57e98ae24d622175be7000a5e92fd`
- `/home/DIFFER/scripts/ndp_bidirectional_relation_correction.py`: `310ae5eee79c59da4d585eda7f301ac2537c264469351a43c099afcd139f4366`
- `/home/DIFFER/scripts/ndp_reliability_vulnerability_information_audit.py`: `f8ea0edad3cfd3b07aa721eac48b01c788b7a7db29d4429511ecf8cc4b43540a`
- `/home/DIFFER/scripts/ndp_real_triplet_miner_coverage.py`: `f0814e323e75762191763c158af6ee3fde4ce179e1fcdad993294405f0a4640a`
- `/home/DIFFER/scripts/ndp_matched_nonredundancy_audit.py`: `de919bb2f825f9dd67595b6fcf1a4ebbba475d11850b2fdef5aafabcb8895655`
- `/home/DIFFER/tests/test_ndp_relation_correction.py`: `068af7edeffa79a09ad20d8b9150f5e61e5ed709e71fff5998ecd7d54d4ffa7e`
