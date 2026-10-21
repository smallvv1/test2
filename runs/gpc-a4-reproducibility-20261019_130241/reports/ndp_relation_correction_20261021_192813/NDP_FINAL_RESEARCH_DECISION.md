# NDP 关系修正批次最终决策

## 已验证

1. 旧 LayerNorm 大差异由上游随机张量未固定造成，旧证据无效；固定输入/参数/上游后，两个 seed 的通用 LayerNorm 和 EVA02 block22.norm1 PPU AMP/FP32 探针稳定。
2. 双侧身份关系不再 last-write-wins；Seed1234/4321 分别得到 80/70 个严格图像级互反对，A→B 与 B→A 的改善/恶化四象限仍不对称。
3. 反向 KNN 已真实计算所有候选负图像（2153/2112），缺失查询不再默认 False。
4. 原始 Triplet 是欧氏 Batch-Hard + SoftMarginLoss（NO_MARGIN），不是此前 cosine ReLU 代理。
5. 在重建采样审计中，2560 条固定 NDP 关系两 seed 的共同 batch、完整 miner 选中和 active Triplet 直接覆盖均为 0；398/512 anchor 有其他关系的 Triplet 约束。

## 未验证/失败

- 没有历史逐步 batch/miner 日志，不能把重建覆盖写成真实训练梯度覆盖。
- Reliability/Vulnerability 分类存在逻辑耦合，尚无独立增量价值证据。
- 双侧关系样本量有限，不能证明双侧保护优于单侧。
- PPU 完整模型同精度 backward 根因仍未定位；LayerNorm 修复不等于 A/A PASS。
- 没有执行 NDP optimizer.step、短程效果或正式训练。

## 门禁

`A/A_REPRODUCIBILITY=FAIL`

`NDP_MECHANISM=INCONCLUSIVE`

`CONTROLLED_SHORT_PILOT=NO_GO`

本批次没有改变 GPC、CE、Triplet、Bio、EVA02、采样协议或评估协议，也没有使用官方 query/gallery/test。

本批次的实际命令保存在 `/home/DIFFER/log_NDP/ndp_relation_correction_20261021_192813/`；所有模型探针使用 `/usr/local/bin/python3.10`，PPU 环境为 `PPU_SDK=/usr/local/PPU_SDK`，离线梯度审计不执行 optimizer.step。

### 代码 SHA256

- `/home/DIFFER/scripts/ndp_ppu_same_precision_kernel_probe.py`: `cde27be76d28097a699ed8b1edba085a1a78c2fe4f0df36d3fbf60ba1203ce22`
- `/home/DIFFER/scripts/ndp_multi_pid_reference_graph.py`: `d1d56bb690001f992bb1889d28769234633f888f362a9a74977e1c48659b34ec`
- `/home/DIFFER/scripts/ndp_multi_pid_trajectory_eval.py`: `13c185ef98963e9abdce5af9ca75310c99a57e98ae24d622175be7000a5e92fd`
- `/home/DIFFER/scripts/ndp_bidirectional_relation_correction.py`: `310ae5eee79c59da4d585eda7f301ac2537c264469351a43c099afcd139f4366`
- `/home/DIFFER/scripts/ndp_reliability_vulnerability_information_audit.py`: `f8ea0edad3cfd3b07aa721eac48b01c788b7a7db29d4429511ecf8cc4b43540a`
- `/home/DIFFER/scripts/ndp_real_triplet_miner_coverage.py`: `f0814e323e75762191763c158af6ee3fde4ce179e1fcdad993294405f0a4640a`
- `/home/DIFFER/scripts/ndp_matched_nonredundancy_audit.py`: `de919bb2f825f9dd67595b6fcf1a4ebbba475d11850b2fdef5aafabcb8895655`
- `/home/DIFFER/tests/test_ndp_relation_correction.py`: `068af7edeffa79a09ad20d8b9150f5e61e5ed709e71fff5998ecd7d54d4ffa7e`

### 归档

- 当前输出: `/home/DIFFER/outputs/ndp_relation_correction_20261021_192813`
- 报告目录: `/home/DIFFER/reports/ndp_relation_correction_20261021_192813`
- 回归测试: 23 项通过；历史报告和失败结果保留，未覆盖。
