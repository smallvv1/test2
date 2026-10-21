# NDP 真实 Triplet 机制对照：最终研究决策

## 总结决策

- `A/A_REPRODUCIBILITY = FAIL`
- `NDP_MECHANISM = INCONCLUSIVE`
- `CONTROLLED_SHORT_PILOT = NO_GO`
- `NDP_FORMULA_CORRECTION = PASS`
- `RAW_TRIPLET_FEATURE_ALIGNMENT = PASS_READONLY`
- `MULTILEVEL_COVERAGE = PASS_RECONSTRUCTED_NOT_HISTORICAL`
- `REACTIVE_FEATURE_GRADIENT = PASS_FIRST_ORDER_ONLY`

## 已验证事实

1. 原始 Triplet 是未归一化 `feat[0]` 上的欧氏 Batch-Hard + SoftMarginLoss。旧重建公式方向错误，现已与 PyTorch 精确一致。
2. 归一化检索 bank 不能默认代表真实 Triplet 输入；seed4321 的 raw/normalized hard negative 已发生变化。
3. 重建 sampler 的 PID 级机会/选择约 8%，但固定图像级完整 NDP 关系为 0；因此“精确图像关系 0”不能等价为“Triplet 完全没有身份监督”，也不能等价为“真实 miner 覆盖已证实”。
4. 首次翻转应只在 update200 正确关系中评价。当前 Margin 接近零的关系最能解释事件；pre-200 趋势跨 seed 不稳定，预测式 NDP 没有获得可靠证据。
5. 严格互反图像子集存在不少一侧 delta<0，但没有严格翻转，样本量太小，双侧保护尚无独立价值证明。
6. update200 冻结特征上的反应式梯度有限，但这不是网络参数梯度，也不是效果证据；其核心 Hinge 仍与参考 Margin Triplet 重叠。
7. 修复后的 LayerNorm/直接 norm1 探针稳定，但完整 EVA02 A/A 仍 FAIL，旧完整模型梯度差异没有被解释。
8. 本批最终回归测试为 `33 passed`；失败的早期 miner/reliability 进程日志保留，修复后的重跑日志分别标记为 `*_rerun.log`/`*_pre200.log`。

## 对核心问题的回答

### 原始 Triplet 真正覆盖了多少 NDP 关系？

按当前重建口径，PID 级 hard-negative 覆盖为 seed1234 `197/2560=7.6953%`、seed4321 `200/2560=7.8125%`；固定图像级机会为 `5/2560` 与 `0/2560`，完整关系选中均为 0。由于没有历史 miner 日志且输入特征曾有 raw/normalized 差异，这些数字只能是近似覆盖，不是实际梯度覆盖率。

### 历史可靠性和双侧多身份是否提供独立信息？

当前没有足够证据。历史最小 Margin 的高 AUC主要反映边界接近度；趋势不稳定，双侧样本只有 80/70 个图像级转移，双侧+可靠交集过小。反应式梯度与普通参考 Margin Hinge 的差异主要来自 mask，而非已证明的额外判别监督。

### 最小值得保留的机制

仅保留“固定历史关系、当前已发生退化时的单侧 Margin 诊断函数”作为研究接口；暂不命名为已成立的 NDP Loss。若继续，下一实验必须先记录真实 raw Triplet miner、解决完整 EVA02 同精度 A/A，并在相同关系预算下比较可靠关系 mask 与随机/当前 Hard Margin mask。

## 研究边界

本轮所有模型计算均为 PPU 只读前向或无 optimizer.step 探针；LTCC train-only；未使用 query/gallery/test；GPC-v1、CE、Triplet、Bio、EVA02 和历史 checkpoint 未修改。

## 版本与校验

- 远端 main 检查值：`0423469770dc90d85d6354667d4931d768c3235b`。
- `/data/DIFFER` 工作树 HEAD：`573b0502a231568d509fe3a463664c1a3c42e258`。
- 基线文件 SHA256：见 `RUN_MANIFEST.json` 和本批 `source_snapshot/SHA256SUMS.json`。
- 本批源码快照已保存；未复制 checkpoint、训练图片或完整特征库。

## 可审计命令

`/usr/local/bin/python3.10 -u scripts/ndp_triplet_formula_audit.py --output outputs/ndp_triplet_alignment_20261021_134148/triplet_formula_audit.json`
`/usr/local/bin/python3.10 -u scripts/ndp_triplet_raw_feature_alignment.py --seed 1234 --output-dir outputs/ndp_triplet_alignment_20261021_134148/seed1234 --epoch 60 --rank 0`
`/usr/local/bin/python3.10 -u scripts/ndp_triplet_raw_feature_alignment.py --seed 4321 --output-dir outputs/ndp_triplet_alignment_20261021_134148/seed4321 --epoch 60 --rank 0`
`/usr/local/bin/python3.10 -u scripts/ndp_real_triplet_miner_coverage.py --seed 1234 --features outputs/ndp_true_multipid_20261021_103952/full_features/seed1234/update0_full_features_seed1234.npz --graph outputs/ndp_true_multipid_20261021_103952/reference_graph/seed1234/multi_pid_reference_graph.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/miner_coverage/seed1234 --epochs 60 61`
`/usr/local/bin/python3.10 -u scripts/ndp_real_triplet_miner_coverage.py --seed 4321 --features outputs/ndp_true_multipid_20261021_103952/full_features/seed4321/update0_full_features_seed4321.npz --graph outputs/ndp_true_multipid_20261021_103952/reference_graph/seed4321/multi_pid_reference_graph.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/miner_coverage/seed4321 --epochs 60 61`
`/usr/local/bin/python3.10 -u scripts/ndp_reliability_incremental_audit.py --seed 1234 --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed1234/multi_pid_transition_rows.csv --reliability-rows outputs/ndp_true_multipid_20261021_103952/analysis/seed1234/reliability/reliability_vulnerability_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/reliability_incremental/seed1234`
`/usr/local/bin/python3.10 -u scripts/ndp_reliability_incremental_audit.py --seed 4321 --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed4321/multi_pid_transition_rows.csv --reliability-rows outputs/ndp_true_multipid_20261021_103952/analysis/seed4321/reliability/reliability_vulnerability_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/reliability_incremental/seed4321`
`/usr/local/bin/python3.10 -u scripts/ndp_bidirectional_additional_audit.py --seed 1234 --pairs outputs/ndp_true_multipid_20261021_103952/analysis/seed1234/bidirectional/bidirectional_pairs.csv --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed1234/multi_pid_transition_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/bidirectional_additional/seed1234`
`/usr/local/bin/python3.10 -u scripts/ndp_bidirectional_additional_audit.py --seed 4321 --pairs outputs/ndp_true_multipid_20261021_103952/analysis/seed4321/bidirectional/bidirectional_pairs.csv --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed4321/multi_pid_transition_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/bidirectional_additional/seed4321`
`/usr/local/bin/python3.10 -u scripts/ndp_reactive_feature_gradient_audit.py --seed 1234 --features outputs/ndp_multipid_reconstruction_20261021_101900/feature_export/seed1234/trajectory_features_seed1234.npz --groups outputs/ndp_future_boundary_20261021_094250/coverage_expansion/seed1234_retry1/fixed_diagnostic_groups.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/reactive_feature_gradient/seed1234`
`/usr/local/bin/python3.10 -u scripts/ndp_reactive_feature_gradient_audit.py --seed 4321 --features outputs/ndp_multipid_reconstruction_20261021_101900/feature_export/seed4321/trajectory_features_seed4321.npz --groups outputs/ndp_future_boundary_20261021_094250/coverage_expansion/seed4321/fixed_diagnostic_groups.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/reactive_feature_gradient/seed4321`
`/usr/local/bin/python3.10 -u scripts/ndp_ppu_same_precision_kernel_probe.py --output outputs/ndp_triplet_alignment_20261021_134148/ppu_same_precision_kernel_probe.json --repeats 3 --include-eva-norm`
`PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER /usr/local/bin/python3.10 -m pytest -q tests/test_ndp_*.py`
