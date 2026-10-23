# EVA02 完整 A/A 反向首个可观测差异

本轮不重复孤立 LayerNorm probe，复用固定输入、上游梯度、E60 checkpoint、AMP、dtype/stride 和独立进程的完整 A/A。

| seed | 最大已观测参数张量 | relative L2 | max abs | cosine(float64) |
|---|---|---:|---:|---:|
| 1234 | scaled_gradients/blocks.0.attn.proj.weight | 0.0041188509 | 4.5 | 0.999992491987 |
| 4321 | scaled_gradients/blocks.0.attn.proj.weight | 0.0041252099 | 1.84375 | 0.999991496742 |

最大差异仍位于早期 `blocks.0.attn.proj.weight`（约0.00412/0.00413）；block22 q_proj 约8.6e-6，输出投影为零差异。独立 LayerNorm/norm1 probe 稳定，不能把 blocks.22.norm1 hash 变化认定为根因。
当前只定位到完整组合 backward/传播路径，尚未对 block22 Attention、Q/K/V、softmax、matmul、输出投影、残差、跨 Block、AMP kernel、DDP all-reduce 给出逐级首差异。因此 `A/A_REPRODUCIBILITY=FAIL`，不改变 A/A 决策。

统计不确定性：这是固定输入/固定上游梯度的重复进程数值比较，不是随机抽样；相对 L2 为探针精确差异，不能转换成总体误差概率。

## 可复现元数据

- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，不伪造本地 commit。
- 固定 baseline SHA256：`gpc_v2_train.py=d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`; `make_loss.py=9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043`; `eva02_l_bio.yml=c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97`; `triplet_loss.py=86656d44255172866d1e9854880c16e5d9536c6a756a577876312b663512092a`。
- checkpoint SHA256：`seed1234=e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`; `seed4321=d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7`。
- 新研究代码 SHA256：`/home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py=25ab1256933cf6cb6c0037834da08915b805aba68125b6f75147d106aadea6b8`; `/home/DIFFER/scripts/ndp_overnight_recovery_stage3.py=3a68a1542b25a296ed3d3f38bbbcb574c63993ef4e8c617f078bed868d28f488`; `/home/DIFFER/scripts/ndp_real_trajectory_activation.py=4588075428cd5e453221f4a75237bb83f328817a93bdac249877b0ced16528ed`。
- 数据来源：`outputs/ndp_margin_consistency_20261021_220823/aa_probe/aa_repeat_comparison_float64.json`。
- 全部为 train-only、冻结特征或无更新只读审计；无 query/gallery；无新的 optimizer.step。

## 运行命令

- `/usr/local/bin/python3.10 -u scripts/ndp_aa_operator_probe.py --mode amp --no-optimizer-step ...`
- `/usr/local/bin/python3.10 -u scripts/ndp_aa_operator_analysis.py --root ... --seed {1234,4321} ...`

未完成：真实网络因果效果、官方评估变化、完整历史 Batch-Hard 逐步日志和 A/A 根因仍未证明。
