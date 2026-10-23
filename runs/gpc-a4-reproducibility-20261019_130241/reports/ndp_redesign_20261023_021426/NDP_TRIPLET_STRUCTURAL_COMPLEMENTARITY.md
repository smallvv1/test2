# NDP 与真实 Triplet 的结构互补性

原始 DIFFER Triplet 固定为未归一化 raw `feat[0]`、欧氏距离、hard positive=max 同 PID、hard negative=min 异 PID、`SoftMarginLoss(dist_an-dist_ap)`。以下复用真实 raw 路径对齐和重建 Batch-Hard 覆盖；NDP 只在固定 difficult-K5 图上做跨 Batch 关系诊断。

| seed | K5关系 | PID级有机会 | PID级hard负选择 | 固定图像级有机会 | 完整关系选择 | active triplet |
|---|---:|---:|---:|---:|---:|---:|
| seed1234 | 2560 | 215 | 197 | 5 | 0 | 0 |
| seed4321 | 2560 | 220 | 200 | 0 | 0 | 0 |

PID级机会约8%，固定图像级完整关系选择为0；因此 NDP 有结构访问增量，但不能说真实训练中这些身份完全没有间接监督（anchor_any_triplet=398）。冻结关系梯度余弦既有结果为 seed1234 0.0486、seed4321 0.0230；这不是有效性证明。
结论：`STRUCTURAL_COMPLEMENTARITY = DESCRIPTIVE_ONLY`，没有真实参数增量或有益冲突的因果证据。

统计不确定性：覆盖数来自固定重建 sampler 和固定 K5 图；由于没有历史 miner 日志，未对真实训练覆盖率给出置信区间。

## 可复现元数据

- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，不伪造本地 commit。
- 固定 baseline SHA256：`gpc_v2_train.py=d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`; `make_loss.py=9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043`; `eva02_l_bio.yml=c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97`; `triplet_loss.py=86656d44255172866d1e9854880c16e5d9536c6a756a577876312b663512092a`。
- checkpoint SHA256：`seed1234=e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`; `seed4321=d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7`。
- 新研究代码 SHA256：`/home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py=25ab1256933cf6cb6c0037834da08915b805aba68125b6f75147d106aadea6b8`; `/home/DIFFER/scripts/ndp_overnight_recovery_stage3.py=3a68a1542b25a296ed3d3f38bbbcb574c63993ef4e8c617f078bed868d28f488`; `/home/DIFFER/scripts/ndp_real_trajectory_activation.py=4588075428cd5e453221f4a75237bb83f328817a93bdac249877b0ced16528ed`。
- 数据来源：`/home/DIFFER/outputs/ndp_redesign_20261023_021426`; `/home/DIFFER/outputs/ndp_triplet_alignment_20261021_134148`; `/home/DIFFER/outputs/ndp_overnight_recovery_20261023_021426`。
- 全部为 train-only、冻结特征或无更新只读审计；无 query/gallery；无新的 optimizer.step。

## 运行命令

- `/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py --run-id 20261023_021426 --output-root /home/DIFFER/outputs/ndp_redesign_20261023_021426 --report-dir /home/DIFFER/reports/ndp_redesign_20261023_021426 --remote-main d6e24867a9bf33423d3342682c96d053d37ff190`
- `/usr/local/bin/python3.10 -u scripts/ndp_real_triplet_miner_coverage.py ... --epochs 60 61`
- `/usr/local/bin/python3.10 -u scripts/ndp_triplet_complementarity_audit.py ... --no-optimizer-step`

未完成：真实网络因果效果、官方评估变化、完整历史 Batch-Hard 逐步日志和 A/A 根因仍未证明。
