# Historical Safe Boundary 候选审计

预注册候选：`M_safe=max(0.05,Q25(M0,M20,M50,M100)-0.02)`；C 只对历史可靠关系计算。A 使用固定 update0 Margin，B 使用 update100，D 使用固定当前阈值0.05，E 使用 `softplus(-M_current)` 的 Batch-Hard 关系代理。A/B/C 的 reference 均停止梯度；当前改善（当前 Margin 高于目标）对应零 hinge。

| seed | 候选 | R内选择 | R内激活 | 平均原始值 | 原始值L2 |
|---|---|---|---|---|---|
| 1234 | A 固定 update0 | 775 | 38 | 0.000621049 | 0.115418 |
| 1234 | B 历史最后可信节点 | 775 | 17 | 0.000398701 | 0.0866075 |
| 1234 | C 历史 Q25 安全边界 | 775 | 16 | 0.000143012 | 0.0434852 |
| 1234 | D 当前固定0.05 | 775 | 12 | 0.000122817 | 0.0425159 |
| 1234 | E Batch-Hard SoftMargin代理 | 775 | 775 | 0.632148 | 17.613 |
| 4321 | A 固定 update0 | 761 | 9 | 7.682e-05 | 0.0255 |
| 4321 | B 历史最后可信节点 | 761 | 0 | 0 | 0 |
| 4321 | C 历史 Q25 安全边界 | 761 | 8 | 2.8963e-05 | 0.00984783 |
| 4321 | D 当前固定0.05 | 761 | 6 | 1.75744e-05 | 0.00702154 |
| 4321 | E Batch-Hard SoftMargin代理 | 761 | 761 | 0.63204 | 17.4494 |

C 若与 A/B 只有激活边界不同，不能称为独立机制；必须同时查看目标激活集合、梯度范数和 P4 局部影响。结果未执行更新，因此不构成训练收益证据。

统计不确定性：候选激活是在固定关系全集上的精确计数；没有把关系全集伪装成随机样本，故本表不提供抽样 CI。

- 运行命令：`/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py --run-id 20261023_021426 --output-root /home/DIFFER/outputs/ndp_redesign_20261023_021426 --report-dir /home/DIFFER/reports/ndp_redesign_20261023_021426 --remote-main d6e24867a9bf33423d3342682c96d053d37ff190`
- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，因此不伪造本地 commit。
- 独立研究代码 SHA256：`25ab1256933cf6cb6c0037834da08915b805aba68125b6f75147d106aadea6b8`。
- 数据来源和 SHA256：见 `RUN_MANIFEST.json`；固定 checkpoint SHA256：见已有 `outputs/ndp_triplet_alignment_20261021_134148/RUN_MANIFEST.json`。
- 明确 checkpoint SHA256：seed1234=`e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`；seed4321=`d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7`。
- 协议：只读 train-only 特征；update400 只作事后标签；无 optimizer.step、无官方 query/gallery。
