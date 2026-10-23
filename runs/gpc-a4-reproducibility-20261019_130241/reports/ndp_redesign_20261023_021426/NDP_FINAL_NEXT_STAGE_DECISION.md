# NDP 最终下一阶段决策

`NDP_MECHANISM = INCONCLUSIVE`；`A/A_REPRODUCIBILITY = FAIL`；`CONTROLLED_SHORT_PILOT = NO_GO`。本轮没有修改 DIFFER+GPC、EVA02、CE、原始 Triplet、clipBio、GPC 权重/正样本、优化器或 sampler。

| 问题 | 证据结论 | 门禁 |
|---|---|---|
| 1 | 有真实 R∩V：637/618；翻转严格复现28/18。 | 描述性 PASS，不是预测 PASS |
| 2 | C激活16/8，A=38/9，B=17/0，D=12/6；有集合差异，但仍是历史动态阈值变换。 | 新机制未证明 |
| 3 | 解耦后新纳入12/6；未覆盖未来翻转，不能回顾性声称预测。 | NO causal claim |
| 4 | safe target 的其他关系负影响15/9；双侧 safe 32/26。 | collateral 风险存在 |
| 5 | PID级 Triplet 机会约8%，固定完整关系0；结构访问有增量，真实参数增量未证。 | INCONCLUSIVE |
| 6 | Hard 回顾性覆盖28/18但混入大量当前错误；R类0翻转；预算样本小且无因果。 | 不足以 GO |
| 7 | 完整 A/A 最大差异约0.00412，根因未唯一定位。 | FAIL |
| 8 | 可保留冻结诊断接口；训练损失当前不值得接入。 | NO_GO |

建议停止新增复杂门控、双侧/多 PID 组合和正式训练分支。若继续，只先记录真实 raw Batch-Hard 逐步覆盖并完成 A/A 首差异定位，再申请预注册预算匹配实验。

统计不确定性总说明：本轮全部结论来自固定关系全集、固定重建 sampler、冻结特征或重复进程 probe；任何 CI 仅用于描述性 bootstrap，不代表网络训练效果或测试集泛化。

## 可复现元数据

- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，不伪造本地 commit。
- 固定 baseline SHA256：`gpc_v2_train.py=d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`; `make_loss.py=9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043`; `eva02_l_bio.yml=c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97`; `triplet_loss.py=86656d44255172866d1e9854880c16e5d9536c6a756a577876312b663512092a`。
- checkpoint SHA256：`seed1234=e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`; `seed4321=d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7`。
- 新研究代码 SHA256：`/home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py=25ab1256933cf6cb6c0037834da08915b805aba68125b6f75147d106aadea6b8`; `/home/DIFFER/scripts/ndp_overnight_recovery_stage3.py=3a68a1542b25a296ed3d3f38bbbcb574c63993ef4e8c617f078bed868d28f488`; `/home/DIFFER/scripts/ndp_real_trajectory_activation.py=4588075428cd5e453221f4a75237bb83f328817a93bdac249877b0ced16528ed`。
- 数据来源：`/home/DIFFER/outputs/ndp_redesign_20261023_021426`; `/home/DIFFER/outputs/ndp_triplet_alignment_20261021_134148`; `/home/DIFFER/outputs/ndp_overnight_recovery_20261023_021426`。
- 全部为 train-only、冻结特征或无更新只读审计；无 query/gallery；无新的 optimizer.step。

## 运行命令

- `/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py --run-id 20261023_021426 --output-root /home/DIFFER/outputs/ndp_redesign_20261023_021426 --report-dir /home/DIFFER/reports/ndp_redesign_20261023_021426 --remote-main d6e24867a9bf33423d3342682c96d053d37ff190`
- `source /usr/local/PPU_SDK/envsetup.sh ppu && /usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_overnight_recovery_stage3.py 20261023_021426`
- `PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER /usr/local/bin/python3.10 -m pytest -q tests scripts/tests`

未完成：真实网络因果效果、官方评估变化、完整历史 Batch-Hard 逐步日志和 A/A 根因仍未证明。
