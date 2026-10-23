# 夜间阶段3只读恢复

原失败确为 `NameError: name 'protocol' is not defined`。本轮只修复入口变量：`runtime_protocol = install_protocol(...)` 后显式传给 `choose_groups()`；并在 choose_groups 入口读取一次原始 loader，避免重复包装。没有重跑400步 continuation，阶段4/5未启动。

| seed | groups | 节点激活（update:count） | 自然激活 |
|---|---:|---|---|
| 1234 | 64 | 0:0, 20:22, 50:22, 100:20, 200:13, 400:37 | True |
| 4321 | 64 | 0:0, 20:11, 50:9, 100:18, 200:27, 400:26 | True |

阶段3=`COMPLETED`；阶段4=`NOT_RUN`；阶段5=`NOT_RUN`。自然激活只是保存节点上的只读关系诊断，不是训练效果。
单测检查协议初始化顺序、choose_groups 不调用 install_protocol、只读激活脚本无 optimizer.step。

统计不确定性：每个 seed 的 64 个固定诊断 groups 和 6 个固定节点均已枚举，激活数是描述性全集计数，不是泛化率。

## 可复现元数据

- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，不伪造本地 commit。
- 固定 baseline SHA256：`gpc_v2_train.py=d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`; `make_loss.py=9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043`; `eva02_l_bio.yml=c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97`; `triplet_loss.py=86656d44255172866d1e9854880c16e5d9536c6a756a577876312b663512092a`。
- checkpoint SHA256：`seed1234=e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`; `seed4321=d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7`。
- 新研究代码 SHA256：`/home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py=25ab1256933cf6cb6c0037834da08915b805aba68125b6f75147d106aadea6b8`; `/home/DIFFER/scripts/ndp_overnight_recovery_stage3.py=3a68a1542b25a296ed3d3f38bbbcb574c63993ef4e8c617f078bed868d28f488`; `/home/DIFFER/scripts/ndp_real_trajectory_activation.py=4588075428cd5e453221f4a75237bb83f328817a93bdac249877b0ced16528ed`。
- 数据来源：`/home/DIFFER/outputs/ndp_overnight_recovery_20261023_021426`; `outputs/ndp_overnight_20261020_201841`。
- 全部为 train-only、冻结特征或无更新只读审计；无 query/gallery；无新的 optimizer.step。

## 运行命令

- `source /usr/local/PPU_SDK/envsetup.sh ppu && /usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_overnight_recovery_stage3.py 20261023_021426`
- `PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER /usr/local/bin/python3.10 -m pytest -q tests/test_ndp_stage3_readonly_recovery.py`

未完成：真实网络因果效果、官方评估变化、完整历史 Batch-Hard 逐步日志和 A/A 根因仍未证明。
