# NDP Margin 定义一致性修复

本批次使用固定 Bio+GPC-v1 continuation 的 train-only 数据，未执行 optimizer.step，未访问 query/gallery/test。

旧审计的参考值是 `mean(positive)-max(negative)`，autograd 当前值却只取第一正、第一负。本轮新增 `scripts/ndp_margin_consistency.py`，统一实现三种候选：A=`Q25(pos cosine)-max(neg cosine)`，B=`mean(pos cosine)-max(neg cosine)`，C=`pos[0]-neg[0]`。同一关系的 reference/current 必须使用同一 mode、同一索引集合；reference detach，current 保留图。

反应式损失为 `mean(ReLU(M_ref-M_cur-epsilon))`，只在 mask 内且 raw hinge>0 的行计 active。非有限行、空关系和零权重不会用 `NaN*0`，而是返回有限且与安全 current 张量相连的零值。`negative_pid_count` 按真实 PID 去重，不再统计负图像数量。

全套测试命令 `PYTHONPATH=/home/DIFFER/scripts /usr/local/bin/python3.10 -m pytest -q tests`：`38 passed`。K=5 主审计使用 B 定义时，seed1234 的 A类可靠/观察退化/active 为 `763/183/33`，seed4321 为 `755/68/9`；两组梯度均有限。完整逐关系证据见 `outputs/ndp_margin_consistency_20261021_220823/reactive_k5/seed*/`。

实际运行：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -u scripts/ndp_k5_trajectory_feature_export.py ... --graph-type difficult --k-pid 5
/usr/local/bin/python3.10 -u scripts/ndp_full_multipid_reactive_audit.py ... --epsilon 0.02
```

结论：定义错位已修复；这只证明数学接口和冻结特征梯度正确，不证明 NDP 训练收益。
