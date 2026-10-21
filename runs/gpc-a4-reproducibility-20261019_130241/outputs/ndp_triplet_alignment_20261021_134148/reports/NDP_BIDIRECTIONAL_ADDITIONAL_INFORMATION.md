# NDP 双侧身份判别增量信息

双侧记录按同一 interval 与 relation index 对齐，保留 A→B、B→A 两个有向方向；没有用下一 transition 的 target_margin 代替当前 transition。结果：

| seed | image-pair records | identity pairs | 平均方向 delta<0 | 最危险方向 delta<0 | 任一严格翻转 |
|---:|---:|---:|---:|---:|---:|
| 1234 | 80 | 15 | 39 | 53 | 0 |
| 4321 | 70 | 13 | 30 | 47 | 0 |

五个时间区间均被保留。两 seed 在该严格互反图像子集上没有严格翻转，但分别有 53/80 与 47/70 条记录至少一侧 delta 为负；这说明一侧改善/另一侧恶化的“增量风险”可能存在，但当前集合没有足够严格翻转来证明双侧保护优于单侧保护。identity-pair 仅 15/13 个，不能外推全训练集。

双侧关系因此目前是诊断信息，不是已验证的独立训练机制；需要更大的、预注册且不按未来风险筛选的双侧关系集合，以及可重复的真实训练梯度比较。

## 可审计命令

`/usr/local/bin/python3.10 -u scripts/ndp_bidirectional_additional_audit.py --seed 1234 --pairs outputs/ndp_true_multipid_20261021_103952/analysis/seed1234/bidirectional/bidirectional_pairs.csv --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed1234/multi_pid_transition_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/bidirectional_additional/seed1234`
`/usr/local/bin/python3.10 -u scripts/ndp_bidirectional_additional_audit.py --seed 4321 --pairs outputs/ndp_true_multipid_20261021_103952/analysis/seed4321/bidirectional/bidirectional_pairs.csv --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed4321/multi_pid_transition_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/bidirectional_additional/seed4321`
