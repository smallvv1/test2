# NDP Reliability/Vulnerability 增量信息审计

## 正确分母

只使用 `M_200 > 0` 的关系作为候选，标签为 `M_400 <= 0`。已在 update200 错误的关系不进入首次翻转分母。update400 只产生标签，不参与分数、关系选择或阈值。

| seed | eligible | 严格翻转 | 稳健 1e-3 | 稳健 1e-2 | 当前 Margin AUC/AP | pre-200 波动 AUC/AP | pre-200 最小 Margin AUC/AP | pre-200 斜率 AUC/AP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1020 | 31 (3.0392%) | 21 | 4 | 0.9571/0.4950 | 0.6375/0.0539 | 0.9818/0.6912 | 0.3244/0.0216 |
| 4321 | 977 | 9 (0.9212%) | 4 | 0 | 0.9890/0.5193 | 0.7233/0.0197 | 0.9917/0.6881 | 0.4014/0.0127 |

当前边界距离是最强的描述变量，但这有明显的“靠近零定义效应”：两 seed 的事件分别为 31/1020 与 9/977，且事件主要集中于 `M_200<=0.05`（31/356 与 9/320）。pre-200 历史最小 Margin 仍然很高，但它本质上也是历史边界接近度的代理，不能直接称为对未来更新方向的独立预测；历史下降斜率 AUC 仅 0.3244/0.4014，方向不稳定且 AP 很低。pre-200 波动 AUC 0.6375/0.7233，但 AP 0.0539/0.0197，事件稀少，不能作为跨 seed 已确认的动态机制。

做了 200 次按当前 Margin 最近邻的事件/非事件描述性匹配。匹配后的历史不稳定性 AUC 均值为 seed1234 `0.5857 [0.5379,0.6317]`、seed4321 `0.6844 [0.5182,0.8645]`；历史斜率均值为 `0.2431 [0.1904,0.2945]` / `0.2225 [0.1358,0.2963]`。小事件数和匹配近似使其仅为探索性控制，不能升级为预测 PASS。

**判断：**当前结果支持“当前局部边界接近零的关系更容易翻转”，不支持“历史下降趋势能稳定预测新的退化”。预测式 NDP 仍 `NOT_SUPPORTED`；若继续研究，应优先转为反应式、事后单侧保护，并且仍需证明其超出真实 Triplet 的覆盖增量。

## 可审计命令

`/usr/local/bin/python3.10 -u scripts/ndp_reliability_incremental_audit.py --seed 1234 --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed1234/multi_pid_transition_rows.csv --reliability-rows outputs/ndp_true_multipid_20261021_103952/analysis/seed1234/reliability/reliability_vulnerability_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/reliability_incremental/seed1234`
`/usr/local/bin/python3.10 -u scripts/ndp_reliability_incremental_audit.py --seed 4321 --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed4321/multi_pid_transition_rows.csv --reliability-rows outputs/ndp_true_multipid_20261021_103952/analysis/seed4321/reliability/reliability_vulnerability_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/reliability_incremental/seed4321`
