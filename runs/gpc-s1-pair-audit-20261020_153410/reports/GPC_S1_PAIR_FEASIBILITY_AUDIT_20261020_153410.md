# GPC S1-Pair 严格可行性审计

本报告使用冻结 LTCC train-only batch 计划，对 seed1234/4321、rank0/1、完整 epoch60 计划进行 CPU 离线关系分析；共 596 batch/rank，合计 2384 个 batch。没有执行模型 forward/backward。

## 定义

S1-Pair 只允许：

1. 保持 S0 active batch；
2. 保持每批 selected pair 数；
3. 保持被选 PID；
4. 保持同衣/跨衣关系类型；
5. 替换为当前 batch 内、同 PID、不同图片的另一条关系；
6. 全局不重复；无合法替代时保留原 S0 关系。

## 结果

| seed | 全 batch | S0 active batch | S0 关系 | 可局部替换 | 可全局替换 | 无法替换 | 实际替换 | 替换率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1192 | 142 | 153 | 28 | 28 | 125 | 28 | 18.30% |
| 4321 | 1192 | 140 | 148 | 27 | 27 | 121 | 27 | 18.24% |
| 合计 | 2384 | 282 | 301 | 55 | 55 | 246 | 55 | 18.27% |

“可局部替换”表示存在至少一个满足 PID 和衣服类型约束的候选；“可全局替换”还排除了已被 S0 或其他 S1-Pair 关系占用的 relation key。此次两者相同，说明全局去重没有进一步压缩可替换数量。

## 约束门禁

四个 seed/rank 组合全部通过：

- S0/S1 active batch 集合相同；
- 每批关系数量相同；
- PID 与同衣/跨衣类型 multiset 相同；
- S1 relation key 无重复；
- 所有关系两端来自当前 batch；
- 关系定义仍为同 PID、不同图片；
- 未使用验证 PID、官方 query/gallery 或跨 batch 特征。

S0 与 S1-Pair 的同衣/跨衣计数逐 rank 完全一致。例如 seed1234 rank0 为 23/52，seed1234 rank1 为 23/55；seed4321 rank0 为 26/36，rank1 为 31/55。变化只发生在具体图像对，PID 和衣服关系属性没有变化。

## Hash

| seed/rank | S0 relation SHA256 | S1-Pair relation SHA256 |
|---|---|---|
| 1234/0 | `22f52ee25db99786e1344cb64b87df9befb7f84a12f59dd35a0d9594b67b768a` | `217c6147030159e940347ab9d9dac76f51ee9676d8549d5e4c250d9ab27a0251` |
| 1234/1 | `bd5b7825281577047a9d13ddc6296c82e688df63392af3f6a804395bf0854b1b` | `a1149ae13c2fbf3799ffc95f4c49284a8891cda405a11b90725d0396e8cb8f1c` |
| 4321/0 | `aed6800440b91f79f03000233138abcfbe8a11ddfad89e34c4ae33deb6463f6e` | `b8691aba9be74e1d628f4d5c2a91a2a83fa6236b538c8b9fb2cc2cbe5ed7610b` |
| 4321/1 | `e787a20747b8f37ed958f4daccc6860c78187e527cd4e5d04d3147bb81b6c356` | `0631e1ac754cd5c27945ce72cf264f3712de5924fc6e03eff1246a2eb414c570` |

## 判定

`S1_PAIR_FEASIBILITY_LIMITED`。

严格的纯正样本图像对选择在当前冻结采样计划中只覆盖约 18.3% 的 S0 关系；其余 81.7% 没有满足全部约束的替代关系。因此，不能把现有 S1 全部结果重新解释成纯 Pair 机制收益。

实际脚本：`/home/DIFFER/scripts/gpc_s1_pair_feasibility_audit.py`。

