# GPC-A4 机制有效性报告

## PID 暴露均衡

| 运行 | 总有效 relation | active batch | active rate | PID contribution CV | Gini | min | max | AMP skip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old_B0_seed1234 | 4429.0 | 4013.0 | 0.112220 | 1.048756 | 0.514921 | 6.0 | 358.0 | 16.0 |
| old_A4_seed1234 | 4429.0 | 4013.0 | 0.112220 | 1.820747 | 0.693268 | 1.0 | 795.0 | 18.0 |
| new_B0_seed1234 | 4429.0 | 4013.0 | 0.112220 | 1.027179 | 0.514356 | 5.0 | 348.0 | 16.0 |
| new_A4_seed1234 | 4429.0 | 4013.0 | 0.112220 | 0.396828 | 0.227572 | 10.0 | 124.0 | 16.0 |
| new_R2_seed1234 | 4429.0 | 4013.0 | 0.112220 | 1.037944 | 0.514657 | 4.0 | 362.0 | 14.0 |
| new_B0_seed4321 | 4582.0 | 4175.0 | 0.116751 | 0.998367 | 0.505238 | 6.0 | 368.0 | 18.0 |
| new_A4_seed4321 | 4582.0 | 4175.0 | 0.116751 | 0.392878 | 0.225316 | 22.0 | 124.0 | 16.0 |
| new_R2_seed4321 | 4582.0 | 4175.0 | 0.116751 | 1.028595 | 0.517292 | 8.0 | 363.0 | 20.0 |

新 A4 两个 seed 相比新 B0 的 PID contribution CV/Gini 均显著下降：seed1234 为 `1.027179/0.514356 → 0.396828/0.227572`，seed4321 为 `0.998367/0.505238 → 0.392878/0.225316`。总有效 relation、active batch rate、optimizer updates 和 matched pair budget 保持一致。因此可以确认：**A4 的 PID 训练暴露均衡机制生效**，不是增加 relation 数量造成的。

## 暴露均衡不等于检索均衡

逐 PID 检索结果显示 B0→A4 seed1234 的 Rank-1 为改善/恶化/不变 `3/5/3`，seed4321 为 `4/5/2`；A4 的 Image-micro Rank-1 一正一负，PID-macro Rank-1 两个 seed 均为负。因此不能把训练侧 CV/Gini 下降直接解释为逐身份检索收益。

## 梯度、相似度和局部 margin

训练日志只记录关系数量、raw loss、AMP 事件，没有记录 A4 的 GPC 梯度范数、CE/Triplet/Bio 梯度范数、GPC/基础梯度比例、正样本相似度、最近异身份相似度和局部身份 margin。故 A4 的这些字段状态为 **`NOT_RECORDED`**，机制因果判定为 **`INCONCLUSIVE`**。

已有的 train-only 梯度兼容性探针只覆盖 B0/A5、共享 `feat[0]` 输出层、无 optimizer update，且明确未测完整参数梯度；它不能替代 A4 训练侧记录，也没有证明梯度冲突导致性能损伤。原始文件：`/home/DIFFER/outputs/gpc_gradient_compatibility_20261016_141306_recovery4/gradient_compatibility.json`。

## 判定

均衡 selector 的实现效果已确认；“均衡 selector 导致/改善了逐身份检索效果”的因果链没有足够证据。A4 机制有效性总体为 `INCONCLUSIVE`。
