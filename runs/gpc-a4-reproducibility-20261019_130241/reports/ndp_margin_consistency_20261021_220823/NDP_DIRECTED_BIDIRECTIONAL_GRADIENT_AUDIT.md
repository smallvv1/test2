# 修复后双向身份边界梯度审计

输入采用已修复的 `ndp_relation_correction_20261021_192813/.../image_level_reciprocal_pairs.csv`，逐行核验同一时间区间、同一 A1/B1 交叉图像、A→B 与 B→A 各自合法正样本。seed1234 有 80 条图像级转移、15 个身份对；seed4321 有 70 条、13 个身份对。

| seed | A单侧恶化 | B单侧恶化 | 双侧恶化 | 双侧改善 | A翻转 | B翻转 |
|---|---:|---:|---:|---:|---:|---:|
|1234|9|18|25|28|0|0|
|4321|13|16|18|23|0|2|

所有 pair 的 `same_cross_images_verified=true`。mean-positive/max-negative 重算后，seed4321 的 B侧出现2条严格翻转；旧 pair 文件的 flip 字段来自上一版 q25/transition 聚合口径，不能与本轮统一 Margin 混用。这解释了旧汇总与新逐图重算的差异。

冻结特征梯度均有限；seed1234 在 0→20 和 200→400 分别为 1/32、4/32 active，seed4321 各区间无 active。样本量和无 optimizer.step 限制下，双侧仅支持“存在非对称风险结构”，不支持训练增益。
