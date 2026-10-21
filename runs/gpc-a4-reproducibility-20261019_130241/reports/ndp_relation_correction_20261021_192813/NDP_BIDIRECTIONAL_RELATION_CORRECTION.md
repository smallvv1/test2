# 双侧身份关系修正

## 修正内容

旧实现以 `(interval, anchor_pid, negative_pid)` 为字典键并 last-write-wins，会覆盖同一身份对下不同图像边界。本批次改为保存完整 directed row 列表；图像级双侧关系要求同一时间区间、A 图像恰为 B 的负图像、B 图像恰为 A 的负图像、PID 对调一致，并保留 A→B 与 B→A 两个方向。

## 结果

| Seed | 输入 transition | 身份有向组 | 严格图像互反对 | 身份对-区间组 | A改善/B恶化 | A恶化/B改善 | 双改善 | 双恶化 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1234|12800|3950|80|2590|19|11|27|23|
|4321|12800|4005|70|2655|11|17|23|19|

双侧不对称仍然出现，但它是固定关系上的描述性现象，不是 NDP 训练收益证据。两个方向不能被平均成一个“身份对风险”。

证据 CSV：`/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/bidirectional_correction/seed1234/identity_directed_rows.csv`、`/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/bidirectional_correction/seed1234/image_level_reciprocal_pairs.csv`，以及 Seed4321 对应文件。

本批次的实际命令保存在 `/home/DIFFER/log_NDP/ndp_relation_correction_20261021_192813/`；所有模型探针使用 `/usr/local/bin/python3.10`，PPU 环境为 `PPU_SDK=/usr/local/PPU_SDK`，离线梯度审计不执行 optimizer.step。
