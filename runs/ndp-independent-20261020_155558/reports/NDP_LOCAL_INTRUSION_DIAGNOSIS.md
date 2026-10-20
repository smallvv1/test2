# NDP 局部异身份侵入诊断

RUN_TIME：`20261020_155558`

本轮使用 LTCC train-only 的 9576 张图、77 个 PID、256 个衣物状态，复用已审计的 Bio-only 与 Bio+GPC-v1 E60 特征库。K=5、10、20 在运行前固定；没有读取官方 query/gallery/test，也没有训练或更新 checkpoint。

这里的 Bio-only→Bio+GPC 对照是冻结状态之间的几何差异审计，不是同一条逐步训练轨迹的因果证明；PPU AMP 的 A/A 数值不确定性仍单独保留。

## 主要结果

| seed | K | 错误 PID 侵入 anchor 比例 | 有正侵入 PID | Δ最近异身份相似度 | Δ局部主 margin | 原本正确但翻转 |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 5 | 3.64% | 61/77 | +0.00565 | +0.00245 | 1,970/35,977（5.48%）|
| 1234 | 10 | 8.89% | 69/77 | +0.00565 | +0.00245 | 3,580/74,653（4.80%）|
| 1234 | 20 | 18.83% | 75/77 | +0.00565 | +0.00245 | 6,346/154,153（4.12%）|
| 4321 | 5 | 3.77% | 58/77 | +0.00605 | −0.00404 | 1,885/36,457（5.17%）|
| 4321 | 10 | 8.47% | 67/77 | +0.00605 | −0.00404 | 3,362/75,391（4.46%）|
| 4321 | 20 | 18.21% | 72/77 | +0.00605 | −0.00404 | 5,735/155,283（3.69%）|

表中“原本正确但翻转”是参考 `g_ref>0` 且当前 `g_current<=0` 的严格翻转；它不是所有 margin 下降。GPC 后最近异身份相似度在两个 seed 都上升，但局部身份 margin 一正一负，不能形成跨种子一致的有害方向。

## 判定

存在 train-only 局部邻域漂移和部分异身份侵入，但“GPC 稳定破坏正确局部身份关系”的证据为 `MIXED`。旧的 PPU 短程 NDP 原型也只有 4 步、激活不足，不能补足该门禁。因此不启动 NDP 训练。

## 产物

- [完整 NDP-A 审计](/home/DIFFER/reports/NDP_A_NEIGHBORHOOD_DISCRIMINATIVE_FEASIBILITY_AUDIT.md)
- [严格参考关系摘要](/home/DIFFER/outputs/ndp_local_intrusion_20261020_155558/reference_relations/summary.csv)
- [Promotion gate](/home/DIFFER/outputs/ndp_local_intrusion_20261020_155558/tables/promotion_gate.json)
- [运行 Manifest](/home/DIFFER/outputs/ndp_local_intrusion_20261020_155558/NDP_A_MANIFEST.json)

