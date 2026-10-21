# NDP 真实 Triplet 公式纠错

## 结论

原重建脚本把 `SoftMarginLoss(dist_an-dist_ap, target=+1)` 错写成 `softplus(dist_an-dist_ap)`，会把正确分离关系的损失方向反过来。已修正为：

\[
 d = d_{an}-d_{ap},\qquad L=\log(1+\exp(-d)),\qquad rac{\partial L}{\partial d}=-\sigma(-d).
\]

逐元素对照 `torch.nn.SoftMarginLoss(reduction='none')` 的结果：

- loss 最大绝对误差：`0`；
- 梯度最大绝对误差：`1.11e-16`；
- `d_an` 增大时 loss 降低：`True`；
- 方向正确：`True`。

`gradient_above_1e-8` 只是连续 SoftMargin 梯度的报告阈值，不能解释为 Hinge 的 active/inactive 机制。B0 的 `hard_factor=0`，欧氏距离下限为 `1e-12`；默认 Triplet 输入不在 `TripletLoss` 内归一化。

## 原始实现核对

`loss/triplet_loss.py` 使用欧氏距离；同 PID 中取最大正距离，不同 PID 中取最小负距离；`margin=None` 时调用 SoftMarginLoss；`make_loss.py` 以 `TripletLoss(feat[0], target, normalize_feature=False)` 调用。自身对角线不构成最大正样本，只要 batch 中存在另一张同 PID 图像。

## 证据与边界

修正前的 miner loss/梯度解释标记为 `SUPERSEDED`，历史文件保留。本报告没有执行 optimizer.step，也没有修改 GPC、CE、Triplet 或 Bio。

## 可审计命令

`/usr/local/bin/python3.10 -u scripts/ndp_triplet_formula_audit.py --output outputs/ndp_triplet_alignment_20261021_134148/triplet_formula_audit.json`
