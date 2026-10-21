# NDP 下一阶段决策

## 状态

- `A/A_REPRODUCIBILITY: FAIL`
- `NDP_LRD_MECHANISM: INCONCLUSIVE`
- `NDP_SHORT_PILOT: NO_GO`

## 判断依据

1. A/A 第 1 步 forward/loss 相同，但反向梯度在 block 22 内部已经出现约 `1e-5` 到 `1e-4` 量级的可测差异，随后向早期层放大；两次运行均没有 AMP skip。当前不能把小幅 NDP 差异解释为方法效应。
2. DDP all-reduce 与显式平均方向高度一致，未发现归约方向错误，但 rank-local 梯度本身不逐位一致，因此不能宣称完整 A/A 可重复。
3. LRD-B 完整 KL 在大量“正身份总质量未下降”的 anchor 上仍施加非零约束，且大部分 KL 来自组内形状变化；它存在过约束风险。
4. LRD-C 与自适应 Triplet、SupCon 的特征梯度方向可区分，满足进一步机制验证的必要条件，但尚无参数更新后效果证据。

## 不允许的动作

本轮不启动任何 NDP optimizer-step、30 epoch 或 60 epoch 训练；不修改 GPC-v1、CE、Triplet、Bio、EVA02 或历史 checkpoint。

## 下一项最小工作

优先修复/界定 PPU 反向非确定性：在固定单卡、双卡、AMP/FP32 条件下继续追踪 block22 attention backward 的底层算子；之后才可预注册 LRD-C 与三组控制的无更新梯度探针。若 A/A 仍无法给出可接受噪声边界，停止因果效果比较。
