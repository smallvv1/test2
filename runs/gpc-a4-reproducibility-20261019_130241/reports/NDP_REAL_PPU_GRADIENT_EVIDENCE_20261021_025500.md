# NDP 真实 PPU 参数梯度证据

## 探针

在 update=400 的真实 continuation 节点，对阶段3自然激活且 `A_reference>0.05` 的全部关系做无更新 PPU 探针：seed1234=26 个，seed4321=17 个。当前模型为节点 checkpoint，参考特征来自同一 seed 的 E60 起始状态；正负候选与阶段3完全一致。未调用 optimizer.step，故这是机制梯度证据，不是训练效果证据。

参数梯度范数是固定的可审计参数子集（patch embedding、若干 attention qkv、head）的和，不冒充全模型参数范数；特征梯度和所有记录梯度均检查 finite。

| seed | 候选 | anchor数 | 非零loss数 | 平均loss | 平均参数梯度范数 | 梯度finite |
|---|---|---:|---:|---:|---:|---|
|1234|C0 Batch-Hard Triplet|26|26|0.158364|56.6618|是|
|1234|C1 参考自适应margin|26|26|0.070310|184.1599|是|
|1234|C2 detached-cache NDP|26|21|0.051262|73.3837|是|
|1234|C3 live-neighbor NDP|26|26|0.070310|184.1341|是|
|1234|C4 SupCon/N-pair|26|26|0.286119|42.4660|是|
|4321|C0 Batch-Hard Triplet|17|17|0.135011|48.4937|是|
|4321|C1 参考自适应margin|17|17|0.058100|141.7792|是|
|4321|C2 detached-cache NDP|17|9|0.039686|73.8634|是|
|4321|C3 live-neighbor NDP|17|17|0.058100|141.7633|是|
|4321|C4 SupCon/N-pair|17|17|0.350168|41.0423|是|

## 解释

C1 与 C3 在同一关系、同一 `logmeanexp` 优势和同一阈值下数值及梯度几乎相同；这确认参考 margin NDP 在该实现上仍是自适应 margin Triplet 的同方向形式，不构成独立机制。C2 的缓存版本只让当前 anchor 保留梯度，缓存邻居 detach，因此激活和梯度规模不同；但其信息来自冻结缓存，不能由该差异直接推出性能收益。

现有证据支持：NDP 候选可在真实 EVA02/PPU 图上产生有限、非零的参数梯度，并在两个 seed 的真实激活节点出现。尚未支持：NDP optimizer-step 后减少局部损害、超过 Triplet/SupCon 或提高 Rank-1。
