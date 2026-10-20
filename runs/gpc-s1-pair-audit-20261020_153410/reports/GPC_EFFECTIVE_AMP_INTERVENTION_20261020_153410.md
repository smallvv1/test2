# GPC 有效 AMP 干预审计

本报告把“计划选中关系”“进入 GPC loss/梯度计算”“该步真正执行 optimizer.step”分开统计。

## 最新 20 步 A/B

| seed | active step | 每个 arm 计划关系 | GPC step skip | 每个 arm 有效参数更新关系 |
|---:|---|---:|---|---:|
| 1234 | 16、20 | 2 | 无 | 2 |
| 4321 | 16 | 1 | 无 | 1 |

在 `20261020_112219` 的 6 个 A/B 子实验中，S0/S1 的 active relation step 都执行了 `optimizer.step()`；因此按一条冻结关系计划计，seed1234 为 2/2、seed4321 为 1/1。三次 repeat 后，S0 和 S1 各自分别产生 9 次有效关系更新事件。

但这不代表所有配对都有效：seed1234 repeat3 的 S1 在第4步 AMP skip，且该步 `selected_relation_count=0`，所以它仍因整体更新序列不一致而 INVALID，但不是 GPC active step 被跳过。

## A/A 失败与 GPC 的关系

四个 A/A 失败的 AMP skip 均发生在 `selected_relation_count=0` 的步骤；因此 0 条 GPC active relation 被这些失败事件直接阻断。A/A 失败反映共享 AMP/PPU 数值轨迹不稳定，不反映关系选择。

## 之前的 seed1234 20 步复核

在 `20261020_104944/pair_seed1234_steps20_retry2`，第16步同时是 GPC active step，也是 S0/S1 的共同 AMP skip；该步关系进入了前向、GPC loss 和梯度统计，但没有完成参数更新。第20步关系更新成功，所以每个 arm 为 2 条计划关系、1 条有效参数更新关系。

这说明不能仅看 selected relation hash 或 relation count 推断“有效 GPC 更新”。

## 结论

在最新 20 步 A/B 中，有效关系更新量为 seed1234 的 2 条、seed4321 的 1 条；跨 repeat 是关系事件的重复，不是新的关系计划。历史 retry2 还证明 active GPC step 可能被 AMP skip。

由于 A/A 仍有 4/6 轨迹门禁失败，数值基线尚未控制，当前有效干预量不足以支持 S1 训练效用判断。

