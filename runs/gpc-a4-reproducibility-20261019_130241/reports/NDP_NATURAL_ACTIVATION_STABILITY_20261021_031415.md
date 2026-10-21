# NDP 真实轨迹自然激活稳定性复核

## 数据边界

复用 `/home/DIFFER/outputs/ndp_overnight_recovery_20261021_024248/` 的固定 B0+GPC-v1
continuation。两种子各有 64 个预注册 train-only 诊断 group，节点为 update 0、20、50、
100、200、400；阶段2 每个 seed 完成 400 个有效 optimizer update，AMP skip=0。

宽松激活定义为 `advantage_drop > 0.02`，可靠激活进一步要求
`reference_advantage > 0.05`。该定义只用于事后审计，不作为训练中的未来信息选择器。

## 激活计数

| seed | reliable groups | update0 | update20 | update50 | update100 | update200 | update400 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 38 | 0 | 15 | 13 | 10 | 3 | 26 |
| 4321 | 36 | 0 | 9 | 6 | 9 | 16 | 17 |

在 update400，可靠激活率分别为 `26/38=68.42%` 和 `17/36=47.22%`。这证明固定
B0 continuation 的真实后续状态中确实存在自然激活机会；它仍然不是 NDP 参数更新
后的有效性证明。

## 状态稳定性

可靠激活集合的相邻节点 Jaccard：

- seed1234：20→50 `0.8667`，50→100 `0.7692`，100→200 `0.1818`，200→400 `0.0741`。
- seed4321：20→50 `0.3636`，50→100 `0.5000`，100→200 `0.5625`，200→400 `0.6500`。

激活不是固定的静态样本集合；其状态依赖性在两个 seed 中表现不同。update200→400
的变化来自真实模型参数更新节点之间的比较，而不是同一状态重复编码。

## 同 checkpoint 重复编码控制

新增无更新 PPU 审计脚本：`/home/DIFFER/scripts/ndp_activation_repeat_audit.py`，
SHA256 `477fb523b46740e4af37be422cb759496e9a1c367e5a37d63f93e2bae7f3f7f8`。

在两个 seed 的 update400 模型中，对同一组固定诊断关系进行两次 eval 重编码：

| seed | 图像数 | 最大绝对特征差 | 最小余弦 | optimizer.step |
|---:|---:|---:|---:|---|
| 1234 | 170 | 0 | 0.999999821 | 否 |
| 4321 | 160 | 0 | 0.999999821 | 否 |

因此本次激活变化不能由同一 checkpoint 的重复前向噪声解释。该控制不等同于跨进程
训练 A/A；训练 backward 仍被上一报告的 A/A 门禁阻塞。

## 判定

`NDP_ACTIVATION_STABILITY: STATE_DEPENDENT_NATURAL_OPPORTUNITY`

自然激活已在固定 train-only 轨迹中出现，且同状态重编码稳定；但目前不能推出 NDP
更新会降低身份损害，也不能推出 Rank-1/mAP 提升。

