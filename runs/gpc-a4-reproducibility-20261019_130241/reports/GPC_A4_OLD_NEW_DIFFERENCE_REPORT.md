# GPC-A4 旧新运行差异报告

比较：旧运行 `20261015_155822`，新运行 `20261016_161631`，均为 seed1234、30 epoch。结论状态：**`REPRODUCIBILITY_UNRESOLVED`**。

## 指标漂移

| 模型 | 旧 Rank-1 | 新 Rank-1 | Δ | 旧 mAP | 新 mAP | Δ |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 55.1227% | 58.0087% | +2.8860 pp | 45.0203% | 47.8634% | +2.8431 pp |
| A4 | 61.6162% | 57.5758% | -4.0404 pp | 47.6299% | 48.1193% | +0.4894 pp |

旧 A4−B0 为 `+6.4935 pp Rank-1`、`+2.6097 pp mAP`；新 A4−B0 为 `-0.4329 pp Rank-1`、`+0.2559 pp mAP`。B0 自身也从 `55.1227%` 变为 `58.0087%`，所以漂移不能全部归因于 A4。

## 已核对相同

- common initialization / 预训练初始化 SHA256：`72892762c37044b85dc9ed9999c833da574026c3715e760885363201a8a40ed0`；meta SHA256：`dd3acd0da3d935b2d734567b8db9bb171d63d64d24fe2a378f5b05cf335bee67`。
- development split manifest：`266786038e96fc87ce0b3ddced692d808064dfecc5228122c58d482158a90594`；CSV：`60b98e7bdfc9f9d31edc1efde4cbb7fe47ac5150b021f9f0d5e195fc681a2722`。
- 可见配置：batch 8、596 steps/epoch、SGD、base LR 2e-6、weight decay 0.05、lambda 0.1、30 epoch、训练 PID 60。
- 评估 protocol / Query / Gallery SHA256：`4a73d0382380f8518d638fc2c447db4f2e84e55cbe3264ea7d4b1b07e68f5702` / `3bb1035b7667fa3836e38594e189bf3c396ef995d6c1ef90ed9dae4880afb1d7` / `5832d32aebfed2204703af6ec9f2a9cb5980a7b2214c8182f2cc10cc915a584e`。

## 已核对不同或无法还原

1. 旧 A4 是 batch-local `balanced_selection`；新 A4 是全局累计 `cumulative_pid_balanced_selection`，新版本预计算并跨 rank broadcast epoch plan。
2. Phase7I stable hash 包含 phase 名称；旧 phase 与新 phase 不同，即使数值 seed 相同，关系选择流也不同。
3. 旧/新训练目录未保存完整每轮 sampler plan hash、各 rank 随机数状态和增强随机流，因此无法证明图像顺序、DDP 时序、AMP 事件序列完全相同。
4. 旧/新 wrapper、A4 selector、模型构建源码和实际 data 工作树状态不能由 GitHub 参考提交完整还原；旧 wrapper 和新 wrapper 的 SHA256 已写入 `/home/DIFFER/outputs/gpc_a4_reproducibility_20261019_130241/gpc_a4_old_new_audit.json`。

因此不能凭经验从 B0/A4 漂移中指定单一根因；已确认的是协议一致但训练实现/关系流/未记录随机状态不完全可比。
