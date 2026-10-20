# GPC A/A 失败根因审计（20261020_153410）

本报告复核 `20261020_112219`、`20261020_104944` 的全部 A/A、A/B 子实验；没有启动新模型训练。

## A/A 四次返回码 2

| seed | repeat | 失败分支 | AMP skip | found_inf 参数 | 关系数 | 有效更新 | 诊断基础损失差值 |
|---:|---:|---|---|---|---:|---:|---:|
| 1234 | 1 | A1 | 第14步 | `blocks.0.attn.proj.weight` | 0 | 20/19 | +0.300633 |
| 1234 | 2 | A1 | 第4步 | `attn.v_proj`、`attn.proj`、`mlp.fc1_g` | 0 | 20/19 | -0.654738 |
| 4321 | 1 | A0 | 第13步 | `attn.v_proj`、`attn.proj`、`mlp.fc1_g` | 0 | 19/20 | +0.006680 |
| 4321 | 3 | A0/A1 | 第13/19步 | `attn.v_proj`、`attn.proj`、`mlp.fc1_g`；`attn.proj` | 0/0 | 19/19 | +0.331932 |

每次事件均为 `found_inf=1`、GradScaler `1024→512`、`optimizer.step()` 未执行。返回码 2 来自显式配对门禁，而不是未捕获异常：

- `gpc_s0_s0_noise.py:127`：`return 0 if pair_valid else 2`。
- `gpc_s0_s1_controlled_pair.py:506`：同样按配对门禁返回 2。

全部 12 个子实验 JSON 和 child.log 均存在；child.log 未发现 `Traceback` 或 `SIGABRT`。

## 协议一致性

6 个 A/A 配对均满足：初始模型、optimizer/scaler 初始状态、输入 batch、增强输入、诊断 batch、batch plan、candidate plan、selected relation hash 一致。共同 checkpoint SHA256：

`72892762c37044b85dc9ed9999c833da574026c3715e760885363201a8a40ed0`

共同 config SHA256：

`1047516fb8efd81d5328c52d99065951620ea6a42a9a27aa228eeafde61ee944`

共同 split SHA256：

`266786038e96fc87ce0b3ddced692d808064dfecc5228122c58d482158a90594`

seed1234 repeat1 的左右分支初始状态记录还显示：model state SHA256 `49b5955d680c10b2f6a65249f8f4c44e7ef17ec9fa8be166b782523d16bac359`、optimizer SHA256 `96a9be2d83cac7b03da4949f06bd5d7999360c7216262208bf8b9b337f486332`、GradScaler SHA256 `a178b45c5b63539daf933138e1b755e301bca80b4b1993392cb599b6f08e50eb`、初始 scale `1024`；两分支一致。RNG hash（Python/NumPy/Torch/PPU）也逐项一致。

因此未发现实验程序、数据划分或随机状态初始化不一致。A/A 的致命差异是 PPU AMP 数值轨迹在早期已出现不同，随后在不同步骤触发非有限梯度检测；但没有证据把问题精确归因到某一个 PPU 算子。

严格状态：`AA_ROOT_CAUSE_UNRESOLVED`。根因类别已定位为 PPU AMP/数值非确定性导致的 skip 轨迹差异，但算子级根因仍未闭合。

## 真实数值波动

6 个 A/A 的首个标量差异均在第2步；参数状态 hash 在第1或第2步分叉。Δ基础损失范围：

- seed1234：`-0.654738～+0.300633`；
- seed4321：`-0.027419～+0.331932`。

这已经足以覆盖当前 A/B 20 步差异量级，不能将 A/B 的单次诊断差异解释成 S1 收益。

## 证据路径

- `/home/DIFFER/outputs/gpc_s0s0_noise_20261020_112219/`
- `/home/DIFFER/outputs/gpc_s0s1_ab_repeats_20261020_112219/`
- `/home/DIFFER/outputs/gpc_s0s1_controlled_20261020_104944/`
- `/home/DIFFER/outputs/gpc_aa_failure_audit_20261020_141052/failure_root_cause_audit.json`

