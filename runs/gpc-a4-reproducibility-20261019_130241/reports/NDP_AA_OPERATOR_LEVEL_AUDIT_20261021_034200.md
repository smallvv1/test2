# A/A 反向传播算子级审计

运行标识：`20261021_034200`。远端参考提交：`5cdb8bdc2af703fd42e9c9105e0a2107d3464cf4`。

## 结论

`A/A_REPRODUCIBILITY=FAIL`。

两个独立单卡 AMP 运行使用同一物化 batch、同一 E60 checkpoint、同一 GPC 关系计划和恢复后的 optimizer/scaler 状态。第 1 步的 forward 输出 hash 和 loss 相同，`found_inf=0`，scaler 不变且 optimizer step 均执行；但未缩放参数梯度已经不同，随后第 2 步 forward 分叉。该 batch 的 `selected_relation_count=0`，所以这次首步探针隔离的是 CE/Triplet/Bio 反向，不能把差异归因于 GPC。

## 逐层定位

在 23 个 Transformer block 及 block 22 内部模块安装 hook。seed1234 的 reverse 路径中：

- `blocks.22` 顶层输出梯度相同；
- 首个非零可观测差异出现在 `blocks.22.norm1` 输出梯度，最大绝对差约 `7.63e-6`；
- `blocks.22.attn.q_proj` 输出梯度最大绝对差约 `3.05e-5`；
- 随后 `blocks.21`、`blocks.20` 及更早 block 的差异逐步放大；
- `head` 输出梯度保持相同。

这把分叉定位到 block 22 内部反向链，但没有定位到 PPU 的单个底层 kernel。所有模块 forward summary hash 在两次运行间一致，因此当前证据不支持“第 1 步 forward 先分叉”。

seed1234 选定参数的 A/A 未缩放梯度相对 L2 差异最高约 `3.83e-3`；seed4321 约 `4.53e-3`。对应余弦接近 1，但不是逐元素可复现。两次运行均无 AMP skip。

## DDP 对照

双 PPU 使用同一物化 batch 做真实 DDP backward，并与显式 rank-local FP32 梯度平均比较：

- `patch_embed.proj.weight` 最大相对差约 `1.73e-3`，余弦 `0.9999985`；
- `blocks.0.attn.q_proj.weight` 最大相对差约 `1.67e-3`，余弦 `0.9999986`；
- `blocks.11.attn.q_proj.weight` 最大相对差约 `6.48e-4`，余弦 `0.9999998`；
- `blocks.22.attn.q_proj.weight` 最大相对差约 `2.91e-4`，余弦 `0.99999996`；
- `head.weight` 相对差约 `3.24e-11`。

两个 rank 的 local gradient hash 本身不同，故 DDP 结果不能被解释为逐位相同；但没有观察到 all-reduce 引入的大方向错误。首个 A/A 分叉仍应归入归约前的 PPU 反向数值非确定性，底层算子根因尚未确定。

## 证据路径与源码哈希

- 探针：`/home/DIFFER/scripts/ndp_aa_operator_probe.py`，SHA256 `ddf3fb30bd01df708c8278826ab90b6a01cac27ac741f80d886a6c2a98b4ed4f`。
- 张量比较：`/home/DIFFER/scripts/ndp_aa_operator_analysis.py`，SHA256 `a6685be54e23717be5c177dd82c7474bbee3933aefc24cad1059db12735526fa`。
- DDP 探针：`/home/DIFFER/scripts/ndp_ddp_gradient_probe.py`，SHA256 `7710a48e70ac1976e5b4cd6f3adce92a35623286a293fdda5db2b6baf3658d2b`。
- 实际训练引擎 `/data/DIFFER/scripts/gpc_v2_train.py` SHA256 `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`。
- EVA02 `/data/DIFFER/model/eva_meta_bio.py` SHA256 `0a707bc0ecbe62816685147c3b2bb5534459b5110c34f457f405f472b62f9b76`。
- 逐层 JSON/CSV：`/home/DIFFER/outputs/ndp_aa_operator_audit_20261021_033800/`。
- DDP JSON：`/home/DIFFER/outputs/ndp_ddp_gradient_audit_20261021_033042/`。

大体积 `.tensors.pt` 只保留在本地，不应归档到 GitHub。
