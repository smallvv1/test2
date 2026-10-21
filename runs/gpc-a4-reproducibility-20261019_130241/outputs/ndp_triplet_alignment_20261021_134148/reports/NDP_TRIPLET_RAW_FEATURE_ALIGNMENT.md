# NDP Triplet 原始特征对齐审计

## 结论

已在 PPU `cuda:0` 对两个 B0 checkpoint 做只读 `eval/no_grad` 前向，并用相同 batch 的 `feat[0]` 叶张量调用原始 `TripletLoss`；没有请求模型参数梯度，也没有 optimizer.step。原训练契约是未归一化的 1024 维 `feat[0]`，而 update0 全量检索库是独立的 L2 归一化特征，不能混用。

| seed | raw 特征范数范围 | raw loss | normalized-bank loss | hard positive 相同 | hard negative 相同 |
|---:|---:|---:|---:|:---:|:---:|
| 1234 | 31.4815–31.6272 | 0.0011238739 | 0.54416764 | True | True |
| 4321 | 31.473–31.7483 | 1.2694973e-05 | 0.45249343 | True | False |

seed4321 的 hard negative 在 raw 与 normalized 路径不同；因此旧的归一化 update0 特征只能用于离线邻域覆盖近似，不能声称复现真实训练 Batch-Hard。seed1234 本批恰好索引相同，但 loss 与梯度规模仍显著不同。

checkpoint：seed1234 `e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`；seed4321 `d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7`。模型输出为 `torch.float32`，shape `[8,1024]`，均有限。

## 数据边界

这是从 checkpoint 重建的 sampler batch，不是保存下来的历史 batch 特征；只用于输入契约和 miner 对齐。正式历史 Batch 逐步记录仍不可用。数据仅为 LTCC train，未读取官方 query/gallery/test。

## 可审计命令

`/usr/local/bin/python3.10 -u scripts/ndp_triplet_raw_feature_alignment.py --seed 1234 --output-dir outputs/ndp_triplet_alignment_20261021_134148/seed1234 --epoch 60 --rank 0`
`/usr/local/bin/python3.10 -u scripts/ndp_triplet_raw_feature_alignment.py --seed 4321 --output-dir outputs/ndp_triplet_alignment_20261021_134148/seed4321 --epoch 60 --rank 0`
