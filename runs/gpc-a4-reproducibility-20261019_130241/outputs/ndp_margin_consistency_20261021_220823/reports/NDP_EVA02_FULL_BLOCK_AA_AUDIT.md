# EVA02 完整模型/Block22 A/A 无更新反向审计

使用固定物化 train-only batch、固定 E60 checkpoint、同 AMP 配置、同 PPU 环境，两个独立进程重复 forward/backward，均 `found_inf=0`、loss 相同、未执行 optimizer.step。

| seed | 全量 hook/参数最大相对 L2 | 最大差异位置 | loss | found_inf |
|---|---:|---|---:|---:|
|1234|0.00411885|`blocks.0.attn.proj.weight`|6.56139946|0/0|
|4321|0.00412521|`blocks.0.attn.proj.weight`|4.35378170|0/0|

固定输入、固定上游梯度的独立 LayerNorm 与真实 `blocks.22.norm1` 局部 probe 稳定，上一轮 LayerNorm 巨大差异已撤回。完整模型 hook 仍显示 `blocks.22.norm1` 和 `blocks.22.attn.q_proj` 输出梯度 hash 在重复进程间不同，早期 block 参数梯度达到约0.00412相对差异。证据缩小到完整组合 backward/梯度传播路径，但尚未唯一定位到底层单一 kernel。

余弦使用 float64 dot/norm 复核，最大超界仅约1e-11，没有用 clamp 掩盖；历史 >1.0026 不是本轮主结论。A/A 仍 FAIL。

命令：`source /usr/local/PPU_SDK/envsetup.sh ppu && /usr/local/bin/python3.10 -u scripts/ndp_aa_operator_probe.py --mode amp --no-optimizer-step ...`。
