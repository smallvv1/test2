#!/usr/local/bin/python3
"""GPC-v2 双 PPU 最小 forward/backward/optimizer.step 预检。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 预检失败：torch.cuda.is_available() 为 False，禁止 CPU 回退")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"PPU 预检失败：要求两张卡，实际为 {torch.cuda.device_count()} 张")
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if any(name != "PPU-ZW810E" for name in names):
        raise RuntimeError(f"PPU 预检失败：设备名称异常 {names}")
    rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world = int(os.environ.get("WORLD_SIZE", "-1"))
    if world != 2 or rank not in (0, 1):
        raise RuntimeError(f"PPU 预检失败：要求 WORLD_SIZE=2 且 LOCAL_RANK 为 0/1，实际 {world}/{rank}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", init_method="env://")
    result = {"rank": rank, "world_size": world, "device": str(device), "device_name": names[rank]}
    try:
        torch.manual_seed(20261015 + rank)
        torch.cuda.manual_seed_all(20261015 + rank)
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        x = torch.randn(4, 32, device=device)
        target = torch.randn(4, 8, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if output.device != device or loss.device != device:
            raise RuntimeError(f"forward 张量离开 PPU：output={output.device}, loss={loss.device}")
        if not grads or any(grad.device != device for grad in grads):
            raise RuntimeError("backward 未产生 PPU 梯度")
        if any(not bool(torch.isfinite(grad).all().item()) for grad in grads):
            raise RuntimeError("backward 产生 NaN/Inf 梯度")
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.step()
        changed = any(not bool(torch.equal(old, new.detach())) for old, new in zip(before, model.parameters()))
        if not changed:
            raise RuntimeError("optimizer.step 未改变 PPU 模型参数")
        checksum = torch.stack([grad.detach().float().abs().mean() for grad in grads]).sum()
        dist.all_reduce(checksum, op=dist.ReduceOp.SUM)
        dist.barrier()
        result.update({
            "forward_device": str(output.device),
            "loss_device": str(loss.device),
            "gradient_devices": sorted({str(grad.device) for grad in grads}),
            "optimizer_parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
            "loss": float(loss.detach().cpu()),
            "gradient_checksum_all_ranks": float(checksum.detach().cpu()),
            "optimizer_step_changed_parameters": changed,
            "passed": True,
        })
        print("GPC_V2_PPU_PREFLIGHT_PASS " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GPC_V2_PPU_PREFLIGHT_FAIL 类型={type(exc).__name__} 信息={exc}", file=sys.stderr, flush=True)
        raise
