#!/usr/bin/env python3
"""固定物化 LTCC batch 的双 PPU DDP 梯度归约探针。

不执行 optimizer.step；比较每个 rank 的未归约梯度显式平均与真实 DDP
backward 后梯度，避免把单卡反向差异误判为 all-reduce 问题。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path("/home/DIFFER")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
import ndp_aa_reproducibility as aa  # noqa: E402


def digest(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    x, y = a.detach().cpu().double().flatten(), b.detach().cpu().double().flatten()
    d = x - y
    nx, ny = float(torch.linalg.vector_norm(x)), float(torch.linalg.vector_norm(y))
    return {"max_abs": float(d.abs().max()), "relative_l2": float(torch.linalg.vector_norm(d)) / max(ny, 1e-12),
            "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
            "hash_a": digest(a), "hash_b": digest(b)}


def one_backward(model, batch, loss_fn, optimizer, scaler, device, ddp=False):
    optimizer.zero_grad(set_to_none=True)
    images = batch["image"].to(device); pids = batch["pid"].to(device)
    cams = batch["camid"].to(device); clothes = batch["clothes_id"].to(device)
    caps = batch["caption_feature"].to(device)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=cams)
        writer = type("Writer", (), {"add_scalar": lambda self, *args, **kwargs: None})()
        loss = loss_fn(score, feat, pids, cams, caps, clothes_ids=clothes, train_writer=writer, step=1)
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    wanted = {"patch_embed.proj.weight", "blocks.0.attn.q_proj.weight", "blocks.11.attn.q_proj.weight", "blocks.22.attn.q_proj.weight", "head.weight"}
    return loss.detach().float().cpu(), {name.removeprefix("module."): p.grad.detach().clone() for name, p in model.named_parameters()
                                        if name.removeprefix("module.") in wanted and p.grad is not None}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--materialized", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 2: raise RuntimeError(f"需要2 rank，实际 {world}")
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method="env://")
    try:
        materialized = torch.load(args.materialized, map_location="cpu", weights_only=False)
        if int(materialized["seed"]) != args.seed: raise RuntimeError("seed mismatch")
        v2, model, loss_fn, optimizer, _scheduler, scaler, source = aa.load_runtime(args.seed, materialized, args.output / f"rank{rank}", device)
        batch = materialized["batches"][0]
        base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        loss_local, local_grads = one_backward(model, batch, loss_fn, optimizer, scaler, device)
        model.load_state_dict(base_state, strict=True)
        optimizer.zero_grad(set_to_none=True)
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=0.0)
        ddp_scaler = amp.GradScaler(init_scale=v2.base.INITIAL_SCALE)
        loss_ddp, ddp_grads = one_backward(ddp_model, batch, loss_fn, ddp_opt, ddp_scaler, device)
        gathered = {}
        comparisons = {}
        for name, local in local_grads.items():
            reduced = local.to(device).clone(); dist.all_reduce(reduced, op=dist.ReduceOp.SUM); reduced.div_(world)
            all_local = [torch.empty_like(reduced) for _ in range(world)]; dist.all_gather(all_local, local.to(device))
            if rank == 0:
                explicit = torch.stack([x.cpu().double() for x in all_local]).mean(0).float()
                comparisons[name] = compare(ddp_grads[name], explicit)
                gathered[name] = {"local_grad_hashes": [digest(x) for x in all_local], "explicit_avg_hash": digest(explicit), "ddp_hash": digest(ddp_grads[name])}
            dist.barrier()
        record = {"rank": rank, "local_rank": local_rank, "seed": args.seed, "world_size": world,
                  "checkpoint_sha256": materialized["state_sha256"], "batch_hash": materialized["batch_fingerprints"][0]["batch_hash"],
                  "selected_relation_count": len(materialized["selected_plan"].get(1, [])),
                  "local_loss": float(loss_local), "ddp_loss": float(loss_ddp), "local_grads": {k: digest(v) for k,v in local_grads.items()},
                  "ddp_grads": {k: digest(v) for k,v in ddp_grads.items()}, "comparisons_rank0": comparisons if rank == 0 else None, "gathered_rank0": gathered if rank == 0 else None}
        args.output.mkdir(parents=True, exist_ok=True); (args.output / f"rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
        dist.barrier()
        if rank == 0:
            print(json.dumps({"status":"PASS", "comparisons": comparisons}, ensure_ascii=False), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__": main()
