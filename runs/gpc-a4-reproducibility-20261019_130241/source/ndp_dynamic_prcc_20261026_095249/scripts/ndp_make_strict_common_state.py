#!/usr/local/bin/python3.10
"""Create a shared model-only B0 warm-start rank state for strict NDP pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ndp_v5_prcc_runner import (
    COMMON_INIT,
    base,
    configure,
    prcc,
    set_seed,
    sha256_file,
    state_hash,
    verify_common_init,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1234, 4321), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--global-step", type=int, default=1653)
    args = parser.parse_args()

    rank, device = prcc.phase6b.init_dist()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        set_seed(args.seed)
        audit = verify_common_init()
        cfg = configure(args.seed, args.output)
        state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
        dataset = prcc.build_dataset(cfg)
        model, _loss_fn, optimizer, scheduler, scaler = prcc.build_runtime(cfg, state, device, dataset)
        model_state = torch.load(args.model, map_location="cpu", weights_only=False)
        raw_model = model.module if isinstance(model, base.DDP) else model
        raw_model.load_state_dict(model_state, strict=True)

        # Save one complete rank state per PPU. The optimizer/scheduler/scaler
        # are freshly constructed from the same config, and the RNG snapshot
        # is restored by every B0/N5/NF0 arm before its first batch.
        base.save_rank_state(args.output / f"rank{rank}_state.pt", model, optimizer, scheduler, scaler, args.epoch, args.global_step, device)
        if rank == 0:
            model_path = args.output / "model.pth"
            torch.save(base.cpu_state_dict(model), model_path)
            write_json(args.output / "manifest.json", {
                "status": "PASS",
                "seed": args.seed,
                "source_model": str(args.model),
                "source_model_sha256": sha256_file(args.model),
                "model_sha256": sha256_file(model_path),
                "source_epoch": args.epoch,
                "source_global_step": args.global_step,
                "optimizer_scheduler_scaler_rng": "fresh and shared across B0/N5/NF0 rank states",
                "common_init": audit,
                "common_init_tensor_hash": state_hash(state),
                "dataset_train_images": len(dataset.train),
            })
        torch.distributed.barrier()
    finally:
        prcc.phase6b.cleanup_dist()


if __name__ == "__main__":
    main()
