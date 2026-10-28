#!/usr/local/bin/python3.10
"""Create a complete per-rank warm-start state from one model-only t1 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import gpc_v2_train as gpc  # noqa: E402,F401
import phase7a_prcc_relation_validation as prcc  # noqa: E402
import ndp_v5_prcc_runner as runner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--t1-model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--global-step", type=int, default=2204)
    args = parser.parse_args()

    rank, device = prcc.phase6b.init_dist()
    try:
        runner.set_seed(args.seed)
        cfg = runner.configure(args.seed, args.run_dir)
        dataset = prcc.build_dataset(cfg)
        state = torch.load(runner.COMMON_INIT, map_location="cpu", weights_only=False)
        model, _loss_fn, optimizer, scheduler, scaler = prcc.build_runtime(cfg, state, device, dataset)
        model_state = torch.load(args.t1_model, map_location="cpu", weights_only=False)
        raw_model = model.module if isinstance(model, runner.base.DDP) else model
        raw_model.load_state_dict(model_state, strict=True)
        payload = {
            "phase": "ndp_dynamic_top5_common_warmstart",
            "epoch": int(args.epoch),
            "global_step": int(args.global_step),
            "model": runner.base.cpu_state_dict(model),
            "optimizer": runner.base.tensor_to_cpu(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": runner.base.rng_state(device),
        }
        args.run_dir.mkdir(parents=True, exist_ok=True)
        rank_path = args.run_dir / f"rank{rank}_state.pt"
        torch.save(payload, rank_path)
        dist.barrier()
        if rank == 0:
            model_path = args.run_dir / "model.pth"
            torch.save(runner.base.cpu_state_dict(model), model_path)
            manifest = {
                "status": "PASS",
                "seed": args.seed,
                "epoch": args.epoch,
                "global_step": args.global_step,
                "initialization": "t1 model-only checkpoint + newly constructed optimizer/scheduler/scaler/RNG",
                "t1_model": str(args.t1_model),
                "t1_model_sha256": runner.sha256_file(args.t1_model),
                "model_sha256": runner.sha256_file(model_path),
                "rank_states": [str(args.run_dir / f"rank{r}_state.pt") for r in range(runner.WORLD_SIZE)],
                "optimizer": "fresh PRCC Bio+GPC runtime optimizer; no historical t1 optimizer state was available",
                "gpc_source_sha256": runner.sha256_file(ENGINE / "scripts/gpc_v2_train.py"),
            }
            (args.run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        dist.barrier()
    finally:
        prcc.phase6b.cleanup_dist()


if __name__ == "__main__":
    main()
