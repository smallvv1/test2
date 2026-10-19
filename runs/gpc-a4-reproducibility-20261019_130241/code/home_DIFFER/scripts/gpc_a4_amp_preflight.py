#!/usr/local/bin/python3
"""Technical AMP preflight using the real Bio+GPC forward path.

This script is a hardware/runtime check only. It does not write a checkpoint,
does not alter the scientific trainer, and uses a reduced GradScaler initial
scale to distinguish PPU AMP overflow from model or graph failures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda import amp

import sys
sys.path.insert(0, "/data/DIFFER/scripts")
import phase7i_relation_specificity as p7  # noqa: E402

base = p7.base
b6 = p7.b6


def configure():
    cfg = p7.configure()
    cfg.defrost()
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.MAX_EPOCHS = 3
    cfg.freeze()
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-scale", type=float, default=128.0)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    rank, device = b6.init_dist()
    local = {"rank": rank, "device": str(device), "steps": args.steps, "init_scale": args.init_scale}
    try:
        cfg = configure()
        b6.set_seed(1234)
        state, init_audit = b6.load_common_init()
        model, loss_fn, optimizer, _scheduler, _old_scaler = base.setup_runtime(cfg, state, device)
        scaler = amp.GradScaler(init_scale=args.init_scale)
        dataset = base.build_dataset(cfg)
        transform_train, _transform_test = base.build_img_transforms(cfg)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        bank = b6.parse_relation_bank(dataset, cfg)
        bank["path_to_index"] = path_to_index
        active_state = b6.full_active_state(dataset)
        loader, _meta = base.make_epoch_loader(dataset, transform_train, active_state, base.build_index_map(dataset), 1, rank)
        iterator = iter(loader)
        skips = []
        losses = []
        for iteration in range(1, args.steps + 1):
            batch = next(iterator)
            optimizer.zero_grad(set_to_none=False)
            result = p7.forward_batch(model, loss_fn, batch, device, path_to_index, bank, "I3", 1, rank, iteration, iteration)
            loss = result["total"]
            if not bool(torch.isfinite(loss.detach()).item()):
                raise RuntimeError(f"联合 loss 非有限，rank={rank}, iteration={iteration}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            found_inf = int(base.found_inf(scaler, optimizer) > 0)
            scaler.update()
            losses.append(float(loss.detach().float().cpu()))
            skips.append(found_inf)
        local.update({
            "losses_finite": all(math.isfinite(value) for value in losses),
            "losses": losses,
            "amp_skips": skips,
            "amp_skip_count": sum(skips),
            "model_device": str(next(model.parameters()).device),
            "model_on_ppu": str(next(model.parameters()).device).startswith("cuda:"),
            "final_scale": float(scaler.get_scale()),
            "common_init_sha256": init_audit["sha256"],
        })
        gathered = [None] * 2
        dist.all_gather_object(gathered, local)
        dist.barrier()
        if rank == 0:
            payload = {
                "status": "PASS" if all(item["losses_finite"] and item["amp_skip_count"] == 0 and item["model_on_ppu"] for item in gathered) else "FAIL",
                "technical_recovery": "GradScaler init_scale=128; scientific trainer remains unchanged",
                "runtime": {key: os.environ.get(key) for key in ("PPU_HOME", "PPU_SDK", "LD_LIBRARY_PATH")},
                "ranks": gathered,
                "official_query_gallery_used": False,
                "checkpoint_written": False,
            }
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "amp_preflight.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if payload["status"] != "PASS":
                raise RuntimeError("AMP 技术预检仍有 skipped step")
    finally:
        b6.cleanup_dist()


if __name__ == "__main__":
    main()
