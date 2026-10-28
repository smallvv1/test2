#!/usr/local/bin/python3
"""双 PPU 接口预检：覆盖 B0/A4 的真实训练前向和 Phase 7I 统计接口。

本脚本只做技术门禁，不保存 checkpoint，也不改变正式训练的 optimizer、AMP
或科学协议。每次只检查一个 arm，由 torchrun 启动两个 rank；B0 和 A4
分别运行，避免复用已释放的计算图或污染另一分支的 selector 状态。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/DIFFER/scripts")
import gpc_a4_recheck_train as trainer  # noqa: E402

p7 = trainer.p7
base = trainer.base
b6 = trainer.b6


def finite_grads(model: torch.nn.Module) -> bool:
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            return False
    return True


def relation_info_audit(info: dict, entries: list[dict], path_to_index: dict, bank: dict) -> dict:
    candidates = p7.candidate_lists(entries, path_to_index, bank)
    expected_counts = {key: len(value) for key, value in candidates.items()}
    if info.get("candidate_counts") != expected_counts:
        raise RuntimeError(f"candidate_counts mismatch: actual={info.get('candidate_counts')} expected={expected_counts}")
    matched = min(expected_counts["cross"], expected_counts["same"], expected_counts["id"])
    if int(info.get("matched_count", -1)) != matched:
        raise RuntimeError(f"matched_count mismatch: {info.get('matched_count')} != {matched}")
    keys = list(info.get("keys", []))
    if int(info.get("selected_count", -1)) != len(keys):
        raise RuntimeError("selected_count is not the number of selected relation keys")
    expected_pid_counts = dict(Counter(int(key[0]) for key in keys))
    actual_pid_counts = {int(key): int(value) for key, value in info.get("pid_counts", {}).items()}
    if actual_pid_counts != expected_pid_counts:
        raise RuntimeError(f"pid_counts mismatch: actual={actual_pid_counts} expected={expected_pid_counts}")
    if list(info.get("weights", [])) != [1.0] * len(keys):
        raise RuntimeError("weights do not match the original unit-weight GPC reduction")
    if bool(info.get("active")) != bool(keys):
        raise RuntimeError("active does not match selected relation count")
    if set(info) < {"candidate_counts", "matched_count", "selected_count", "active", "keys", "pid_counts", "weights"}:
        raise RuntimeError("relation_info is missing a Phase 7I field")
    return {
        "candidate_counts": expected_counts,
        "matched_count": matched,
        "selected_count": len(keys),
        "active": bool(keys),
        "pid_counts": expected_pid_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--arm", choices=("B0", "A4"), required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 3:
        raise SystemExit("接口预检至少运行三个连续有效 batch")

    rank, device = b6.init_dist()
    protocol = None
    local = {"rank": rank, "device": str(device), "arm": args.arm, "steps": args.steps}
    try:
        split = trainer.load_split(args.output)
        run_dir = args.output / "interface_preflight" / args.arm
        run_dir.mkdir(parents=True, exist_ok=True)
        protocol_args = SimpleNamespace(
            full_data=False,
            seed=1234,
            epochs=1,
            stage="interface_preflight",
            arm=args.arm,
            run_time=args.run_time,
        )
        original_selected = p7.selected_relation_pairs
        original_forward = p7.forward_batch
        protocol = trainer.install_protocol(protocol_args, split, run_dir, None)
        cfg = p7.configure()
        b6.set_seed(1234)
        state, init_audit = b6.load_common_init()
        model, loss_fn, optimizer, _scheduler, _old_scaler = base.setup_runtime(cfg, state, device)
        # Reduced scale is a technical recovery check only. Formal training
        # continues to use the frozen trainer's original AMP configuration.
        scaler = amp.GradScaler(init_scale=128.0)
        dataset = base.build_dataset(cfg)
        transform_train, _transform_test = base.build_img_transforms(cfg)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        bank = b6.parse_relation_bank(dataset, cfg)
        bank["path_to_index"] = path_to_index
        active_state = b6.full_active_state(dataset)
        index_map = base.build_index_map(dataset)
        loader, loader_meta = base.make_epoch_loader(dataset, transform_train, active_state, index_map, 1, rank)
        # Inspect the real Phase 7I batch plan without decoding images first.
        # This keeps the technical preflight short while preserving the exact
        # sampler, candidate bank, and original iteration numbers.
        batch_records = []
        for original_iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
            entries = [
                {
                    "pid": int(loader.dataset.samples[index][0]["pid"]),
                    "clothes_id": int(loader.dataset.samples[index][0]["clothes_id"]),
                    "image_path": str(loader.dataset.samples[index][0]["image_path"]),
                }
                for index in batch_indices
            ]
            candidates = p7.candidate_lists(entries, path_to_index, bank)
            matched = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
            batch_records.append((original_iteration, list(batch_indices), entries, matched))
        inactive_record = next((record for record in batch_records if record[3] == 0), None)
        active_records = [record for record in batch_records if record[3] > 0][:args.steps]
        if len(active_records) < args.steps:
            raise RuntimeError(f"真实 sampler 仅找到 {len(active_records)} 个有效 batch，少于要求的 {args.steps} 个")
        selected_records = ([inactive_record] if inactive_record is not None else []) + active_records
        preflight_sampler = type(loader.batch_sampler)([record[1] for record in selected_records])
        selected_loader = DataLoader(
            loader.dataset,
            batch_sampler=preflight_sampler,
            num_workers=0,
            pin_memory=False,
            collate_fn=loader.collate_fn,
        )
        iterator = iter(selected_loader)
        step_rows = []
        local_pid_counts = Counter()
        local_candidate_counts = Counter()
        local_selected = 0
        local_matched = 0
        local_active = 0
        scanned_batches = len(batch_records)
        active_updates = 0
        active_streak = 0
        for logical_step, record in enumerate(selected_records, start=1):
            iteration = int(record[0])
            batch = next(iterator)
            entries = [
                {
                    "pid": int(batch["pid"][index]),
                    "clothes_id": int(batch["clothes_id"][index]),
                    "image_path": str(batch["image_path"][index]),
                }
                for index in range(len(batch["pid"]))
            ]
            optimizer.zero_grad(set_to_none=False)
            result = trainer.make_forward(args.arm, None)(
                model, loss_fn, batch, device, path_to_index, bank,
                "I3", 1, rank, iteration, iteration,
            )
            total = result["total"]
            if not bool(torch.isfinite(total.detach()).item()):
                raise RuntimeError(f"non-finite joint loss at rank={rank}, step={iteration}")
            audited = relation_info_audit(result["relation_info"], entries, path_to_index, bank)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grads_finite = finite_grads(model)
            if not grads_finite:
                raise RuntimeError(f"non-finite gradient at rank={rank}, step={iteration}")
            first_parameter = next(model.parameters())
            before = first_parameter.detach().clone()
            scaler.step(optimizer)
            found_inf = int(base.found_inf(scaler, optimizer) > 0)
            scaler.update()
            after = first_parameter.detach()
            parameter_changed = bool(torch.isfinite(after).all().item()) and bool((after - before).abs().sum().item() > 0.0)
            if found_inf:
                raise RuntimeError(f"AMP skipped optimizer step at rank={rank}, step={iteration}")
            if not parameter_changed:
                raise RuntimeError(f"optimizer did not update a finite parameter at rank={rank}, step={iteration}")
            local_candidate_counts.update(audited["candidate_counts"])
            local_selected += audited["selected_count"]
            local_matched += audited["matched_count"]
            local_active += int(audited["active"])
            if audited["active"]:
                active_updates += 1
                active_streak += 1
            else:
                active_streak = 0
            local_pid_counts.update(audited["pid_counts"])
            step_rows.append({
                "iteration": iteration,
                "loss": float(total.detach().float().cpu()),
                "candidate_counts": audited["candidate_counts"],
                "matched_count": audited["matched_count"],
                "selected_count": audited["selected_count"],
                "active": audited["active"],
                "pid_counts": audited["pid_counts"],
                "gradients_finite": grads_finite,
                "amp_found_inf": found_inf,
                "optimizer_parameter_changed": parameter_changed,
            })
        local.update({
            "status": "PASS",
            "loader_meta": loader_meta,
            "steps": step_rows,
            "scanned_batches": scanned_batches,
            "active_updates": active_updates,
            "required_active_updates": args.steps,
            "executed_batch_count": len(selected_records),
            "inactive_batch_tested": inactive_record is not None,
            "final_active_streak": active_streak,
            "candidate_counts_total": dict(local_candidate_counts),
            "matched_count_total": local_matched,
            "selected_count_total": local_selected,
            "active_batch_count": local_active,
            "pid_counts_total": dict(local_pid_counts),
            "amp_skip_count": 0,
            "model_device": str(next(model.parameters()).device),
            "model_on_ppu": str(next(model.parameters()).device).startswith("cuda:"),
            "common_init_sha256": init_audit["sha256"],
        })
        gathered = [None] * 2
        dist.all_gather_object(gathered, local)
        dist.barrier()
        if rank == 0:
            aggregate = {
                "candidate_counts_total": dict(Counter({
                    key: sum(int(item["candidate_counts_total"].get(key, 0)) for item in gathered)
                    for key in ("cross", "same", "id")
                })),
                "matched_count_total": sum(int(item["matched_count_total"]) for item in gathered),
                "selected_count_total": sum(int(item["selected_count_total"]) for item in gathered),
                "active_batch_count": sum(int(item["active_batch_count"]) for item in gathered),
                "pid_counts_total": dict(Counter({
                    str(pid): sum(int(item["pid_counts_total"].get(str(pid), item["pid_counts_total"].get(pid, 0))) for item in gathered)
                    for pid in set().union(*(set(item["pid_counts_total"]) for item in gathered))
                })),
            }
            payload = {
                "status": "PASS" if all(item.get("status") == "PASS" and item["model_on_ppu"] for item in gathered) else "FAIL",
                "run_time": args.run_time,
                "arm": args.arm,
                "steps_per_rank": args.steps,
                "ranks": gathered,
                "aggregate": aggregate,
                "relation_info_contract": ["candidate_counts", "matched_count", "selected_count", "active", "keys", "pid_counts", "weights"],
                "algorithm_changed": False,
                "official_query_gallery_used": False,
                "checkpoint_written": False,
                "amp_init_scale": 128.0,
            }
            output_path = args.output / f"interface_preflight_{args.arm}.json"
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            if payload["status"] != "PASS":
                raise RuntimeError(f"{args.arm} 接口预检失败")
    finally:
        if protocol is not None:
            restore = protocol["restore"]
            p7.configure, b6.full_active_state, base.make_epoch_loader, base.logger_for, base.build_val_loader, base.evaluate_ltcc = restore
        if protocol is not None:
            p7.selected_relation_pairs = original_selected
            p7.forward_batch = original_forward
        b6.cleanup_dist()


if __name__ == "__main__":
    main()
