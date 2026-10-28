#!/usr/local/bin/python3
"""Independent GPC-v2 LTCC trainer.

This wrapper reuses the audited Phase-7I training engine but keeps all
historical GPC-v1 files untouched.  The development split is enforced by an
active-state restriction: the original LTCC labels and frozen relation-bank
indices remain unchanged (required by the Phase-7I bank audit), while only
development-training PIDs enter the sampler.  Official query/gallery
evaluation is disabled during candidate training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
import phase7i_relation_specificity as p7  # noqa: E402

base = p7.base
b6 = p7.b6

ARM_CHOICES = ("B0", "A1", "A2", "A3", "A4", "A5", "R2")
ARM_LABELS = {
    "B0": "GPC-v1 原始 I3 matched-budget selector",
    "A1": "GPC-v2 Weak GPC，lambda=0.05",
    "A2": "GPC-v2 Strong GPC，lambda=0.20",
    "A3": "GPC-v2 Warmup GPC，前10 epoch线性升至0.10",
    "A4": "GPC-v2 PID-Balanced GPC，保持原始 pair-budget",
    "A5": "GPC-v2 Saturated GPC，冻结 tau 后使用 hinge",
    "R2": "随机关系预算控制：相同 pair-budget、无 PID 优先级",
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_split(output: Path) -> dict:
    manifest = json.loads((output / "development_split_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("train_validation_pid_disjoint") or not manifest.get("train_validation_path_disjoint"):
        raise RuntimeError("development split failed disjointness checks")
    if manifest.get("query_gallery_used_for_training"):
        raise RuntimeError("development split manifest permits query/gallery in training")
    return manifest


def arm_lambda(arm: str) -> float:
    return {"B0": 0.1, "A1": 0.05, "A2": 0.2, "A3": 0.1, "A4": 0.1, "A5": 0.1, "R2": 0.1}[arm]


def lambda_at(arm: str, epoch: int) -> float:
    if arm == "A3":
        return 0.1 * min(float(epoch) / 10.0, 1.0)
    return arm_lambda(arm)


def balanced_selection(candidates, count: int, epoch: int, rank: int, iteration: int):
    if count <= 0:
        return []
    ordered = p7.stable_selection(candidates, len(candidates), epoch, rank, iteration, "id")
    by_pid: dict[int, list] = defaultdict(list)
    for item in ordered:
        by_pid[int(item[3][0])].append(item)
    result = []
    position = 0
    pids = sorted(by_pid)
    while len(result) < count and pids:
        progressed = False
        for pid in pids:
            values = by_pid[pid]
            if position < len(values):
                result.append(values[position])
                progressed = True
                if len(result) == count:
                    break
        if not progressed:
            break
        position += 1
    if len(result) != count:
        raise RuntimeError(f"PID-balanced selector returned {len(result)} of {count} pairs")
    return result


def random_budget_selection(candidates, count: int, epoch: int, rank: int, iteration: int):
    """Deterministically randomize the fixed relation budget without PID priority."""
    if count <= 0:
        return []
    return p7.stable_selection(candidates, count, epoch, rank, iteration, "r2_random_budget")


def cumulative_pid_balanced_selection(candidates, count: int, epoch: int, rank: int, iteration: int, pid_totals: Counter):
    """Allocate a fixed budget by lowest cumulative PID exposure.

    The candidate pool remains the original I3 same-ID pool. No difficulty
    score or clothing label is used. The caller processes both PPU ranks in a
    fixed epoch order, making the quota global across the distributed batch
    stream rather than only balanced inside one batch.
    """
    if count <= 0:
        return []
    ordered = p7.stable_selection(candidates, len(candidates), epoch, rank, iteration, "a4_candidate")
    by_pid: dict[int, list] = defaultdict(list)
    for item in ordered:
        by_pid[int(item[3][0])].append(item)
    result = []
    while len(result) < count:
        available = [pid for pid, values in by_pid.items() if values]
        if not available:
            break
        pid = min(
            available,
            key=lambda value: (
                int(pid_totals[value]),
                hashlib.sha256(f"a4-tie|{epoch}|{rank}|{iteration}|{value}".encode()).hexdigest(),
            ),
        )
        result.append(by_pid[pid].pop(0))
        pid_totals[pid] += 1
    if len(result) != count:
        raise RuntimeError(f"global PID-balanced selector returned {len(result)} of {count} pairs")
    return result


_SELECTION_PLAN: dict[tuple[int, int, int], tuple[list, int, dict]] = {}


def selected_v2(entries, path_to_index, bank, arm, epoch, rank, iteration):
    candidates = p7.candidate_lists(entries, path_to_index, bank)
    count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
    planned = _SELECTION_PLAN.get((int(epoch), int(rank), int(iteration)))
    if planned is not None:
        selected, planned_count, planned_info = planned
        if int(planned_count) != int(count):
            raise RuntimeError(f"precomputed pair budget mismatch: planned={planned_count}, actual={count}")
        info = dict(planned_info)
        info["candidate_counts"] = {key: len(value) for key, value in candidates.items()}
    elif arm == "A4":
        raise RuntimeError("A4 selection plan was not prepared before the training batch")
    elif arm == "R2":
        selected = random_budget_selection(candidates["id"], count, epoch, rank, iteration)
        # Phase 7I's training loop consumes candidate_counts even for the
        # non-planned selector path.  Build it from this batch's actual
        # candidate lists; do not synthesize or zero-fill statistics.
        info = {"candidate_counts": {key: len(value) for key, value in candidates.items()}}
    else:
        selected = p7.stable_selection(candidates["id"], count, epoch, rank, iteration, "id")
        # B0 and the scalar-weight variants use the original I3 selector
        # without a precomputed plan, but retain the complete Phase 7I info
        # contract required by train_arm().
        info = {"candidate_counts": {key: len(value) for key, value in candidates.items()}}
    chosen = {"cross": [], "same": [], "id": selected}
    info.update({
        "matched_count": count,
        "selected_count": len(selected),
        "active": bool(selected),
        "keys": [item[3] for item in selected],
        "pid_counts": dict(Counter(item[3][0] for item in selected)),
        "weights": [1.0] * len(selected),
        "selector": "pid_balanced_global_epoch" if arm == "A4" else ("random_budget" if arm == "R2" else "phase7i_i3"),
    })
    info["same_clothes_selected"] = sum(int(item[3][1] == item[3][2]) for item in selected)
    info["cross_clothes_selected"] = sum(int(item[3][1] != item[3][2]) for item in selected)
    return chosen, count, info


def make_forward(arm: str, tau: float | None):
    def forward(model, loss_fn, batch, device, path_to_index, bank, _unused_arm, epoch, rank, iteration, global_step):
        writer = base.MetricCapture()
        images = batch["image"].to(device, non_blocking=False)
        pids = batch["pid"].to(device, non_blocking=False)
        camids = batch["camid"].to(device, non_blocking=False)
        clothes = batch["clothes_id"].to(device, non_blocking=False)
        captions = batch["caption_feature"].to(device, non_blocking=False)
        with amp.autocast(enabled=True):
            score, feat = model(images, cam_label=camids)
        base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=global_step)
        entries = [{
            "pid": int(pids[index].detach().cpu()),
            "clothes_id": int(clothes[index].detach().cpu()),
            "image_path": str(batch["image_path"][index]),
        } for index in range(len(pids))]
        chosen, _matched_count, info = selected_v2(entries, path_to_index, bank, arm, epoch, rank, iteration)
        representation = F.normalize(feat[0].float(), dim=-1)
        values = []
        for i, j, _key, _state in chosen["id"]:
            cosine = torch.sum(representation[i] * representation[j])
            values.append(F.relu(float(tau) - cosine) if arm == "A5" else (1.0 - cosine))
        relation = torch.stack(values).mean() if values else representation.sum() * 0.0
        applied_lambda = lambda_at(arm, epoch)
        total = base_total + applied_lambda * relation
        components = getattr(loss_fn, "last_components", {})
        info["relation_loss"] = relation
        info["selected_kind"] = "id"
        info["lambda_applied"] = applied_lambda
        return {
            "total": total,
            "ce": components.get("ce", torch.tensor(float("nan"), device=device)),
            "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)),
            "bio": components.get("bio", torch.tensor(float("nan"), device=device)),
            "relation_raw": relation,
            "relation_applied": relation,
            "relation_info": info,
            "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
            "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1],
            "writer": writer,
        }
    return forward


def install_protocol(args, split: dict, run_dir: Path, tau: float | None):
    train_pids = set(range(77)) if args.full_data else set(map(int, split["train_pids"]))
    original_configure = p7.configure
    original_full_active_state = b6.full_active_state
    original_make_loader = base.make_epoch_loader
    original_logger_for = base.logger_for
    original_build_val_loader = base.build_val_loader
    original_evaluate = base.evaluate_ltcc

    p7.PHASE = f"gpc_v2_ltcc_{args.run_time}_{args.stage}_{args.arm}_seed{args.seed}"
    b6.PHASE = p7.PHASE
    base.PHASE = p7.PHASE
    p7.SEED = args.seed
    b6.SEED = args.seed
    base.SEED = args.seed
    p7.TOTAL_EPOCHS = args.epochs
    p7.WARMUP_EPOCHS = 20
    p7.LAMBDA_REL = arm_lambda(args.arm)
    b6.NUM_CLASSES = len(train_pids)
    b6.NUM_CLOTHING_STATES = len({(int(row_pid), int(cloth)) for row_pid, cloth in []})  # set below after dataset load
    p7.TRAIN_DIRS = {"I3": run_dir}
    _SELECTION_PLAN.clear()
    planned_epochs: set[int] = set()
    plan_context: dict[str, Any] = {"path_to_index": None, "bank": None}

    def configure_v2():
        cfg = original_configure()
        cfg.defrost()
        cfg.DATA.ROOT = "/home/data"
        cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"
        cfg.DATA.NUM_WORKERS = 0
        cfg.DATA.PIN_MEMORY = False
        cfg.SOLVER.SEED = args.seed
        cfg.SOLVER.MAX_EPOCHS = args.epochs
        cfg.SOLVER.CHECKPOINT_PERIOD = args.epochs
        cfg.OUTPUT_DIR = str(run_dir)
        cfg.freeze()
        return cfg

    def active_state_v2(dataset):
        values: dict[int, set[int]] = defaultdict(set)
        for item in dataset.train:
            if int(item["pid"]) in train_pids:
                values[int(item["pid"])].add(int(item["clothes_id"]))
        if set(values) != train_pids:
            raise RuntimeError("active state does not cover exactly the requested LTCC training PIDs")
        b6.NUM_CLOTHING_STATES = sum(len(value) for value in values.values())
        return {pid: sorted(clothes) for pid, clothes in sorted(values.items())}

    def loader_v2(dataset, transform, active_state, index_map, epoch, rank):
        loader, meta = original_make_loader(dataset, transform, active_state, index_map, epoch, rank)
        if args.arm in ("A4", "R2") and epoch not in planned_epochs:
            if plan_context["path_to_index"] is None:
                plan_context["path_to_index"] = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
                plan_context["bank"] = b6.parse_relation_bank(dataset, configure_v2())
                plan_context["bank"]["path_to_index"] = plan_context["path_to_index"]

            def batch_entries(local_loader, batch_indices):
                return [{
                    "pid": int(local_loader.dataset.samples[index][0]["pid"]),
                    "clothes_id": int(local_loader.dataset.samples[index][0]["clothes_id"]),
                    "image_path": str(local_loader.dataset.samples[index][0]["image_path"]),
                } for index in batch_indices]

            def build_global_plan(loaders_by_rank):
                records = []
                for plan_rank, plan_loader in loaders_by_rank.items():
                    for iteration, batch_indices in enumerate(plan_loader.batch_sampler, start=1):
                        entries = batch_entries(plan_loader, batch_indices)
                        candidates = p7.candidate_lists(entries, plan_context["path_to_index"], plan_context["bank"])
                        count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
                        records.append((iteration, plan_rank, candidates, count))
                records.sort(key=lambda item: (item[0], item[1]))
                result = {}
                pid_totals = Counter()
                for plan_iteration, plan_rank, candidates, count in records:
                    if args.arm == "A4":
                        selected = cumulative_pid_balanced_selection(candidates["id"], count, epoch, plan_rank, plan_iteration, pid_totals)
                        selector_name = "pid_balanced_global_epoch"
                    else:
                        selected = random_budget_selection(candidates["id"], count, epoch, plan_rank, plan_iteration)
                        selector_name = "random_budget"
                    info = {
                        "candidate_counts": {key: len(value) for key, value in candidates.items()},
                        "matched_count": count,
                        "selected_count": len(selected),
                        "active": bool(selected),
                        "keys": [item[3] for item in selected],
                        "pid_counts": dict(Counter(item[3][0] for item in selected)),
                        "weights": [1.0] * len(selected),
                        "selector": selector_name,
                    }
                    result[(int(epoch), int(plan_rank), int(plan_iteration))] = (selected, count, info)
                return result

            if rank == 0:
                loaders_by_rank = {0: loader}
                for plan_rank in range(1, int(p7.WORLD_SIZE)):
                    loaders_by_rank[plan_rank], _ = original_make_loader(dataset, transform, active_state, index_map, epoch, plan_rank)
                holder = [build_global_plan(loaders_by_rank)]
                dist.broadcast_object_list(holder, src=0)
            else:
                holder = [None]
                dist.broadcast_object_list(holder, src=0)
            _SELECTION_PLAN.update(holder[0])
            planned_epochs.add(int(epoch))
        remap = {pid: idx for idx, pid in enumerate(sorted(train_pids))}
        meta = dict(meta)
        meta["covered_pids"] = [remap[int(pid)] for pid in meta["covered_pids"]]
        return loader, meta

    def fake_val_loader(_cfg, _dataset):
        return None

    def fake_evaluate(_cfg, _raw_model, _dataset, _val_loader, _device, _epoch, _logger):
        return {"cc_rank1": 0.0, "cc_rank5": 0.0, "cc_rank10": 0.0, "cc_mAP": 0.0,
                "general_rank1": 0.0, "general_rank5": 0.0, "general_rank10": 0.0, "general_mAP": 0.0}

    def logger_v2(_path):
        return original_logger_for(Path("/home/DIFFER/logs") / f"GPC训练_{args.stage}_{args.arm}_seed{args.seed}_{args.run_time}.log")

    p7.configure = configure_v2
    b6.full_active_state = active_state_v2
    base.make_epoch_loader = loader_v2
    base.build_val_loader = fake_val_loader
    base.evaluate_ltcc = fake_evaluate
    base.logger_for = logger_v2
    p7.selected_relation_pairs = selected_v2
    p7.forward_batch = make_forward(args.arm, tau)
    return {
        "restore": (original_configure, original_full_active_state, original_make_loader, original_logger_for, original_build_val_loader, original_evaluate),
        "train_pids": sorted(train_pids),
        "tau": tau,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--arm", choices=ARM_CHOICES, required=True)
    parser.add_argument("--stage", choices=("screen", "two_seed", "formal"), default="screen")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--full-data", action="store_true", help="use all 77 LTCC train PIDs for frozen formal validation")
    args = parser.parse_args()
    output = args.output
    split = load_split(output)
    tau = None
    threshold_path = output / "saturation_threshold.json"
    if args.arm == "A5":
        if not threshold_path.is_file():
            raise RuntimeError("A5 requires the pre-frozen saturation_threshold.json")
        tau = float(json.loads(threshold_path.read_text(encoding="utf-8"))["tau"])
    run_dir = output / "training" / args.stage / f"{args.arm}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = install_protocol(args, split, run_dir, tau)
    rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    try:
        p7.train_arm("I3")
        if rank == 0:
            write_json(run_dir / "v2_arm_manifest.json", {
                "phase": "DIFFER+GPC-v2 LTCC",
                "stage": args.stage,
                "arm": args.arm,
                "label": ARM_LABELS[args.arm],
                "seed": args.seed,
                "epochs": args.epochs,
                "development_split_manifest": str(output / "development_split_manifest.json"),
                "development_split_sha256": sha256_file(output / "development_split_manifest.json"),
                "train_pid_count": len(protocol["train_pids"]),
                "train_pids": protocol["train_pids"],
                "full_ltcc_train": bool(args.full_data),
                "official_query_gallery_evaluation_during_training": False,
                "gpc_v1_selector_preserved": True,
                "selector": (
                    "PID-balanced global epoch cumulative selector over same I3 candidate pool"
                    if args.arm == "A4" else
                    ("deterministic random selector over same I3 candidate pool" if args.arm == "R2" else "Phase7I I3 deterministic SHA-256 ID-only selector")
                ),
                "pair_budget": "same I3 per-batch minimum count; no additional relation pairs",
                "lambda": arm_lambda(args.arm),
                "lambda_schedule": "0.1*min(epoch/10,1)" if args.arm == "A3" else "constant",
                "saturation_tau": tau,
                "training_log": f"/home/DIFFER/logs/GPC训练_{args.stage}_{args.arm}_seed{args.seed}_{args.run_time}.log",
            })
    finally:
        # The process exits immediately after p7.train_arm; restoring globals
        # keeps this module safe when imported by a diagnostic harness.
        restore = protocol["restore"]
        p7.configure, b6.full_active_state, base.make_epoch_loader, base.logger_for, base.build_val_loader, base.evaluate_ltcc = restore


if __name__ == "__main__":
    main()
