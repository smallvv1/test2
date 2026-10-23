#!/usr/local/bin/python3.10
"""E60 Bio+GPC paired continuation with the smallest trainable NDP.

The runner imports the frozen GPC-v1 continuation engine but does not patch its
source.  NDP uses only the E60/update0 train-only reference bank and the fixed
K=5 graph.  The graph endpoints remain frozen; only the current anchor feature
comes from the live model, so the NDP term has a direct autograd path to EVA02.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
GRAPH_ROOT = ROOT / "outputs/ndp_true_multipid_20261021_103952/reference_graph"
FEATURE_ROOT = ROOT / "outputs/ndp_margin_consistency_20261021_220823/feature_export"
WORLD_SIZE = 2
STEPS_PER_EPOCH = 596
START_EPOCH = 61
SOURCE_GLOBAL_STEP = 35760
EPSILON = 0.02
LAMBDA_NDP = 0.05
LAMBDA_GPC = 0.1
TARGETS = (20, 50, 100, 200, 400)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE / "scripts"))
import gpc_v2_train as v2  # noqa: E402


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_reference(seed: int, feature_path: Path | None = None, graph_path: Path | None = None) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Load an update0-only reference bank and a fixed K=5 graph.

    The defaults preserve the historical N1 runner.  V2 passes a full
    train-only feature bank and its independently written 2048-anchor graph.
    """
    feature_path = feature_path or (FEATURE_ROOT / f"seed{seed}/k5_trajectory_features_seed{seed}.npz")
    graph_path = graph_path or (GRAPH_ROOT / f"seed{seed}/multi_pid_reference_graph.json")
    data = np.load(feature_path, allow_pickle=False)
    if "updates" in data.files:
        updates = [int(value) for value in data["updates"]]
        if 0 not in updates:
            raise RuntimeError(f"reference bank lacks update0: {feature_path}")
        update0 = data["features"][updates.index(0)].astype(np.float32)
    elif "update" in data.files and int(np.asarray(data["update"]).item()) == 0:
        update0 = data["features"].astype(np.float32)
    else:
        raise RuntimeError(f"reference bank has no explicit update0: {feature_path}")
    update0 = update0 / np.maximum(np.linalg.norm(update0, axis=1, keepdims=True), 1e-12)
    image_to_local = {int(value): index for index, value in enumerate(data["image_index"])}
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    by_anchor: dict[int, list[dict[str, Any]]] = {}
    for relation_index, row in enumerate(graph["relations"]):
        if row.get("graph_type") != "difficult" or int(row.get("k_pid", -1)) != 5:
            continue
        if int(row.get("source_update", 0)) != 0:
            continue
        anchor_id = int(row["anchor_image_index"])
        positive_ids = [int(value) for value in row["positive_indices"]]
        negative_ids = [int(value) for value in row["negative_indices"]]
        ids = [anchor_id, *positive_ids, *negative_ids]
        if any(value not in image_to_local for value in ids):
            raise RuntimeError(f"reference graph endpoint absent from bank: seed={seed}, relation={relation_index}")
        anchor = update0[image_to_local[anchor_id]]
        positives = update0[[image_to_local[value] for value in positive_ids]]
        negatives = update0[[image_to_local[value] for value in negative_ids]]
        margin = float((positives @ anchor).mean() - (negatives @ anchor).max())
        # N1 is pre-registered as historical-correct + current degradation.
        # At this start point, historical correctness is update0 correctness;
        # future continuation nodes are never read to form this gate.
        if margin <= 0.0:
            continue
        negative_pids = row.get("negative_pid", [])
        if not isinstance(negative_pids, list):
            negative_pids = [int(negative_pids)]
        by_anchor.setdefault(anchor_id, []).append({
            "relation_index": relation_index,
            "anchor_image_index": anchor_id,
            "positive_local": [image_to_local[value] for value in positive_ids],
            "negative_local": [image_to_local[value] for value in negative_ids],
            "reference_margin": margin,
            "anchor_pid": int(row["anchor_pid"]),
            "negative_pid": [int(value) for value in negative_pids],
        })
    for values in by_anchor.values():
        values.sort(key=lambda item: int(item["relation_index"]))
    meta = {
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "graph_path": str(graph_path),
        "graph_sha256": sha256_file(graph_path),
        "reference_update": 0,
        "reference_definition": "mean positive cosine - max negative-PID cosine; same definition used for current margin",
        "candidate_relation_count": int(sum(1 for row in graph["relations"] if row.get("graph_type") == "difficult" and int(row.get("k_pid", -1)) == 5 and int(row.get("source_update", 0)) == 0)),
        "historical_correct_relation_count": int(sum(len(values) for values in by_anchor.values())),
        "historical_gate": "M_ref(update0)>0",
        "current_gate": "ReLU(M_ref-M_current-epsilon)",
        "epsilon": EPSILON,
        "future_updates_read_for_selection": [],
    }
    return by_anchor, {"features": update0, "image_to_local": image_to_local, **meta}


def make_b0_gpc_ndp_forward(
    seed: int,
    reference_rows: dict[int, list[dict[str, Any]]],
    reference_features: torch.Tensor,
    stats: dict[str, Any],
    diagnostic_steps: int,
    ndp_enabled: bool = True,
):
    def forward(model, loss_fn, batch, device, path_to_index, bank, _arm, epoch, rank, iteration, global_step):
        writer = v2.base.MetricCapture()
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
        chosen, _matched_count, gpc_info = v2.p7.selected_relation_pairs(entries, path_to_index, bank, "B0", epoch, rank, iteration)
        representation = F.normalize(feat[0].float(), dim=-1)
        selected_gpc = chosen["id"]
        if selected_gpc:
            gpc_values = [1.0 - torch.sum(representation[i] * representation[j]) for i, j, _key, _state in selected_gpc]
            gpc_loss = torch.stack(gpc_values).mean()
        else:
            gpc_loss = representation.sum() * 0.0
        main_total = base_total + LAMBDA_GPC * gpc_loss

        ndp_values: list[torch.Tensor] = []
        selected_ndp = 0
        active_ndp = 0
        relation_ids: list[int] = []
        for left, entry in enumerate(entries):
            if not ndp_enabled:
                break
            anchor_id = int(path_to_index[entry["image_path"]])
            for relation in reference_rows.get(anchor_id, []):
                selected_ndp += 1
                relation_ids.append(int(relation["relation_index"]))
                anchor = representation[left]
                positive = reference_features[relation["positive_local"]]
                negative = reference_features[relation["negative_local"]]
                current_margin = (positive @ anchor).mean() - (negative @ anchor).max()
                hinge = F.relu(torch.as_tensor(float(relation["reference_margin"] - EPSILON), dtype=current_margin.dtype, device=device) - current_margin)
                ndp_values.append(hinge)
                active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
        if ndp_values:
            ndp_loss = torch.stack(ndp_values).mean()
        else:
            ndp_loss = representation.sum() * 0.0
        total = main_total + (LAMBDA_NDP * ndp_loss if ndp_enabled else ndp_loss)

        # These are diagnostic autograd calls on the feature output only; no
        # optimizer step occurs between them and the real combined backward.
        # ``global_step`` continues from E60 (35760); diagnostics are counted
        # from the beginning of this continuation, not from absolute training.
        need_diag = int(stats["batches"]) < diagnostic_steps
        if need_diag:
            g_main_feat = torch.autograd.grad(main_total, feat[0], retain_graph=True, allow_unused=True)[0]
            g_ndp_feat = torch.autograd.grad(ndp_loss, feat[0], retain_graph=True, allow_unused=True)[0]
            main_norm = float(torch.linalg.vector_norm(g_main_feat.float()).detach().cpu()) if g_main_feat is not None else 0.0
            ndp_norm = float(torch.linalg.vector_norm(g_ndp_feat.float()).detach().cpu()) if g_ndp_feat is not None else 0.0
        else:
            main_norm = float("nan")
            ndp_norm = float("nan")
        components = getattr(loss_fn, "last_components", {})
        stats["selected_relations"] += selected_ndp
        stats["active_relations"] += active_ndp
        stats["active_batches"] += int(active_ndp > 0)
        stats["ndp_loss_sum"] += float(ndp_loss.detach().float().cpu())
        stats["gpc_loss_sum"] += float(gpc_loss.detach().float().cpu())
        stats["batches"] += 1
        stats["relation_ids"].extend(relation_ids)
        gpc_info = dict(gpc_info)
        gpc_info.update({
            "ndp_selected_count": selected_ndp if ndp_enabled else 0,
            "ndp_active_count": active_ndp,
            "ndp_active": bool(active_ndp) if ndp_enabled else False,
            "ndp_relation_ids": relation_ids,
            "ndp_main_feature_grad_norm": ndp_norm,
            "main_feature_grad_norm": main_norm,
        })
        return {
            "total": total,
            "base_total": base_total,
            "ce": components.get("ce", torch.tensor(float("nan"), device=device)),
            "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)),
            "bio": components.get("bio", torch.tensor(float("nan"), device=device)),
            "gpc": gpc_loss,
            "ndp": ndp_loss,
            "main_total": main_total,
            "relation_info": gpc_info,
            "main_feature_grad_norm": main_norm,
            "ndp_feature_grad_norm": ndp_norm if ndp_enabled else 0.0,
            "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
            "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1],
        }
    return forward


def save_rank_node(run_dir: Path, model, optimizer, scheduler, scaler, device, effective_updates: int, epoch_tag: int, global_step: int, rank: int) -> dict[str, Any]:
    node = run_dir / f"node_update{effective_updates:04d}"
    node.mkdir(parents=True, exist_ok=True)
    rank_state = node / f"rank{rank}_state.pt"
    v2.base.save_rank_state(rank_state, model, optimizer, scheduler, scaler, epoch_tag, global_step, device)
    model_hash = None
    if rank == 0:
        raw = model.module if isinstance(model, v2.base.DDP) else model
        model_path = node / "model.pth"
        v2.base.save_model_checkpoint(raw, model_path)
        model_hash = sha256_file(model_path)
    holder = [model_hash]
    dist.broadcast_object_list(holder, src=0)
    dist.barrier()
    return {"effective_updates": effective_updates, "epoch_tag": epoch_tag, "global_step": global_step, "rank_state": str(rank_state), "model": str(node / "model.pth"), "model_sha256": holder[0], "optimizer_step_executed": True}


def model_grad_norm(raw_model: torch.nn.Module) -> float:
    total = torch.zeros((), device=next(raw_model.parameters()).device, dtype=torch.float32)
    for parameter in raw_model.parameters():
        if parameter.grad is not None:
            total = total + torch.sum(parameter.grad.detach().float() ** 2)
    return float(torch.sqrt(total).detach().cpu())


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = v2.base.device_guard(rank) if hasattr(v2.base, "device_guard") else None
    if device is None:
        if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError("PPU-only NDP training requires exactly two visible PPU devices")
        torch.cuda.set_device(rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{rank}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = OLD_OUTPUT / "training/two_seed" / f"B0_seed{args.seed}"
    source_state_path = source_dir / f"rank{rank}_state_epoch060.pt"
    source_state = torch.load(source_state_path, map_location="cpu", weights_only=False)
    if int(source_state["epoch"]) != 60 or int(source_state["global_step"]) != SOURCE_GLOBAL_STEP:
        raise RuntimeError(f"E60 source state mismatch: {source_state['epoch']}/{source_state['global_step']}")

    class Args:
        output = OLD_OUTPUT
        run_time = "20261015_155822"
        arm = "B0"
        stage = "two_seed"
        seed = args.seed
        epochs = 60
        full_data = False

    split = v2.load_split(OLD_OUTPUT)
    protocol = v2.install_protocol(Args, split, args.run_dir, None)
    cfg = v2.p7.configure()
    common_state, _init_audit = v2.b6.load_common_init()
    model, loss_fn, optimizer, scheduler, scaler = v2.base.setup_runtime(cfg, common_state, device)
    model.module.load_state_dict(source_state["model"], strict=True)
    optimizer.load_state_dict(source_state["optimizer"])
    scheduler.load_state_dict(source_state["scheduler"])
    scaler.load_state_dict(source_state["scaler"])
    if args.initial_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = float(args.initial_lr)
    v2.base.restore_rng(source_state["rng"], device)
    model.train()

    local_hash = v2.p7.model_parameter_hash(model)
    hashes = [None] * WORLD_SIZE
    dist.all_gather_object(hashes, local_hash)
    if len(set(hashes)) != 1:
        raise RuntimeError(f"restored rank model mismatch: {hashes}")

    dataset = v2.base.build_dataset(cfg)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    transform_train, _ = v2.base.build_img_transforms(cfg)
    index_map = v2.base.build_index_map(dataset)
    active_state = v2.b6.full_active_state(dataset)
    bank = v2.b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    reference_rows, reference_meta = load_reference(args.seed, args.reference_feature, args.graph)
    reference_features = torch.as_tensor(reference_meta["features"], dtype=torch.float32, device=device)
    if reference_features.shape[1] != 1024:
        raise RuntimeError(f"reference feature dimension mismatch: {reference_features.shape}")
    stats: dict[str, Any] = {"batches": 0, "active_batches": 0, "selected_relations": 0, "active_relations": 0, "ndp_loss_sum": 0.0, "gpc_loss_sum": 0.0, "relation_ids": []}
    forward = make_b0_gpc_ndp_forward(args.seed, reference_rows, reference_features, stats, args.diagnostic_steps, ndp_enabled=args.arm == "N2")
    raw_model = model.module if isinstance(model, v2.base.DDP) else model
    probe_name, probe_param = next(((name, parameter) for name, parameter in raw_model.named_parameters() if name.startswith("blocks.0") and parameter.requires_grad), ("", None))
    if probe_param is None:
        raise RuntimeError("could not find a block0 parameter for NDP gradient probe")

    if rank == 0:
        write_json(args.run_dir / "training_config.json", {
            "run_time": args.run_id, "seed": args.seed, "mode": args.mode, "arm": args.arm, "max_effective_updates": args.max_effective_updates,
            "source_e60_state_rank0": str(source_dir / "rank0_state_epoch060.pt"), "source_e60_state_rank1": str(source_dir / "rank1_state_epoch060.pt"),
            "source_e60_sha256_rank0": sha256_file(source_dir / "rank0_state_epoch060.pt"), "source_e60_sha256_rank1": sha256_file(source_dir / "rank1_state_epoch060.pt"),
            "initial_lr": [float(group["lr"]) for group in optimizer.param_groups], "initial_lr_requested": args.initial_lr, "initial_scaler_scale": float(scaler.get_scale()),
            "base_loss": "CE + Triplet + clipBio + 0.1 * original GPC-v1 matched-budget relation loss",
            "ndp_loss": "mean(ReLU(M_ref(update0)-epsilon-M_current)) over historical-correct fixed K=5 relations whose current anchor is in the batch" if args.arm == "N2" else "disabled for matched B0 control",
            "lambda_ndp": LAMBDA_NDP, "epsilon": EPSILON, "reference_stop_gradient": True, "current_anchor_grad_path": True,
            "gpc_selector_untouched": True, "sampler_untouched": True, "future_updates_read_for_selection": [],
            "reference_meta": reference_meta, "probe_parameter": probe_name, "restored_model_hash_by_rank": hashes,
            "save_targets": args.save_targets,
        })

    metrics_path = args.run_dir / "training_steps.csv"
    events_path = args.run_dir / "amp_events.csv"
    fields = ["batch_index", "epoch_tag", "iteration", "global_step", "effective_updates", "total", "main_total", "base_total", "ce", "triplet", "bio", "gpc", "lambda_gpc", "ndp", "lambda_ndp", "ndp_selected_count", "ndp_active_count", "ndp_active", "ndp_feature_grad_norm", "main_feature_grad_norm", "total_model_grad_norm", "ndp_probe_param_grad_norm", "lr", "scale_before", "scale_after", "found_inf", "step_executed", "parameter_finite", "elapsed_seconds"]
    if rank == 0:
        mf = metrics_path.open("w", newline="", encoding="utf-8")
        ef = events_path.open("w", newline="", encoding="utf-8")
        mw = csv.DictWriter(mf, fieldnames=fields); mw.writeheader(); mf.flush()
        ew = csv.DictWriter(ef, fieldnames=["rank", "batch_index", "global_step", "effective_updates", "found_inf", "step_executed", "scale_before", "scale_after"]); ew.writeheader(); ef.flush()
    else:
        mf = ef = mw = ew = None

    effective_updates = 0
    batch_index = 0
    saved_nodes: list[dict[str, Any]] = []
    saved_effective: set[int] = set()
    started = time.time()
    try:
        while effective_updates < args.max_effective_updates:
            epoch_tag = START_EPOCH + batch_index // STEPS_PER_EPOCH
            loader, plan_meta = v2.base.make_epoch_loader(dataset, transform_train, active_state, index_map, epoch_tag, rank)
            if int(plan_meta["batches"]) != STEPS_PER_EPOCH or not plan_meta["batch_size_ok"] or not plan_meta["pair_size_ok"]:
                raise RuntimeError(f"sampler audit failed: {plan_meta}")
            iterator = iter(loader)
            for _ in range(STEPS_PER_EPOCH):
                batch = next(iterator)
                batch_index += 1
                iteration = (batch_index - 1) % STEPS_PER_EPOCH + 1
                global_step = SOURCE_GLOBAL_STEP + batch_index
                optimizer.zero_grad(set_to_none=True)
                result = forward(model, loss_fn, batch, device, path_to_index, bank, "B0", epoch_tag, rank, iteration, global_step)
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    raise RuntimeError(f"non-finite total loss at batch={batch_index}")
                scale_before = float(scaler.get_scale())
                probe_grad = torch.autograd.grad(result["ndp"], probe_param, retain_graph=True, allow_unused=True)[0] if batch_index <= args.diagnostic_steps else None
                probe_norm = float(torch.linalg.vector_norm(probe_grad.float()).detach().cpu()) if probe_grad is not None else float("nan")
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                total_norm = model_grad_norm(raw_model) if batch_index <= args.diagnostic_steps else float("nan")
                scaler.step(optimizer)
                found_inf = int(v2.base.found_inf(scaler, optimizer))
                scaler.update()
                scale_after = float(scaler.get_scale())
                step_executed = found_inf == 0
                if step_executed:
                    effective_updates += 1
                finite = finite_parameters(raw_model) if step_executed else True
                if not finite:
                    raise RuntimeError(f"parameter became non-finite at batch={batch_index}")
                flag = torch.tensor([found_inf], dtype=torch.int64, device=device); dist.all_reduce(flag, op=dist.ReduceOp.MIN); min_inf = int(flag.item())
                flag = torch.tensor([found_inf], dtype=torch.int64, device=device); dist.all_reduce(flag, op=dist.ReduceOp.MAX); max_inf = int(flag.item())
                if min_inf != max_inf:
                    raise RuntimeError(f"rank AMP decision mismatch: batch={batch_index}, min={min_inf}, max={max_inf}")
                if rank == 0:
                    components = getattr(loss_fn, "last_components", {})
                    info = result["relation_info"]
                    mw.writerow({
                        "batch_index": batch_index, "epoch_tag": epoch_tag, "iteration": iteration, "global_step": global_step, "effective_updates": effective_updates,
                        "total": float(total.detach().float().cpu()), "main_total": float(result["main_total"].detach().float().cpu()), "base_total": float(result["base_total"].detach().float().cpu()),
                        "ce": float(components.get("ce", torch.tensor(float("nan"))).detach().float().cpu()), "triplet": float(components.get("triplet", torch.tensor(float("nan"))).detach().float().cpu()), "bio": float(components.get("bio", torch.tensor(float("nan"))).detach().float().cpu()),
                        "gpc": float(result["gpc"].detach().float().cpu()), "lambda_gpc": LAMBDA_GPC, "ndp": float(result["ndp"].detach().float().cpu()), "lambda_ndp": LAMBDA_NDP,
                        "ndp_selected_count": int(info["ndp_selected_count"]), "ndp_active_count": int(info["ndp_active_count"]), "ndp_active": bool(info["ndp_active"]),
                        "ndp_feature_grad_norm": result["ndp_feature_grad_norm"], "main_feature_grad_norm": result["main_feature_grad_norm"], "total_model_grad_norm": total_norm, "ndp_probe_param_grad_norm": probe_norm,
                        "lr": float(optimizer.param_groups[0]["lr"]), "scale_before": scale_before, "scale_after": scale_after, "found_inf": found_inf, "step_executed": step_executed, "parameter_finite": finite, "elapsed_seconds": time.time() - started,
                    }); mf.flush()
                    if found_inf:
                        ew.writerow({"rank": rank, "batch_index": batch_index, "global_step": global_step, "effective_updates": effective_updates, "found_inf": found_inf, "step_executed": step_executed, "scale_before": scale_before, "scale_after": scale_after}); ef.flush()
                if effective_updates in args.save_targets and effective_updates not in saved_effective:
                    node = save_rank_node(args.run_dir, model, optimizer, scheduler, scaler, device, effective_updates, epoch_tag, global_step, rank)
                    saved_effective.add(effective_updates)
                    if rank == 0:
                        saved_nodes.append(node); write_json(args.run_dir / "saved_nodes.json", saved_nodes)
                if effective_updates >= args.max_effective_updates:
                    break
            del iterator, loader
            gc.collect()
        if rank == 0:
            write_json(args.run_dir / "run_summary.json", {
                "run_id": args.run_id, "seed": args.seed, "mode": args.mode, "effective_updates": effective_updates, "batch_index": batch_index,
                "saved_nodes": saved_nodes, "optimizer_step_executed": True, "failure": None, "elapsed_seconds": time.time() - started,
                "ndp_candidate_relations": reference_meta["candidate_relation_count"], "ndp_historical_correct_relations": reference_meta["historical_correct_relation_count"],
                "ndp_selected_relations_local_rank0": stats["selected_relations"], "ndp_active_relations_local_rank0": stats["active_relations"], "ndp_active_batches_local_rank0": stats["active_batches"],
                "ndp_active_rate_local_rank0": stats["active_batches"] / max(stats["batches"], 1), "ndp_mean_loss_local_rank0": stats["ndp_loss_sum"] / max(stats["batches"], 1),
            })
        # Validate that the just-written checkpoint is loadable, without
        # changing the live model or running another optimizer step.
        if args.max_effective_updates in saved_effective:
            path = args.run_dir / f"node_update{args.max_effective_updates:04d}" / f"rank{rank}_state.pt"
            restored = torch.load(path, map_location="cpu", weights_only=False)
            required = {"phase", "epoch", "global_step", "model", "optimizer", "scheduler", "scaler", "rng"}
            if set(restored) != required:
                raise RuntimeError(f"saved checkpoint fields mismatch: {sorted(restored)}")
            write_json(args.run_dir / f"restore_check_rank{rank}.json", {"status": "PASS", "path": str(path), "sha256": sha256_file(path), "fields": sorted(restored)})
        dist.barrier()
    except Exception as exc:
        if rank == 0:
            write_json(args.run_dir / "failure.json", {"status": "FAIL", "type": type(exc).__name__, "message": str(exc), "batch_index": batch_index, "effective_updates": effective_updates})
        raise
    finally:
        if rank == 0:
            if mf is not None: mf.close()
            if ef is not None: ef.close()
        v2.b6.cleanup_dist()


def finite_parameters(model: torch.nn.Module) -> bool:
    raw = model.module if isinstance(model, v2.base.DDP) else model
    return all(bool(torch.isfinite(parameter.detach()).all().item()) for parameter in raw.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", choices=(1234, 4321), type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("B0", "N2"), default="N2")
    parser.add_argument("--reference-feature", type=Path, default=None)
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--initial-lr", type=float, default=1e-6)
    parser.add_argument("--max-effective-updates", type=int, default=None)
    parser.add_argument("--save-updates", type=str, default="20,50,100,200,400")
    parser.add_argument("--diagnostic-steps", type=int, default=20)
    args = parser.parse_args()
    args.max_effective_updates = args.max_effective_updates or (20 if args.mode == "smoke" else 400)
    args.save_targets = tuple(sorted({int(value) for value in args.save_updates.split(",") if value.strip()}))
    if args.max_effective_updates not in args.save_targets:
        args.save_targets = tuple(sorted(set(args.save_targets) | {args.max_effective_updates}))
    run(args)


if __name__ == "__main__":
    main()
