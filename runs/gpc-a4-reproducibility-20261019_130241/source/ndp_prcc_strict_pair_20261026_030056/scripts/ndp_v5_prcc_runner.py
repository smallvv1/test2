#!/usr/local/bin/python3.10
"""PRCC B0/N5/NF0 paired experiment runner.

This file is intentionally independent from the frozen LTCC continuation
runner.  It reuses the released PRCC Bio-only model/loss/sampler/evaluator and
adds only the already-frozen GPC-v1 relation term plus NDP-V5.  The fixed GPC
source under /data/DIFFER is imported, never edited.
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
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import gpc_v2_train as gpc  # noqa: E402
import phase5b_id_damage_ds as base  # noqa: E402
import phase7a_prcc_relation_validation as prcc  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402


WORLD_SIZE = 2
BATCH_SIZE = 16
NUM_INSTANCES = 4
EPOCH_STEPS = 551
MAX_SCHEDULE_EPOCHS = 60
EPSILON = 0.02
LAMBDA_NDP = 0.05
LAMBDA_GPC = 0.1
INITIAL_SCALE = 512
COMMON_INIT = ENGINE / "outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT.pth"
COMMON_META = ENGINE / "outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT_HASHES.json"
DEFAULT_FEATURE = ROOT / "log_NDP" / "ndp_v5_prcc_reference" / "prcc_update0_features.npz"
DEFAULT_GRAPH = ROOT / "log_NDP" / "ndp_v5_prcc_reference" / "prcc_v5_k5_graph.json"


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_hash(state: dict[str, torch.Tensor]) -> str:
    items = {}
    for key, value in sorted(state.items()):
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        items[key] = hashlib.sha256(raw).hexdigest()
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def configure(seed: int, run_dir: Path) -> Any:
    cfg = prcc.configure(run_dir)
    cfg.defrost()
    # PRCC images are mounted at /home/data while the caption bank is mounted
    # at /data/data on this server.  The released config uses the former for
    # both, so the independent runner makes the existing asset mount explicit.
    cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/PRCC"
    cfg.SOLVER.SEED = seed
    cfg.SOLVER.MAX_EPOCHS = MAX_SCHEDULE_EPOCHS
    cfg.SOLVER.CHECKPOINT_PERIOD = MAX_SCHEDULE_EPOCHS
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()
    return cfg


def verify_common_init() -> dict[str, Any]:
    if not COMMON_INIT.is_file() or not COMMON_META.is_file():
        raise FileNotFoundError(f"PRCC common initialization missing: {COMMON_INIT}")
    meta = json.loads(COMMON_META.read_text(encoding="utf-8"))
    state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    if sha256_file(COMMON_INIT) != meta.get("state_sha256"):
        raise RuntimeError("PRCC common-init file SHA256 mismatch")
    if state_hash(state) != meta.get("state_hash"):
        raise RuntimeError("PRCC common-init tensor hash mismatch")
    if meta.get("model", {}).get("reverse") != "OFF" or meta.get("model", {}).get("active_losses") != ["ce", "triplet", "clipBio"]:
        raise RuntimeError("common init topology is not Bio-only")
    return {"path": str(COMMON_INIT), "sha256": sha256_file(COMMON_INIT), "state_hash": meta["state_hash"], "meta_sha256": sha256_file(COMMON_META), "meta": meta}


def export_reference(args: argparse.Namespace) -> None:
    """Export update-0 train-only features and the deterministic PRCC K=5 graph."""
    output = Path(args.feature).parent
    output.mkdir(parents=True, exist_ok=True)
    if Path(args.feature).exists() or Path(args.graph).exists():
        raise RuntimeError("refusing to overwrite existing PRCC reference artifacts")
    audit = verify_common_init()
    cfg = configure(args.seed, output)
    dataset = prcc.build_dataset(cfg)
    if len(dataset.train) != 17896 or int(dataset.num_train_pids) != 150:
        raise RuntimeError(f"unexpected PRCC topology: {len(dataset.train)} / {dataset.num_train_pids}")
    source_checkpoint = None
    if args.checkpoint:
        source_checkpoint = Path(args.checkpoint)
        if source_checkpoint.is_dir():
            source_checkpoint = source_checkpoint / "model.pth"
        payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        source_update = int(args.source_update)
    else:
        state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
        source_update = 0
    device = torch.device("cuda:0")
    model = prcc.make_model_from_state(cfg, state, int(dataset.num_train_pids), int(dataset.num_camera)).to(device)
    model.eval()
    _, transform_test = prcc.build_img_transforms(cfg)
    loader = DataLoader(ImageDataset(dataset.train, transform=transform_test), batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    features: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=False)
            camids = batch["camid"].to(device, non_blocking=False)
            with amp.autocast(enabled=True):
                output_value = model(images, cam_label=camids)
            # EVA02 returns feat_bio directly in eval mode and
            # (score_dict, [feat_bio]) in train mode.
            if torch.is_tensor(output_value):
                identity = output_value
            elif isinstance(output_value, (tuple, list)) and len(output_value) == 2:
                _score, feat = output_value
                identity = feat[0] if isinstance(feat, (tuple, list)) else feat
            else:
                raise TypeError(f"unexpected EVA02 eval output: {type(output_value)!r}")
            identity = F.normalize(identity.float(), dim=-1).cpu().numpy().astype(np.float32)
            features.append(identity)
            if batch_index % 100 == 0:
                print(f"reference forward {batch_index}/{len(loader)}", flush=True)
    feature_array = np.concatenate(features, axis=0)
    image_index = np.arange(len(dataset.train), dtype=np.int32)
    np.savez(Path(args.feature), features=feature_array, image_index=image_index, update=np.asarray(source_update, dtype=np.int32))

    pid = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int32)
    clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int32)
    by_pid_clothes: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    by_pid: dict[int, list[int]] = defaultdict(list)
    for index, (p, c) in enumerate(zip(pid, clothes)):
        by_pid_clothes[int(p)][int(c)].append(index)
        by_pid[int(p)].append(index)
    pids = sorted(by_pid)
    pid_position = {p: i for i, p in enumerate(pids)}
    centroids = np.stack([feature_array[by_pid[p]].mean(axis=0) for p in pids]).astype(np.float32)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

    relations = []
    historical_correct = 0
    valid_anchor = 0
    for anchor in range(len(dataset.train)):
        anchor_pid = int(pid[anchor])
        anchor_cloth = int(clothes[anchor])
        positive_indices = []
        for cloth_id in sorted(by_pid_clothes[anchor_pid]):
            if cloth_id != anchor_cloth:
                positive_indices.append(by_pid_clothes[anchor_pid][cloth_id][0])
        if not positive_indices:
            positive_indices = [index for index in by_pid[anchor_pid] if index != anchor][:1]
        if not positive_indices:
            continue
        valid_anchor += 1
        centroid_scores = feature_array[anchor] @ centroids.T
        centroid_scores[pid_position[anchor_pid]] = -np.inf
        top_pid_positions = np.argsort(-centroid_scores)[:5]
        negative_indices = []
        negative_pids = []
        positive_mean = float(np.mean(feature_array[positive_indices] @ feature_array[anchor]))
        pid_margins = []
        for position in top_pid_positions:
            neg_pid = pids[int(position)]
            candidates = by_pid[neg_pid]
            scores = feature_array[candidates] @ feature_array[anchor]
            neg_index = int(candidates[int(np.argmax(scores))])
            negative_indices.append(neg_index)
            negative_pids.append(int(neg_pid))
            pid_margins.append(positive_mean - float(feature_array[neg_index] @ feature_array[anchor]))
        valid_mask = [margin > 0.0 for margin in pid_margins]
        historical_correct += int(sum(valid_mask))
        if not any(valid_mask):
            continue
        relations.append({
            "graph_type": "difficult",
            "k_pid": 5,
            "source_update": source_update,
            "anchor_image_index": anchor,
            "anchor_pid": anchor_pid,
            "positive_indices": positive_indices,
            "negative_indices": negative_indices,
            "negative_pid": negative_pids,
            "historical_pid_margin": pid_margins,
            "historical_valid_pid": valid_mask,
        })
        if anchor % 1000 == 0:
            print(f"reference graph {anchor}/{len(dataset.train)}", flush=True)
    graph = {
        "version": "ndp-v5-prcc-train-only-k5",
        "created_utc": utc_now(),
        "feature_path": str(Path(args.feature)),
        "dataset_root": "/home/data/prcc",
        "train_only": True,
        "query_gallery_used": False,
        "reference_source": "checkpoint" if source_checkpoint else "common_init",
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else str(COMMON_INIT),
        "source_checkpoint_sha256": sha256_file(source_checkpoint) if source_checkpoint else sha256_file(COMMON_INIT),
        "source_update": source_update,
        "anchor_count": valid_anchor,
        "relation_count": len(relations),
        "historical_correct_pid_count": historical_correct,
        "selection": "all PRCC train images with a legal same-PID positive; top five negative PIDs by source-checkpoint centroid cosine; one hardest reference image per negative PID",
        "relations": relations,
    }
    write_json(Path(args.graph), graph)
    write_json(output / "reference_manifest.json", {
        "common_init": audit,
        "reference_source": "checkpoint" if source_checkpoint else "common_init",
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else str(COMMON_INIT),
        "source_checkpoint_sha256": sha256_file(source_checkpoint) if source_checkpoint else sha256_file(COMMON_INIT),
        "source_update": source_update,
        "feature": {"path": str(Path(args.feature)), "sha256": sha256_file(Path(args.feature)), "shape": list(feature_array.shape)},
        "graph": {"path": str(Path(args.graph)), "sha256": sha256_file(Path(args.graph)), "anchor_count": valid_anchor, "relation_count": len(relations), "historical_correct_pid_count": historical_correct},
        "future_updates_read_for_selection": [],
        "query_gallery_used": False,
    })
    print(json.dumps({"feature": str(args.feature), "graph": str(args.graph), "feature_shape": list(feature_array.shape), "anchor_count": valid_anchor, "relation_count": len(relations), "historical_correct_pid_count": historical_correct}, indent=2), flush=True)


def load_reference(feature_path: Path, graph_path: Path, epsilon: float = EPSILON) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    data = np.load(feature_path, allow_pickle=False)
    reference_update = int(np.asarray(data["update"]).item())
    if reference_update < 0:
        raise RuntimeError("reference feature bank has invalid source update")
    features = data["features"].astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    image_to_local = {int(value): index for index, value in enumerate(data["image_index"].tolist())}
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if graph.get("query_gallery_used") or not graph.get("train_only"):
        raise RuntimeError("reference graph violates train-only rule")
    by_anchor: dict[int, list[dict[str, Any]]] = {}
    for relation_index, row in enumerate(graph["relations"]):
        anchor_id = int(row["anchor_image_index"])
        positives = [int(v) for v in row["positive_indices"]]
        negatives = [int(v) for v in row["negative_indices"]]
        if any(value not in image_to_local for value in [anchor_id, *positives, *negatives]):
            raise RuntimeError(f"reference endpoint missing at relation {relation_index}")
        anchor = features[image_to_local[anchor_id]]
        positive_values = features[[image_to_local[v] for v in positives]] @ anchor
        negative_values = features[[image_to_local[v] for v in negatives]] @ anchor
        margins = [float(positive_values.mean() - value) for value in negative_values]
        valid = [margin > 0.0 for margin in margins]
        if not any(valid):
            continue
        negative_pids = [int(v) for v in row["negative_pid"]]
        by_anchor.setdefault(anchor_id, []).append({
            "relation_index": relation_index,
            "anchor_image_index": anchor_id,
            "positive_local": [image_to_local[v] for v in positives],
            "negative_local": [image_to_local[v] for v in negatives],
            "reference_margin": float(positive_values.mean() - negative_values.max()),
            "reference_pid_margin": margins,
            "historical_valid_pid": valid,
            "anchor_pid": int(row["anchor_pid"]),
            "negative_pid": negative_pids,
        })
    meta = {
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "graph_path": str(graph_path),
        "graph_sha256": sha256_file(graph_path),
        "reference_update": reference_update,
        "reference_definition": "PRCC train-only common-init normalized feat[0]; mean legal same-PID positive cosine minus one hardest reference image per selected negative PID",
        "candidate_relation_count": int(graph.get("relation_count", 0)),
        "historical_correct_relation_count": int(sum(len(v) for v in by_anchor.values())),
        "historical_correct_pid_count": int(sum(sum(bool(x) for x in r["historical_valid_pid"]) for values in by_anchor.values() for r in values)),
        "historical_gate": "per-negative-PID M_ref(update0)>0",
        "current_gate": "ReLU(M_ref-epsilon-M_current)",
        "epsilon": float(epsilon),
        "future_updates_read_for_selection": [],
    }
    return by_anchor, {"features": features, "image_to_local": image_to_local, **meta}


def scalar(value: Any) -> float:
    return float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)


def set_gpc_pair_context(run_id: str, seed: int) -> str:
    """Set the frozen GPC selector context without encoding the NDP arm.

    phase7i stable_selection hashes PHASE, seed, epoch, rank and iteration.
    Including B0/N5 in PHASE silently changes the selected GPC pairs, so the
    arm must never be part of this context.
    """
    phase = f"ndp_v5_prcc_gpc_pair_{run_id}_seed{seed}"
    gpc.p7.PHASE = phase
    gpc.p7.SEED = seed
    return phase


def batch_entries(batch: dict[str, Any], pids: torch.Tensor, clothes: torch.Tensor) -> list[dict[str, Any]]:
    return [
        {"pid": int(pids[i].detach().cpu()), "clothes_id": int(clothes[i].detach().cpu()), "image_path": str(batch["image_path"][i])}
        for i in range(len(pids))
    ]


def ndp_target_margin(reference_margin: float, epsilon: float, arm: str) -> float:
    """Return the detached target margin for an enabled NDP variant."""
    if arm == "N5":
        return reference_margin - epsilon
    if arm == "NF0":
        return max(reference_margin - epsilon, 0.0)
    raise ValueError(f"unsupported NDP arm: {arm}")


def compose_total_loss(main_total: torch.Tensor, ndp_loss: torch.Tensor, arm: str) -> torch.Tensor:
    """Compose the fixed B0 loss or the B0+NDP loss without cross-contamination."""
    if arm == "B0":
        total = main_total
        if total is not main_total or not torch.equal(total, main_total):
            raise AssertionError("B0 total loss must be exactly main_total")
        return total
    if arm in {"N5", "NF0"}:
        return main_total + LAMBDA_NDP * ndp_loss
    raise ValueError(f"unsupported arm: {arm}")


def verify_gpc_pair(args: argparse.Namespace) -> None:
    """Compare fixed and legacy arm-dependent GPC contexts on 100 batches."""
    rank, _device = prcc.phase6b.init_dist()
    try:
        cfg = configure(args.seed, Path(args.run_dir))
        set_seed(args.seed)
        train_loader, _same, _diff, _gallery, dataset, _sampler, _val, _val_same = prcc.build_dataloader(cfg)
        bank = prcc.build_relation_bank(dataset)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        pair_phase = set_gpc_pair_context(args.run_id, args.seed)
        fixed_mismatches = []
        legacy_mismatches = []
        image_index_batches = []
        compared = 0
        for iteration, batch in enumerate(train_loader, start=1):
            if iteration > 100:
                break
            pids = batch["pid"]
            clothes = batch["clothes_id"]
            entries = batch_entries(batch, pids, clothes)
            image_index_batches.append(
                [int(path_to_index[str(path)]) for path in batch["image_path"]]
            )
            # The fixed comparison changes no selector input; it calls the
            # exact same frozen selector twice under the arm-independent phase.
            set_gpc_pair_context(args.run_id, args.seed)
            fixed_b0, _, fixed_info_b0 = gpc.selected_v2(entries, path_to_index, bank, "B0", args.check_epoch, rank, iteration)
            set_gpc_pair_context(args.run_id, args.seed)
            fixed_n5, _, fixed_info_n5 = gpc.selected_v2(entries, path_to_index, bank, "B0", args.check_epoch, rank, iteration)
            if fixed_info_b0["keys"] != fixed_info_n5["keys"]:
                fixed_mismatches.append({"iteration": iteration, "b0": fixed_info_b0["keys"], "n5": fixed_info_n5["keys"]})
            # Reproduce the old mistake only as a diagnostic: the arm in PHASE
            # should change the hash order, while the fixed context must not.
            gpc.p7.PHASE = f"ndp_v5_prcc_{args.run_id}_B0_seed{args.seed}"
            legacy_b0, _, legacy_info_b0 = gpc.selected_v2(entries, path_to_index, bank, "B0", args.check_epoch, rank, iteration)
            gpc.p7.PHASE = f"ndp_v5_prcc_{args.run_id}_N5_seed{args.seed}"
            legacy_n5, _, legacy_info_n5 = gpc.selected_v2(entries, path_to_index, bank, "B0", args.check_epoch, rank, iteration)
            if legacy_info_b0["keys"] != legacy_info_n5["keys"]:
                legacy_mismatches.append({"iteration": iteration, "b0": legacy_info_b0["keys"], "n5": legacy_info_n5["keys"]})
            compared += 1
        image_index_digest = hashlib.sha256(
            json.dumps(image_index_batches, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload = {"rank": rank, "compared_batches": compared, "fixed_phase": pair_phase, "fixed_mismatch_count": len(fixed_mismatches), "legacy_arm_phase_mismatch_count": len(legacy_mismatches), "fixed_examples": fixed_mismatches[:3], "legacy_examples": legacy_mismatches[:3], "raw_batch_image_index_sha256": image_index_digest}
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, payload)
        if rank == 0:
            write_json(Path(args.run_dir) / "gpc_pair_check.json", {"status": "PASS", "seed": args.seed, "epoch": args.check_epoch, "world_size": WORLD_SIZE, "rank_results": gathered, "fixed_mismatch_count": sum(int(item["fixed_mismatch_count"]) for item in gathered), "legacy_arm_phase_mismatch_count": sum(int(item["legacy_arm_phase_mismatch_count"]) for item in gathered), "algorithm_sha256": sha256_file(ENGINE / "scripts/gpc_v2_train.py")})
    finally:
        try:
            train_loader.shutdown()
        except Exception:
            pass
        prcc.phase6b.cleanup_dist()


def forward_batch(model, loss_fn, batch, device, path_to_index, bank, reference_rows, reference_features, arm, epoch, rank, iteration, global_step, stats, diagnostic_steps, ndp_epsilon):
    writer = base.MetricCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=global_step)
    entries = batch_entries(batch, pids, clothes)

    # This is the unchanged GPC-v1 matched-budget selector from the frozen
    # engine.  It changes no GPC candidate or positive definition.
    chosen, _matched, gpc_info = gpc.selected_v2(entries, path_to_index, bank, "B0", epoch, rank, iteration)
    representation = F.normalize(feat[0].float(), dim=-1)
    selected_gpc = chosen["id"]
    gpc_loss = torch.stack([1.0 - torch.sum(representation[i] * representation[j]) for i, j, _key, _state in selected_gpc]).mean() if selected_gpc else representation.sum() * 0.0
    main_total = base_total + LAMBDA_GPC * gpc_loss

    selected_ndp = active_ndp = current_error_pid = correct_pid = 0
    margin_sum = 0.0
    ndp_enabled = arm in {"N5", "NF0"}
    if ndp_enabled:
        ndp_values = []
        for local_index, entry in enumerate(entries):
            anchor_id = int(path_to_index[entry["image_path"]])
            anchor = representation[local_index]
            anchor_values = []
            for relation in reference_rows.get(anchor_id, []):
                positive = reference_features[relation["positive_local"]]
                negative = reference_features[relation["negative_local"]]
                positive_similarity = (positive @ anchor).mean()
                for pid_index, is_valid in enumerate(relation["historical_valid_pid"]):
                    if not is_valid:
                        continue
                    selected_ndp += 1
                    current_margin = positive_similarity - negative[pid_index] @ anchor
                    reference_margin = float(relation["reference_pid_margin"][pid_index])
                    target_margin = ndp_target_margin(reference_margin, ndp_epsilon, arm)
                    hinge = F.relu(torch.as_tensor(target_margin, dtype=current_margin.dtype, device=device) - current_margin)
                    anchor_values.append(hinge)
                    active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                    current_error_pid += int(float(current_margin.detach().float().cpu()) <= 0.0)
                    correct_pid += int(float(current_margin.detach().float().cpu()) > 0.0)
                    margin_sum += float(current_margin.detach().float().cpu())
            if anchor_values:
                ndp_values.append(torch.stack(anchor_values).mean())
        ndp_loss = torch.stack(ndp_values).mean() if ndp_values else representation.sum() * 0.0
    else:
        # B0 must not inspect or backpropagate through any NDP relation.
        ndp_loss = representation.sum() * 0.0
    total = compose_total_loss(main_total, ndp_loss, arm)

    # Batch-local Purity@10 is a training diagnostic, not the official PRCC metric.
    with torch.no_grad():
        pair = representation @ representation.t()
        pair.fill_diagonal_(-float("inf"))
        k = min(10, max(1, pair.shape[0] - 1))
        top = pair.topk(k=k, dim=1).indices
        purity = (pids[top] == pids[:, None]).float().mean()
    need_diag = stats["batches"] < diagnostic_steps and ndp_enabled
    if need_diag:
        ndp_feature_grad = torch.autograd.grad(ndp_loss, feat[0], retain_graph=True, allow_unused=True)[0]
        main_feature_grad = torch.autograd.grad(main_total, feat[0], retain_graph=True, allow_unused=True)[0]
        ndp_feature_norm = float(torch.linalg.vector_norm(ndp_feature_grad.float()).detach().cpu()) if ndp_feature_grad is not None else 0.0
        main_feature_norm = float(torch.linalg.vector_norm(main_feature_grad.float()).detach().cpu()) if main_feature_grad is not None else 0.0
    else:
        ndp_feature_norm = float("nan")
        main_feature_norm = float("nan")
    components = getattr(loss_fn, "last_components", {})
    stats["batches"] += 1
    stats["active_batches"] += int(active_ndp > 0 and ndp_enabled)
    stats["selected_relations"] += selected_ndp if ndp_enabled else 0
    stats["active_relations"] += active_ndp if ndp_enabled else 0
    stats["current_error_pid"] += current_error_pid if ndp_enabled else 0
    stats["correct_pid"] += correct_pid if ndp_enabled else 0
    stats["margin_sum"] += margin_sum if ndp_enabled else 0.0
    stats["purity_sum"] += float(purity.detach().cpu())
    stats["ndp_loss_sum"] += float(ndp_loss.detach().float().cpu()) if ndp_enabled else 0.0
    stats["gpc_selected"] += len(selected_gpc)
    gpc_info = dict(gpc_info)
    gpc_info.update({"ndp_selected_count": selected_ndp if ndp_enabled else 0, "ndp_active_count": active_ndp if ndp_enabled else 0, "ndp_active": bool(active_ndp > 0 and ndp_enabled), "current_error_pid": current_error_pid if ndp_enabled else 0, "correct_pid": correct_pid if ndp_enabled else 0, "mean_current_margin": margin_sum / max(correct_pid + current_error_pid, 1), "batch_purity10": float(purity.detach().cpu())})
    return {"total": total, "base_total": base_total, "ce": components.get("ce", torch.tensor(float("nan"), device=device)), "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)), "bio": components.get("bio", torch.tensor(float("nan"), device=device)), "gpc": gpc_loss, "ndp": ndp_loss, "main_total": main_total, "ndp_lambda": LAMBDA_NDP if ndp_enabled else 0.0, "relation_info": gpc_info, "ndp_feature_grad_norm": ndp_feature_norm, "main_feature_grad_norm": main_feature_norm, "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(), "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1]}


def finite_parameters(model: torch.nn.Module) -> bool:
    raw = model.module if isinstance(model, base.DDP) else model
    return all(bool(torch.isfinite(parameter.detach()).all().item()) for parameter in raw.parameters())


def model_grad_norm(model: torch.nn.Module) -> float:
    raw = model.module if isinstance(model, base.DDP) else model
    total = torch.zeros((), device=next(raw.parameters()).device, dtype=torch.float32)
    for parameter in raw.parameters():
        if parameter.grad is not None:
            total += torch.sum(parameter.grad.detach().float() ** 2)
    return float(torch.sqrt(total).detach().cpu())


def save_node(run_dir, model, optimizer, scheduler, scaler, device, update, epoch, global_step, rank, node_name=None):
    node = run_dir / (node_name or f"node_update{update:04d}")
    node.mkdir(parents=True, exist_ok=True)
    payload = {"phase": "ndp_v5_prcc", "epoch": epoch, "global_step": global_step, "model": base.cpu_state_dict(model), "optimizer": base.tensor_to_cpu(optimizer.state_dict()), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": base.rng_state(device)}
    torch.save(payload, node / f"rank{rank}_state.pt")
    model_hash = None
    if rank == 0:
        model_path = node / "model.pth"
        temporary = model_path.with_suffix(".tmp")
        torch.save(base.cpu_state_dict(model), temporary)
        temporary.replace(model_path)
        model_hash = sha256_file(model_path)
    holder = [model_hash]
    dist.broadcast_object_list(holder, src=0)
    dist.barrier()
    return {"update": update, "epoch": epoch, "global_step": global_step, "rank_state": str(node / f"rank{rank}_state.pt"), "model": str(node / "model.pth"), "model_sha256": holder[0]}


def logger(run_dir: Path):
    import logging
    log = logging.getLogger(f"ndp_v5_prcc.{run_dir}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    handler = logging.FileHandler(run_dir / "eval_rank0.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.addHandler(logging.StreamHandler(sys.stdout))
    return log


def run_train(args: argparse.Namespace) -> None:
    rank, device = prcc.phase6b.init_dist()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    audit = verify_common_init()
    cfg = configure(args.seed, run_dir)
    state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    model, loss_fn, optimizer, scheduler, scaler = prcc.build_runtime(cfg, state, device, prcc.build_dataset(cfg))
    if args.resume_model:
        model_state = torch.load(args.resume_model, map_location="cpu", weights_only=False)
        raw_model = model.module if isinstance(model, base.DDP) else model
        raw_model.load_state_dict(model_state, strict=True)
    train_loader, _same_loader, _diff_loader, _gallery_loader, dataset, train_sampler, val_loader, val_loader_same = prcc.build_dataloader(cfg)
    if len(dataset.train) != 17896 or len(train_loader) != EPOCH_STEPS:
        raise RuntimeError(f"PRCC schedule mismatch: train={len(dataset.train)}, steps={len(train_loader)}")
    # Rebuild from the live dataset order.  A historical PRCC bank is present
    # on the engine, but its path-order digest was produced by a different
    # mount/configuration and must not be used for GPC indexing.
    bank = prcc.build_relation_bank(dataset)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank["path_to_index"] = path_to_index
    if not math.isfinite(float(args.ndp_epsilon)) or float(args.ndp_epsilon) < 0.0:
        raise ValueError(f"invalid --ndp-epsilon: {args.ndp_epsilon}")
    if args.arm == "B0":
        # B0 is the frozen Bio+GPC control.  It must not load, inspect, or
        # contribute any NDP reference relation.
        reference_rows = {}
        reference_features = torch.empty((0, 1024), dtype=torch.float32, device=device)
        reference_meta = {"reference_source": "not used by B0 control", "reference_update": None, "epsilon": None, "future_updates_read_for_selection": []}
    else:
        reference_rows, reference_meta = load_reference(Path(args.feature), Path(args.graph), float(args.ndp_epsilon))
        reference_features = torch.as_tensor(reference_meta["features"], dtype=torch.float32, device=device)
        if reference_features.shape[0] != len(dataset.train) or reference_features.shape[1] != 1024:
            raise RuntimeError(f"reference feature shape mismatch: {tuple(reference_features.shape)}")
    gpc_phase = set_gpc_pair_context(args.run_id, args.seed)
    resume_payload = None
    start_epoch = 0
    start_global_step = 0
    if args.resume_checkpoint:
        resume_dir = Path(args.resume_checkpoint)
        rank_state_path = resume_dir / f"rank{rank}_state.pt"
        if not rank_state_path.is_file():
            raise FileNotFoundError(f"missing warm-start rank state: {rank_state_path}")
        resume_payload = base.load_rank_state(rank_state_path, model, optimizer, scheduler, scaler, device)
        start_epoch = int(resume_payload["epoch"])
        start_global_step = int(resume_payload["global_step"])
        if start_epoch < 1 or start_global_step < 1:
            raise RuntimeError(f"invalid warm-start state: epoch={start_epoch}, global_step={start_global_step}")
    elif args.resume_model:
        start_epoch = int(args.resume_epoch)
        start_global_step = int(args.resume_global_step)
        if start_epoch < 1 or start_global_step < 1:
            raise RuntimeError(f"invalid model-only warm-start state: epoch={start_epoch}, global_step={start_global_step}")
    raw_model = model.module if isinstance(model, base.DDP) else model
    probe_name, probe_param = next(((name, p) for name, p in raw_model.named_parameters() if name.startswith("blocks.0") and p.requires_grad), ("", None))
    if probe_param is None:
        raise RuntimeError("no EVA02 block parameter available for gradient probe")
    if rank == 0:
        write_json(run_dir / "training_config.json", {"run_id": args.run_id, "arm": args.arm, "seed": args.seed, "initialization": "warm-start from B0 checkpoint with complete per-rank model/optimizer/scheduler/scaler/RNG state" if args.resume_checkpoint else "PRCC_COMMON_INIT fresh optimizer; no historical PRCC E60 state exists", "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None, "resume_epoch": start_epoch, "resume_global_step": start_global_step, "continuation_updates": int(args.steps) if args.steps else int(args.epochs) * EPOCH_STEPS, "raw_epoch_mode": args.steps is None, "evaluation_segment_epochs": sorted(int(x) for x in args.eval_epochs.split(",") if x.strip()), "sampler_policy": "new explicit continuation segment; sampler.set_epoch(absolute_epoch) before every epoch", "common_init": audit, "config": cfg.dump(), "dataset": {"train_images": len(dataset.train), "train_pids": int(dataset.num_train_pids), "train_clothes": int(dataset.num_train_clothes), "steps_per_epoch": len(train_loader), "batch_size": BATCH_SIZE, "num_instances": NUM_INSTANCES}, "base_loss": "CE + Triplet + clipBio + 0.1 * original GPC-v1 matched-budget relation loss", "gpc_source_sha256": sha256_file(ENGINE / "scripts/gpc_v2_train.py"), "gpc_pair_phase": gpc_phase, "gpc_selector": "gpc_v2_train.selected_v2 with arm B0; unchanged GPC matched-budget candidate/positive logic", "ndp": {"version": "NDP-V5", "lambda": LAMBDA_NDP, "epsilon": float(args.ndp_epsilon), "per_negative_pid": True, "reference_meta": reference_meta, "reference_stop_gradient": True, "current_anchor_grad_path": True}, "future_updates_read_for_selection": [], "query_gallery_used_for_training_selection": False, "probe_parameter": probe_name})
    if rank == 0 and args.arm == "NF0":
        write_json(run_dir / "ndp_variant.json", {"version": "NDP-Floor0", "mode": "floor_zero_target", "lambda": LAMBDA_NDP, "epsilon": float(args.ndp_epsilon), "reference_correct_only": True})
    if rank == 0 and args.resume_model:
        write_json(run_dir / "model_only_warm_start.json", {"status": "MODEL_ONLY_WARM_START", "model": str(args.resume_model), "model_sha256": sha256_file(args.resume_model), "epoch": int(args.resume_epoch), "global_step": int(args.resume_global_step), "optimizer_scheduler_scaler_rng": "fresh; complete rank state was unavailable"})
        training_config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
        training_config["initialization"] = "model-only B0 warm start + fresh optimizer/scheduler/scaler/RNG"
        training_config["resume_model"] = str(args.resume_model)
        training_config["ndp"]["version"] = "NDP-Floor0" if args.arm == "NF0" else "NDP-V5"
        training_config["ndp"]["mode"] = "floor_zero_target" if args.arm == "NF0" else "reference_margin_target"
        write_json(run_dir / "training_config.json", training_config)
    fields = ["batch_index", "epoch", "iteration", "global_step", "effective_updates", "total", "base_total", "ce", "triplet", "bio", "gpc", "ndp", "lambda_ndp", "gpc_selected", "ndp_selected", "ndp_active", "current_error_pid", "correct_pid", "batch_purity10", "ndp_feature_grad_norm", "main_feature_grad_norm", "total_model_grad_norm", "ndp_probe_param_grad_norm", "lr", "scale_before", "scale_after", "found_inf", "step_executed"]
    metrics_path = run_dir / "training_steps.csv"
    amp_path = run_dir / "amp_events.csv"
    mf = metrics_path.open("w", newline="", encoding="utf-8") if rank == 0 else None
    af = amp_path.open("w", newline="", encoding="utf-8") if rank == 0 else None
    mw = csv.DictWriter(mf, fieldnames=fields) if rank == 0 else None
    aw = csv.DictWriter(af, fieldnames=["rank", "batch_index", "epoch", "global_step", "found_inf", "step_executed", "scale_before", "scale_after"]) if rank == 0 else None
    if rank == 0:
        mw.writeheader(); aw.writeheader(); mf.flush(); af.flush()
    raw_epoch_mode = args.steps is None
    steps_to_run = int(args.steps) if args.steps else int(args.epochs) * EPOCH_STEPS
    eval_epochs = set(int(x) for x in args.eval_epochs.split(",") if x.strip())
    saved_targets = set(int(x) for x in args.save_updates.split(",") if x.strip())
    if not raw_epoch_mode:
        saved_targets.add(steps_to_run)
    stats = {"batches": 0, "active_batches": 0, "selected_relations": 0, "active_relations": 0, "current_error_pid": 0, "correct_pid": 0, "margin_sum": 0.0, "purity_sum": 0.0, "ndp_loss_sum": 0.0, "gpc_selected": 0}
    saved_nodes = []
    global_step = start_global_step
    effective_updates = 0
    batch_index = 0
    log = logger(run_dir) if rank == 0 else None
    epoch_sums = defaultdict(float)
    epoch_stats = defaultdict(lambda: {"batches": 0, "active_batches": 0, "selected_relations": 0, "active_relations": 0, "current_error_pid": 0, "correct_pid": 0, "margin_sum": 0.0, "purity_sum": 0.0, "ndp_loss_sum": 0.0, "gpc_selected": 0})
    started = time.time()
    try:
        epoch = start_epoch
        segment_epoch = 0
        while (segment_epoch < int(args.epochs)) if raw_epoch_mode else (effective_updates < steps_to_run):
            segment_epoch += 1
            epoch = start_epoch + segment_epoch
            scheduler.step(epoch)
            train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            for iteration, batch in enumerate(iterator, start=1):
                if not raw_epoch_mode and effective_updates >= steps_to_run:
                    break
                batch_index += 1; global_step += 1
                optimizer.zero_grad(set_to_none=True)
                before = dict(stats)
                result = forward_batch(model, loss_fn, batch, device, path_to_index, bank, reference_rows, reference_features, args.arm, epoch, rank, iteration, global_step, stats, args.diagnostic_steps, float(args.ndp_epsilon))
                if not bool(torch.isfinite(result["total"].detach()).item()):
                    raise RuntimeError(f"non-finite total loss at epoch={epoch}, iteration={iteration}")
                diag_probe = None
                if stats["batches"] <= args.diagnostic_steps and args.arm in {"N5", "NF0"}:
                    diag_probe = torch.autograd.grad(result["ndp"], probe_param, retain_graph=True, allow_unused=True)[0]
                scale_before = float(scaler.get_scale())
                scaler.scale(result["total"]).backward()
                scaler.unscale_(optimizer)
                total_norm = model_grad_norm(model) if stats["batches"] <= args.diagnostic_steps else float("nan")
                scaler.step(optimizer)
                found_inf = int(base.found_inf(scaler, optimizer))
                scaler.update()
                scale_after = float(scaler.get_scale())
                step_executed = found_inf == 0
                if step_executed:
                    effective_updates += 1
                if not finite_parameters(model):
                    raise RuntimeError("non-finite model parameter after optimizer step")
                minmax = torch.tensor([found_inf, found_inf], dtype=torch.int64, device=device)
                dist.all_reduce(minmax, op=dist.ReduceOp.MIN); min_inf = int(minmax[0].item())
                dist.all_reduce(minmax, op=dist.ReduceOp.MAX); max_inf = int(minmax[1].item())
                if min_inf != max_inf:
                    raise RuntimeError(f"rank AMP decision mismatch: min={min_inf}, max={max_inf}")
                if rank == 0:
                    info = result["relation_info"]
                    mw.writerow({"batch_index": batch_index, "epoch": epoch, "iteration": iteration, "global_step": global_step, "effective_updates": effective_updates, "total": scalar(result["total"]), "base_total": scalar(result["base_total"]), "ce": scalar(result["ce"]), "triplet": scalar(result["triplet"]), "bio": scalar(result["bio"]), "gpc": scalar(result["gpc"]), "ndp": scalar(result["ndp"]), "lambda_ndp": result["ndp_lambda"], "gpc_selected": len(info.get("keys", [])), "ndp_selected": info["ndp_selected_count"], "ndp_active": info["ndp_active_count"], "current_error_pid": info["current_error_pid"], "correct_pid": info["correct_pid"], "batch_purity10": info["batch_purity10"], "ndp_feature_grad_norm": result["ndp_feature_grad_norm"], "main_feature_grad_norm": result["main_feature_grad_norm"], "total_model_grad_norm": total_norm, "ndp_probe_param_grad_norm": float(torch.linalg.vector_norm(diag_probe.float()).detach().cpu()) if diag_probe is not None else float("nan"), "lr": float(optimizer.param_groups[0]["lr"]), "scale_before": scale_before, "scale_after": scale_after, "found_inf": found_inf, "step_executed": step_executed}); mf.flush()
                    if found_inf:
                        aw.writerow({"rank": rank, "batch_index": batch_index, "epoch": epoch, "global_step": global_step, "found_inf": found_inf, "step_executed": step_executed, "scale_before": scale_before, "scale_after": scale_after}); af.flush()
                if not raw_epoch_mode and effective_updates in saved_targets and effective_updates not in {item["update"] for item in saved_nodes}:
                    node = save_node(run_dir, model, optimizer, scheduler, scaler, device, effective_updates, epoch, global_step, rank)
                    if rank == 0:
                        saved_nodes.append(node); write_json(run_dir / "saved_nodes.json", saved_nodes)
                for key in ("total", "base_total", "ce", "triplet", "bio", "gpc", "ndp", "id_acc", "bio_acc"):
                    epoch_sums[key] += scalar(result[key])
                if not raw_epoch_mode and effective_updates >= steps_to_run:
                    break
            train_loader.shutdown()
            should_stop = (not raw_epoch_mode and effective_updates >= steps_to_run) or (raw_epoch_mode and segment_epoch >= int(args.epochs))
            should_report = (not raw_epoch_mode and effective_updates >= steps_to_run) or (raw_epoch_mode and segment_epoch in eval_epochs)
            if raw_epoch_mode and segment_epoch in eval_epochs:
                node_name = f"node_rawepoch{segment_epoch:02d}_update{effective_updates:04d}"
                node = save_node(run_dir, model, optimizer, scheduler, scaler, device, effective_updates, epoch, global_step, rank, node_name=node_name)
                if rank == 0:
                    saved_nodes.append(node); write_json(run_dir / "saved_nodes.json", saved_nodes)
            if should_report:
                if rank == 0 and args.steps is None:
                    eval_metrics = prcc.evaluate_prcc(cfg, raw_model, dataset, val_loader, val_loader_same, device, epoch, log)
                else:
                    eval_metrics = {}
                holder = [eval_metrics]; dist.broadcast_object_list(holder, src=0); eval_metrics = holder[0]
                # Reduce the per-epoch values accumulated since the previous report.
                values = torch.tensor([epoch_sums[k] / max(1, EPOCH_STEPS) for k in ("total", "ce", "triplet", "bio", "gpc", "ndp", "id_acc", "bio_acc")], dtype=torch.float64, device=device)
                dist.all_reduce(values, op=dist.ReduceOp.SUM); values /= WORLD_SIZE
                counts = torch.tensor([stats["active_batches"], stats["selected_relations"], stats["active_relations"], stats["current_error_pid"], stats["correct_pid"], stats["margin_sum"], stats["purity_sum"], stats["ndp_loss_sum"], stats["gpc_selected"], stats["batches"]], dtype=torch.float64, device=device)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                if rank == 0:
                    row = {"arm": args.arm, "seed": args.seed, "segment_epoch": segment_epoch, "epoch": epoch, "raw_batches": batch_index, "effective_updates": effective_updates, "train_total": float(values[0].cpu()), "ce": float(values[1].cpu()), "triplet": float(values[2].cpu()), "bio": float(values[3].cpu()), "gpc": float(values[4].cpu()), "ndp": float(values[5].cpu()), "id_accuracy": float(values[6].cpu()), "bio_accuracy": float(values[7].cpu()), "ndp_active_batches": int(counts[0].item()), "ndp_selected_relations": int(counts[1].item()), "ndp_active_relations": int(counts[2].item()), "ndp_active_rate": float(counts[0].item() / max(1.0, counts[9].item())), "current_error_pid": int(counts[3].item()), "correct_pid": int(counts[4].item()), "mean_current_margin": float(counts[5].item() / max(1.0, counts[3].item() + counts[4].item())), "purity10_batch_mean": float(counts[6].item() / max(1.0, counts[9].item())), "ndp_mean_loss": float(counts[7].item() / max(1.0, counts[9].item())), "gpc_selected_relations": int(counts[8].item()), "lr": float(optimizer.param_groups[0]["lr"]), "cc_rank1": eval_metrics.get("cc_rank1"), "cc_mAP": eval_metrics.get("cc_mAP"), "general_rank1": eval_metrics.get("general_rank1"), "general_mAP": eval_metrics.get("general_mAP"), "mINP": None, "mINP_status": "not exposed by repository R1_mAP_eval"}
                    with (run_dir / "metrics_per_epoch.csv").open("a", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader() if stream.tell() == 0 else None; writer.writerow(row)
                    write_json(run_dir / "run_progress.json", {"last_epoch": epoch, "effective_updates": effective_updates, "batch_index": batch_index, "saved_nodes": saved_nodes})
                # The next epoch must report its own activation and local
                # margin statistics, not a cumulative sum from epoch 1.
                stats = {"batches": 0, "active_batches": 0, "selected_relations": 0, "active_relations": 0, "current_error_pid": 0, "correct_pid": 0, "margin_sum": 0.0, "purity_sum": 0.0, "ndp_loss_sum": 0.0, "gpc_selected": 0}
                epoch_sums = defaultdict(float)
            del iterator
            gc.collect()
        if rank == 0:
            write_json(run_dir / "run_summary.json", {"status": "PASS", "arm": args.arm, "seed": args.seed, "raw_epoch_mode": raw_epoch_mode, "start_epoch": start_epoch, "end_epoch": epoch, "segment_epochs": segment_epoch, "start_global_step": start_global_step, "effective_updates": effective_updates, "batch_index": batch_index, "saved_nodes": saved_nodes, "elapsed_seconds": time.time() - started, "initialization": "warm-start B0 checkpoint" if args.resume_checkpoint else "PRCC common init + fresh optimizer", "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None, "optimizer_step_executed": True})
            if args.resume_model:
                run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
                run_summary["initialization"] = "model-only B0 warm start + fresh optimizer/scheduler/scaler/RNG"
                run_summary["resume_model"] = str(args.resume_model)
                write_json(run_dir / "run_summary.json", run_summary)
    except Exception as exc:
        if rank == 0:
            write_json(run_dir / "failure.json", {"status": "FAIL", "type": type(exc).__name__, "message": str(exc), "epoch": epoch, "effective_updates": effective_updates, "batch_index": batch_index})
        raise
    finally:
        if mf is not None: mf.close()
        if af is not None: af.close()
        prcc.phase6b.cleanup_dist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "verify_gpc_pair", "train"), required=True)
    parser.add_argument("--seed", type=int, choices=(1234, 4321), default=1234)
    parser.add_argument("--arm", choices=("B0", "N5", "NF0"), default="N5")
    parser.add_argument("--run-id", default="ndp_v5_prcc")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--feature", type=Path, default=DEFAULT_FEATURE)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--checkpoint", type=Path, default=None, help="model.pth or rank state used as train-only reference source")
    parser.add_argument("--source-update", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="node directory containing rank0_state.pt/rank1_state.pt")
    parser.add_argument("--resume-model", type=Path, default=None, help="model.pth-only warm start when a complete rank state is unavailable")
    parser.add_argument("--resume-epoch", type=int, default=3, help="absolute epoch for model-only warm start")
    parser.add_argument("--resume-global-step", type=int, default=1653, help="global step for model-only warm start")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--check-epoch", type=int, default=5)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-updates", default="20,551,1102,1653")
    parser.add_argument("--eval-epochs", default="3,5")
    parser.add_argument("--diagnostic-steps", type=int, default=20)
    parser.add_argument("--ndp-epsilon", type=float, default=0.02, help="NDP-V5 margin tolerance; default preserves historical experiments")
    args = parser.parse_args()
    if args.mode == "prepare":
        export_reference(args)
        return
    if args.mode == "verify_gpc_pair":
        if args.run_dir is None:
            raise SystemExit("--run-dir is required in verify_gpc_pair mode")
        verify_gpc_pair(args)
        return
    if args.run_dir is None:
        raise SystemExit("--run-dir is required in train mode")
    run_train(args)


if __name__ == "__main__":
    main()
