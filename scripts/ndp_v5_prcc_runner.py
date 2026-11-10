#!/usr/local/bin/python3.10
"""PRCC B0/N5/NF0/NRG/NCAP/NDYN/NDYN_TOP5/NGATE/NFRESH/NCOMP/NONLINE/NLIST paired experiment runner.

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
from loss.triplet_loss import euclidean_dist, hard_example_mining  # noqa: E402


WORLD_SIZE = 2
BATCH_SIZE = 16
NUM_INSTANCES = 4
EPOCH_STEPS = 551
MAX_SCHEDULE_EPOCHS = 60
EPSILON = 0.02
LAMBDA_NDP = 0.05
LAMBDA_GPC = 0.1
NDP_SAFE_MARGIN = 0.05
NCAP_MAX_TARGET = 0.02
NDYN_DELTA = 0.005
NDYN_TOP5_DELTA_CAP = 0.005
NDP_RANK_TEACHER_MIN_MARGIN = 0.02
NDP_RANK_TEACHER_MAX_MARGIN = 0.10
NDP_RANK_CURRENT_MAX_MARGIN = 0.05
NDP_RANK_TARGET_MIN_MARGIN = 0.01
NDP_RANK_TARGET_MAX_MARGIN = 0.05
NDP_SR_TEACHER_MIN_MARGIN = 0.02
NDP_SR_TEACHER_MAX_MARGIN = 0.10
NDP_SR_CONFUSION_MARGIN = 0.02
NDP_LATE_TEACHER_MIN_MARGIN = 0.02
NDP_LATE_TEACHER_MAX_MARGIN = 0.10
NDP_LATE_CONFUSION_MARGIN = 0.0
NDP_LATE_TARGET_MARGIN = 0.01
NDP_LATE_START_EPOCH = 16
NDP_GPC_CONFUSION_MARGIN = 0.0
NDP_GPC_TARGET_MARGIN = 0.01
NDP_GPC_IMPACT_TARGET_MARGIN = 0.01
NDP_GPC_DELTA_MAX_RISK = 4096
NDP_DELTA_FIX_MAX_RISK = 4096
NDP_DELTA_FIX_MIN_DROP = 1e-4
# One deterministic, exact pre/post decrease is sufficient evidence.  A
# later occurrence must still show a deficit against this fixed first target
# before any protection is applied, so this is not a static-margin gate.
NDP_DELTA_FIX_MIN_OBSERVATIONS = 1
NDP_DELTA_FIX_MAX_PROTECTIONS_PER_BATCH = 1
NDP_DELTA_FIX_MAX_PROTECTION_BATCHES = 16
# NDP-AD is a deliberately small, batch-local adaptation of ANFI's
# affinity/discrepancy idea.  The topology is used only for detached
# selection/weighting; the ranking term below stays connected to the current
# student features.  Keeping one relation per batch makes this a protection
# path, not a second dense Triplet miner.
NDP_AD_K = 5
NDP_AD_TAU = 0.1
NDP_AD_MAX_RELATIONS_PER_BATCH = 1
NDP_AD_MAX_PROTECTION_BATCHES = 16
# NDP-SHN: soft identity-center hard-negative mining with a multi-positive
# contrastive objective.  It deliberately uses only the current fixed batch.
NDP_SHN_TOPK = 3
NDP_SHN_TAU = 0.2
NDP_SHN_MAX_ANCHORS_PER_BATCH = 8
NDP_SHN_BANK_MAX_CROSS_POSITIVES = 8
NDP_SHN_RELIABLE_BANK_MAX_ANCHORS = 2
NDP_SHN_RELIABLE_BANK_AUX_WEIGHT = 0.25
NDP_SHN_RELIABLE_BANK_RELIABILITY_FLOOR = 0.55
NDP_SHN_RELIABLE_BANK_MIN_BOUNDARY_CANDIDATES = 2
# Select a moderately risky reliable relation rather than the easiest or the
# single hardest bank endpoint.  This is a fixed data-relative quantile, not a
# dataset/test-set tuned margin threshold.
NDP_SHN_RELIABLE_BANK_RISK_QUANTILE = 0.40
# NDP_SHN_MP_CAPR: keep the original SHN-MP eight-anchor path untouched and
# add at most two train-batch anchors from PIDs not covered by that path.
# CAPR-A/B/C are selector/objective ablations, not changes to GPC or Bio.
NDP_SHN_CAPR_MAX_EXTRA_ANCHORS = 2
NDP_SHN_CAPR_HARD_BETA = 0.3
NDP_SHN_CAPR_GAMMA = 0.5
NDP_SHN_CAPR_RISK_WEIGHT_MIN = 0.75
NDP_SHN_CAPR_RISK_WEIGHT_MAX = 1.25
# NDP_SHN_MP_ABM: Anchor-aware Boundary Mining.  These are selector-only
# constants.  The live objective remains exactly the balanced SHN-MP
# softplus, with the historical Top-3/tau/max-anchor/lambda settings.
NDP_SHN_ABM_OUTLIER_MAD = 1.5
NDP_SHN_ABM_TARGET_MAD = 0.5
NDP_SHN_ABM_UPPER_MAD = 0.25
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


def extract_train_feature_bank(model: torch.nn.Module, cfg: Any, dataset: Any, device: torch.device) -> torch.Tensor:
    """Encode the complete PRCC train split with deterministic eval preprocessing.

    This bank is an epoch-level endpoint snapshot. It is intentionally
    train-only, uses no sampler or random augmentation, and is detached before
    being returned so NDP gradients can only flow through the live anchor.
    """
    # This pass is auxiliary. It must be observational with respect to the
    # actual training trajectory: the released sampler, augmentation RNG and
    # frozen GPC selector must see exactly the same RNG state afterwards.
    saved_rng = base.rng_state(device)
    _, transform_test = prcc.build_img_transforms(cfg)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(0x4E445046 + len(dataset.train))
    loader = DataLoader(
        ImageDataset(dataset.train, transform=transform_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        generator=loader_generator,
        num_workers=0,
        pin_memory=False,
    )
    was_training = bool(model.training)
    model.eval()
    chunks: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=False)
                camids = batch["camid"].to(device, non_blocking=False)
                with amp.autocast(enabled=True):
                    output_value = model(images, cam_label=camids)
                if torch.is_tensor(output_value):
                    identity = output_value
                elif isinstance(output_value, (tuple, list)) and len(output_value) == 2:
                    _score, feat = output_value
                    identity = feat[0] if isinstance(feat, (tuple, list)) else feat
                else:
                    raise TypeError(f"unexpected fresh-bank model output: {type(output_value)!r}")
                identity = F.normalize(identity.float(), dim=-1).detach()
                if identity.ndim != 2 or identity.shape[1] != 1024:
                    raise RuntimeError(f"fresh feature shape mismatch: {tuple(identity.shape)}")
                chunks.append(identity)
        fresh = torch.cat(chunks, dim=0).detach()
        if fresh.shape != (len(dataset.train), 1024) or not bool(torch.isfinite(fresh).all().item()):
            raise RuntimeError(f"invalid fresh train feature bank: shape={tuple(fresh.shape)}")
        return fresh
    finally:
        # Wait for auxiliary PPU work before restoring the per-rank generator.
        # base.rng_state/restore_rng cover Python, NumPy, torch CPU and the
        # rank-local PPU/CUDA generator.
        torch.cuda.synchronize(device)
        base.restore_rng(saved_rng, device)
        if was_training:
            model.train()


def build_online_neighbor_graph(
    fresh_features: torch.Tensor,
    train_pids: torch.Tensor,
    device: torch.device,
    max_negative_pids: int = 3,
    block_size: int = 512,
    previous_signature: list[tuple[int, ...]] | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], torch.Tensor, dict[str, Any], list[tuple[int, ...]]]:
    """Build an epoch-local dangerous-neighbor graph entirely with GPU tensors.

    ``fresh_features`` is a detached, normalized train-only bank.  Similarities
    are evaluated in query blocks, so the implementation never materializes an
    N-by-N similarity matrix.  A scatter-reduce first finds the best sample for
    each PID, and top-k then selects distinct negative PIDs.  The returned
    combined bank contains ``[fresh_samples, positive_reference_per_anchor]``;
    all entries are detached and only the live batch anchor can receive an NDP
    gradient.
    """
    if fresh_features.ndim != 2 or train_pids.ndim != 1 or fresh_features.shape[0] != train_pids.shape[0]:
        raise ValueError("online graph feature/PID shape mismatch")
    if max_negative_pids < 1 or block_size < 1:
        raise ValueError("max_negative_pids and block_size must be positive")
    n_samples, feature_dim = fresh_features.shape
    if n_samples < 2:
        raise ValueError("online graph requires at least two train samples")
    features = fresh_features.detach().to(device=device, dtype=torch.float32).contiguous()
    pids = train_pids.detach().to(device=device, dtype=torch.long).contiguous()
    unique_pids, inverse, counts = torch.unique(pids, sorted=True, return_inverse=True, return_counts=True)
    pid_count = int(unique_pids.numel())
    if pid_count < 2:
        raise ValueError("online graph requires at least two distinct PIDs")

    # Mean same-PID reference with the anchor itself removed.
    pid_sums = torch.zeros((pid_count, feature_dim), dtype=torch.float32, device=device)
    pid_sums.index_add_(0, inverse, features)
    positive_counts = counts[inverse] - 1
    positive_valid = positive_counts > 0
    positive = (pid_sums[inverse] - features) / positive_counts.clamp_min(1).to(features.dtype).unsqueeze(1)
    positive = F.normalize(positive, dim=1)

    k = min(int(max_negative_pids), pid_count - 1)
    pid_columns = inverse.unsqueeze(0)
    sample_columns = torch.arange(n_samples, device=device, dtype=torch.long).unsqueeze(0)
    top_indices_chunks: list[torch.Tensor] = []
    top_pid_chunks: list[torch.Tensor] = []
    top_valid_chunks: list[torch.Tensor] = []
    inf = float("inf")
    for start in range(0, n_samples, int(block_size)):
        stop = min(start + int(block_size), n_samples)
        query = features[start:stop]
        query_pid_groups = inverse[start:stop]
        similarities = query @ features.t()
        similarities = similarities.masked_fill(inverse.unsqueeze(0) == query_pid_groups.unsqueeze(1), -inf)

        # Reduce sample similarities to one maximum per negative PID.
        max_by_pid = torch.full((stop - start, pid_count), -inf, dtype=similarities.dtype, device=device)
        max_by_pid.scatter_reduce_(1, pid_columns.expand(stop - start, -1), similarities, reduce="amax", include_self=True)
        max_by_pid[torch.arange(stop - start, device=device), query_pid_groups] = -inf

        # Find the image index attaining each PID maximum.  Ties resolve to
        # the lowest train index, making the graph deterministic.
        max_for_sample = max_by_pid.gather(1, pid_columns.expand(stop - start, -1))
        is_pid_max = similarities == max_for_sample
        candidate_indices = sample_columns.expand(stop - start, -1)
        candidate_indices = torch.where(is_pid_max, candidate_indices, torch.full_like(candidate_indices, n_samples))
        best_index_by_pid = torch.full((stop - start, pid_count), n_samples, dtype=torch.long, device=device)
        best_index_by_pid.scatter_reduce_(1, pid_columns.expand(stop - start, -1), candidate_indices, reduce="amin", include_self=True)

        top_values, top_pid = torch.topk(max_by_pid, k=k, dim=1)
        top_indices = best_index_by_pid.gather(1, top_pid)
        top_valid = torch.isfinite(top_values) & top_indices.lt(n_samples)
        top_indices_chunks.append(top_indices)
        top_pid_chunks.append(top_pid)
        top_valid_chunks.append(top_valid)

    top_indices_all = torch.cat(top_indices_chunks, dim=0)
    top_pid_all = torch.cat(top_pid_chunks, dim=0)
    top_valid_all = torch.cat(top_valid_chunks, dim=0)
    # This copy is only O(N*K), not a similarity matrix.
    top_indices_cpu = top_indices_all.detach().cpu().tolist()
    top_valid_cpu = top_valid_all.detach().cpu().tolist()
    train_pids_cpu = pids.detach().cpu().tolist()
    positive_valid_cpu = positive_valid.detach().cpu().tolist()

    combined_features = torch.cat((features, positive), dim=0).detach()
    rows: dict[int, list[dict[str, Any]]] = {}
    signatures: list[tuple[int, ...]] = []
    valid_anchor_count = 0
    relation_count = 0
    for anchor_id in range(n_samples):
        negative_indices = [
            int(index)
            for index, valid in zip(top_indices_cpu[anchor_id], top_valid_cpu[anchor_id])
            if valid
        ]
        negative_pids = [train_pids_cpu[index] for index in negative_indices]
        signature = tuple(negative_indices)
        signatures.append(signature)
        if not positive_valid_cpu[anchor_id] or not negative_indices:
            continue
        valid_anchor_count += 1
        relation_count += len(negative_indices)
        rows[anchor_id] = [{
            "relation_index": anchor_id,
            "anchor_image_index": anchor_id,
            "positive_local": [n_samples + anchor_id],
            "negative_local": negative_indices,
            "reference_margin": 0.0,
            "reference_pid_margin": [0.0] * len(negative_indices),
            "historical_valid_pid": [True] * len(negative_indices),
            "anchor_pid": int(train_pids_cpu[anchor_id]),
            "negative_pid": [int(value) for value in negative_pids],
            "dynamic_m0_pid_margin": [0.0] * len(negative_indices),
            "dynamic_m1_pid_margin": [0.0] * len(negative_indices),
            "dynamic_drop_pid": [0.0] * len(negative_indices),
        }]

    changed_anchors = None
    change_rate = None
    if previous_signature is not None:
        if len(previous_signature) != len(signatures):
            raise ValueError("online graph previous signature length mismatch")
        changed_anchors = int(sum(old != new for old, new in zip(previous_signature, signatures)))
        change_rate = float(changed_anchors / max(1, n_samples))
    graph_digest = hashlib.sha256(
        np.asarray([value for signature in signatures for value in (len(signature), *signature)], dtype=np.int64).tobytes()
    ).hexdigest()
    summary = {
        "online": True,
        "online_train_samples": int(n_samples),
        "online_pid_count": pid_count,
        "online_max_negative_pids": k,
        "online_block_size": int(block_size),
        "online_valid_anchor_count": valid_anchor_count,
        "online_relation_count": relation_count,
        "online_neighbor_graph_sha256": graph_digest,
        "online_changed_anchor_count": changed_anchors,
        "online_neighbor_change_rate": change_rate,
        "online_reference_positive_excludes_anchor": True,
        "online_similarity_device": str(features.device),
    }
    return rows, combined_features, summary, signatures


def rng_state_digest(state: dict[str, Any]) -> str:
    """Stable digest for the RNG state captured around an auxiliary pass."""
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode("utf-8"))
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode("utf-8"))
    digest.update(np.asarray(numpy_state[1], dtype=np.uint32).tobytes())
    digest.update(str(int(numpy_state[2])).encode("utf-8"))
    digest.update(str(float(numpy_state[3])).encode("utf-8"))
    digest.update(str(int(numpy_state[4])).encode("utf-8"))
    digest.update(state["torch_cpu"].detach().cpu().contiguous().numpy().tobytes())
    digest.update(state["torch_cuda"].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summarize_fresh_bank(reference_rows: dict[int, list[dict[str, Any]]], fresh_features: torch.Tensor) -> dict[str, Any]:
    """Compare the fixed t1 margins with the current epoch endpoint bank."""
    fresh = fresh_features.detach().float().cpu().numpy()
    deltas: list[float] = []
    fresh_margins: list[float] = []
    old_margins: list[float] = []
    selected_count = 0
    for rows in reference_rows.values():
        for relation in rows:
            anchor = fresh[int(relation["anchor_image_index"])]
            positive = fresh[np.asarray(relation["positive_local"], dtype=np.int64)]
            negative = fresh[np.asarray(relation["negative_local"], dtype=np.int64)]
            positive_similarity = float((positive @ anchor).mean())
            for pid_index, valid in enumerate(relation["historical_valid_pid"]):
                if not valid:
                    continue
                current = positive_similarity - float(negative[pid_index] @ anchor)
                old = float(relation["reference_pid_margin"][pid_index])
                old_margins.append(old)
                fresh_margins.append(current)
                deltas.append(current - old)
                selected_count += 1
    values = np.asarray(deltas, dtype=np.float64)
    fresh_values = np.asarray(fresh_margins, dtype=np.float64)
    old_values = np.asarray(old_margins, dtype=np.float64)
    return {
        "fresh_valid_pid_count": int(values.size),
        "fixed_selected_pid_count": int(selected_count),
        "old_margin_mean": float(old_values.mean()) if values.size else 0.0,
        "fresh_margin_mean": float(fresh_values.mean()) if values.size else 0.0,
        "margin_delta_mean": float(values.mean()) if values.size else 0.0,
        "margin_delta_std": float(values.std()) if values.size else 0.0,
        "margin_delta_min": float(values.min()) if values.size else 0.0,
        "margin_delta_max": float(values.max()) if values.size else 0.0,
    }


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
    dynamic_graph = bool(graph.get("dynamic", False))
    dynamic_top5_graph = bool(graph.get("dynamic_top5", False))
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
        if dynamic_top5_graph:
            dynamic_valid = row.get("dynamic_top5_valid_pid", [False] * len(margins))
        else:
            dynamic_valid = row.get("dynamic_valid_pid", [True] * len(margins)) if dynamic_graph else [True] * len(margins)
        if len(dynamic_valid) != len(margins):
            raise RuntimeError(f"dynamic PID mask length mismatch at relation {relation_index}")
        valid = [margin > 0.0 and bool(dynamic_valid[index]) for index, margin in enumerate(margins)]
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
            "dynamic_m0_pid_margin": [float(v) for v in row.get("dynamic_m0_pid_margin", margins)],
            "dynamic_m1_pid_margin": [float(v) for v in row.get("dynamic_m1_pid_margin", margins)],
            "dynamic_drop_pid": [float(v) for v in row.get("dynamic_drop_pid", [0.0] * len(margins))],
        })
    meta = {
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "graph_path": str(graph_path),
        "graph_sha256": sha256_file(graph_path),
        "reference_update": reference_update,
        "dynamic_graph": dynamic_graph,
        "dynamic_top5_graph": dynamic_top5_graph,
        "dynamic_delta": float(graph.get("dynamic_delta", NDYN_DELTA)) if dynamic_graph else None,
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


def finite_gradient_norm(gradient: torch.Tensor | None) -> tuple[float, bool]:
    """Return a finite diagnostic norm without turning unavailable probes into NaN.

    The probe is diagnostic only.  A missing/non-finite autograd result must not
    be confused with a zero NDP gradient, so the boolean is retained by the
    caller while the numeric field stays CSV-safe.
    """
    if gradient is None:
        return 0.0, False
    try:
        value = torch.linalg.vector_norm(gradient.detach().float())
        if not bool(torch.isfinite(value).item()):
            return 0.0, False
        return float(value.cpu()), True
    except (RuntimeError, ValueError, TypeError):
        return 0.0, False


def finite_gradient_cosine(left: torch.Tensor | None, right: torch.Tensor | None) -> tuple[float, bool]:
    """Return a finite cosine diagnostic for two feature gradients."""
    if left is None or right is None:
        return 0.0, False
    try:
        left_flat = left.detach().float().reshape(-1)
        right_flat = right.detach().float().reshape(-1)
        left_norm = torch.linalg.vector_norm(left_flat)
        right_norm = torch.linalg.vector_norm(right_flat)
        if not bool(torch.isfinite(left_norm).item()) or not bool(torch.isfinite(right_norm).item()):
            return 0.0, False
        if float(left_norm.item()) <= 1e-12 or float(right_norm.item()) <= 1e-12:
            return 0.0, False
        value = torch.dot(left_flat, right_flat) / (left_norm * right_norm)
        if not bool(torch.isfinite(value).item()):
            return 0.0, False
        return float(value.cpu()), True
    except (RuntimeError, ValueError, TypeError):
        return 0.0, False


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


def set_exact_gpc_pair_context(phase: str, seed: int) -> str:
    """Use the frozen formal namespace literally, without an NDP prefix."""
    if not phase or "_B0" in phase or "_N5" in phase or "_NDP" in phase:
        raise ValueError(f"invalid arm-dependent GPC phase: {phase!r}")
    gpc.p7.PHASE = str(phase)
    gpc.p7.SEED = int(seed)
    return str(phase)


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
    if arm == "NRG":
        return max(reference_margin - epsilon, 0.01)
    if arm == "NCAP":
        return max(0.0, min(reference_margin - epsilon, NCAP_MAX_TARGET))
    if arm == "NDYN":
        return max(reference_margin - NDYN_DELTA, 0.0)
    if arm == "NDYN_TOP5":
        raise ValueError("NDYN_TOP5 target requires dynamic_drop")
    raise ValueError(f"unsupported NDP arm: {arm}")


def ndyn_top5_target_margin(reference_margin: float, dynamic_drop: float) -> float:
    """Recover only a small, fixed fraction of the observed t0->t1 drop."""
    return reference_margin + min(NDYN_TOP5_DELTA_CAP, 0.5 * max(dynamic_drop, 0.0))


def batch_hard_negative_pid(raw_feature: torch.Tensor, pids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the original Batch-Hard negative PID for each local anchor.

    This is a detached, PID-level filter only.  It deliberately reuses the
    frozen DIFFER Triplet implementation (raw feat[0], Euclidean distance,
    ``hard_example_mining``) and never contributes a Triplet or NDP gradient.
    The second return value marks anchors for which a different-PID sample was
    available in the local Batch.
    """
    with torch.no_grad():
        detached_feature = raw_feature.detach().float()
        detached_pids = pids.detach()
        distances = euclidean_dist(detached_feature, detached_feature)
        _dist_ap, _dist_an, _p_inds, n_inds = hard_example_mining(
            distances, detached_pids, return_inds=True
        )
        valid = detached_pids.unsqueeze(0).ne(detached_pids.unsqueeze(1)).any(dim=1)
        hard_pid = torch.full_like(detached_pids, -1)
        hard_pid[valid] = detached_pids[n_inds[valid]]
        return hard_pid, valid


def ndp_relation_excluded_by_triplet_pid(negative_pid: int, hard_negative_pid: int) -> bool:
    """Whether NCOMP excludes this NDP relation by current hard PID."""
    return int(negative_pid) == int(hard_negative_pid) and int(hard_negative_pid) >= 0


def select_ndp_ad_relations(
    representation: torch.Tensor,
    pids: torch.Tensor,
    selected_gpc: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]],
    max_relations: int = NDP_AD_MAX_RELATIONS_PER_BATCH,
    k: int = NDP_AD_K,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]]:
    """Select sparse GPC-linked discrepancy relations in the live student space.

    This is the lightweight NDP-AD path.  The frozen GPC selector supplies the
    same-ID affinity relation.  A detached KNN adjacency then measures how
    much the positive pair and a candidate different-PID neighbor share local
    neighborhood structure.  The most affinity-close candidate with the
    largest neighborhood discrepancy is retained, while the original
    Batch-Hard negative PID is excluded so the term complements Triplet.

    The returned positive/negative similarities are *not* detached.  Only the
    candidate choice and adaptive topology weight are detached, which keeps a
    real gradient path to the current student representation and avoids any
    teacher-space or historical-margin target.
    """
    if not selected_gpc or representation.ndim != 2 or representation.shape[0] < 3:
        return []
    if max_relations <= 0:
        return []
    n = int(representation.shape[0])
    with torch.no_grad():
        detached = representation.detach().float()
        pair = detached @ detached.t()
        pair.fill_diagonal_(-float("inf"))
        neighbor_k = min(max(1, int(k)), n - 1)
        neighbor_values, neighbor_indices = pair.topk(neighbor_k, dim=1)
        # Softmax adjacency is a compact, scale-stable neighborhood profile.
        adjacency = torch.zeros_like(pair)
        adjacency.scatter_(1, neighbor_indices, F.softmax(neighbor_values / NDP_AD_TAU, dim=1))
        hard_negative_pids, hard_valid = batch_hard_negative_pid(detached, pids)

    candidates: list[tuple[float, int, int, int, float, float]] = []
    for left, right, _image_key, _state_key in selected_gpc:
        left = int(left)
        right = int(right)
        pid = int(pids[left].detach().cpu())
        if int(pids[right].detach().cpu()) != pid:
            continue
        excluded = {pid}
        if bool(hard_valid[left].item()):
            excluded.add(int(hard_negative_pids[left].item()))
        if bool(hard_valid[right].item()):
            excluded.add(int(hard_negative_pids[right].item()))
        candidate_indices = [
            int(index)
            for index in torch.nonzero(pids != pid, as_tuple=False).flatten().detach().cpu().tolist()
            if int(pids[index].detach().cpu()) not in excluded
        ]
        if not candidate_indices:
            continue
        positive_common = float((adjacency[left] * adjacency[right]).sum().item())
        for negative in candidate_indices:
            affinity = float(0.5 * (pair[negative, left] + pair[negative, right]).item())
            if not math.isfinite(affinity) or affinity <= 0.0:
                continue
            negative_common = float(
                0.5
                * (
                    (adjacency[left] * adjacency[negative]).sum()
                    + (adjacency[right] * adjacency[negative]).sum()
                ).item()
            )
            # Reliability is high when the GPC positive pair has a coherent
            # neighborhood, while discrepancy is high when the dangerous
            # negative does not share that neighborhood.  The 0.5 floor keeps
            # weak early neighborhoods from making the branch permanently
            # inactive; the sparse top-1 relation and low lambda control noise.
            reliability = 0.5 + 0.5 * max(0.0, min(1.0, positive_common))
            discrepancy = max(0.0, min(1.0, 1.0 - negative_common))
            score = max(0.0, affinity) * reliability * discrepancy
            if score > 0.0:
                candidates.append((score, left, right, negative, positive_common, negative_common))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    selected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]] = []
    used_pairs: set[tuple[int, int]] = set()
    for score, left, right, negative, positive_common, negative_common in candidates:
        pair_key = (left, right)
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        positive_similarity = torch.sum(representation[left] * representation[right])
        negative_similarity = 0.5 * (
            torch.sum(representation[negative] * representation[left])
            + torch.sum(representation[negative] * representation[right])
        )
        selected.append(
            (
                positive_similarity,
                negative_similarity,
                torch.as_tensor(score, dtype=representation.dtype, device=representation.device),
                int(pids[negative].detach().cpu()),
                positive_common,
                negative_common,
            )
        )
        if len(selected) >= int(max_relations):
            break
    return selected


def _select_ndp_shn_losses(
    representation: torch.Tensor,
    pids: torch.Tensor,
    topk: int = NDP_SHN_TOPK,
    tau: float = NDP_SHN_TAU,
    max_anchors: int = NDP_SHN_MAX_ANCHORS_PER_BATCH,
    positive_aggregation: str = "logsumexp",
    return_info: bool = False,
) -> list[tuple[torch.Tensor, float, int, int]] | tuple[list[tuple[torch.Tensor, float, int, int]], dict[str, Any]]:
    """Return sparse soft-hard-negative, multi-positive losses for a batch.

    Identity centers are formed from the *current student batch*.  Detached
    center affinities choose several close negative identities and give them a
    softmax weight, while the returned contrastive loss uses live centers and
    live positive features.  Thus the branch is different from Batch-Hard
    Triplet: it keeps multiple positives and a soft mixture of multiple
    negative identities rather than one negative image.
    """
    if positive_aggregation not in {"logsumexp", "balanced_mean"}:
        raise ValueError(f"unsupported SHN positive aggregation: {positive_aggregation}")
    empty_info: dict[str, Any] = {
        "candidate_anchor_count": 0,
        "candidate_pid_count": 0,
        "selected_anchor_count": 0,
        "selected_pid_count": 0,
        "selected_anchor_indices": [],
        "selected_anchor_pids": [],
        "topk_by_anchor": [],
    }
    if representation.ndim != 2 or representation.shape[0] < 4 or tau <= 0.0:
        return ([], empty_info) if return_info else []
    unique_pids, inverse = torch.unique(pids.detach(), sorted=True, return_inverse=True)
    num_ids = int(unique_pids.numel())
    if num_ids < 3 or max_anchors <= 0:
        return ([], empty_info) if return_info else []
    live_centers = []
    positive_indices: list[torch.Tensor] = []
    anchor_priority: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        detached = representation.detach().float()
        detached_centers = []
        for class_index in range(num_ids):
            members = torch.nonzero(inverse == class_index, as_tuple=False).flatten()
            detached_centers.append(F.normalize(detached[members].mean(dim=0), dim=0))
        detached_centers_tensor = torch.stack(detached_centers, dim=0)
        center_pair = detached_centers_tensor @ detached_centers_tensor.t()
        center_pair.fill_diagonal_(-float("inf"))
        negative_count = min(max(1, int(topk)), num_ids - 1)
        for class_index in range(num_ids):
            members = torch.nonzero(inverse == class_index, as_tuple=False).flatten()
            if members.numel() < 2:
                continue
            negative_scores, negative_classes = center_pair[class_index].topk(negative_count)
            # This detached priority identifies anchors for which the local
            # identity-center neighborhood is most ambiguous.  It does not
            # alter the original sampler or GPC selector.
            # The diagonal was masked above; the identity-center self-score
            # is one, so the closest negative affinity itself is the bounded
            # ambiguity priority.
            priority = float(negative_scores[0].item())
            for anchor in members.detach().cpu().tolist():
                anchor_priority.append((priority, int(anchor), negative_scores, negative_classes))
        anchor_priority.sort(key=lambda item: (-item[0], item[1]))
        chosen = anchor_priority[: int(max_anchors)]
    if not chosen:
        return ([], empty_info) if return_info else []
    selection_info = {
        "candidate_anchor_count": len(anchor_priority),
        "candidate_pid_count": num_ids,
        "selected_anchor_count": len(chosen),
        "selected_pid_count": len({int(inverse[int(item[1])].item()) for item in chosen}),
        "selected_anchor_indices": [int(item[1]) for item in chosen],
        "selected_anchor_pids": [int(pids[int(item[1])].detach().cpu()) for item in chosen],
        "topk_by_anchor": [
            {
                "anchor": int(item[1]),
                "pid": int(pids[int(item[1])].detach().cpu()),
                "negative_pids": [int(unique_pids[int(value)].detach().cpu()) for value in item[3]],
            }
            for item in chosen
        ],
    }
    for class_index in range(num_ids):
        members = torch.nonzero(inverse == class_index, as_tuple=False).flatten()
        live_centers.append(F.normalize(representation[members].mean(dim=0), dim=0))
        positive_indices.append(members)
    live_centers_tensor = torch.stack(live_centers, dim=0)
    losses: list[tuple[torch.Tensor, float, int, int]] = []
    for _priority, anchor, negative_scores, negative_classes in chosen:
        class_index = int(inverse[anchor].item())
        members = positive_indices[class_index]
        positives = members[members != int(anchor)]
        if positives.numel() == 0:
            continue
        # Recompute the live logits only for selected anchors.  The negative
        # mixture weights are detached topology/center evidence, not a second
        # gradient path or a historical target.
        anchor_feature = representation[int(anchor)]
        positive_logits = representation[positives] @ anchor_feature
        negative_logits = live_centers_tensor[negative_classes] @ anchor_feature
        weights = F.softmax(negative_scores.float() / float(tau), dim=0).detach()
        negative_logsum = torch.logsumexp(
            negative_logits / float(tau) + torch.log(weights.clamp_min(1e-6)), dim=0
        )
        if positive_aggregation == "balanced_mean":
            # The historical logsumexp gives most of its positive gradient to
            # whichever same-PID image is already the most similar.  Averaging
            # the per-positive softplus terms gives every legal same-PID
            # positive an equal topological contribution, while retaining the
            # same detached Top-3 negative mixture and temperature.  This is a
            # soft ranking objective: reliable positives with a safe margin
            # receive only a small residual gradient, so hard positives are
            # not forced to match a fixed target.
            loss = F.softplus(negative_logsum - positive_logits / float(tau)).mean()
        else:
            positive_logsum = torch.logsumexp(positive_logits / float(tau), dim=0)
            loss = F.softplus(negative_logsum - positive_logsum)
        margin = float((positive_logits.mean() - negative_logits.mean()).detach().float().cpu())
        losses.append((loss, margin, int(negative_classes.numel()), int(positives.numel())))
    selection_info["selected_anchor_count"] = len(losses)
    selection_info["selected_pid_count"] = len({int(inverse[int(item[1])].item()) for item in chosen[: len(losses)]})
    return (losses, selection_info) if return_info else losses


def select_ndp_shn_losses(
    representation: torch.Tensor,
    pids: torch.Tensor,
    topk: int = NDP_SHN_TOPK,
    tau: float = NDP_SHN_TAU,
    max_anchors: int = NDP_SHN_MAX_ANCHORS_PER_BATCH,
) -> list[tuple[torch.Tensor, float, int, int]]:
    """Historical SHN loss with logsumexp positive aggregation."""
    return _select_ndp_shn_losses(
        representation,
        pids,
        topk=topk,
        tau=tau,
        max_anchors=max_anchors,
        positive_aggregation="logsumexp",
    )


def select_ndp_shn_mp_losses(
    representation: torch.Tensor,
    pids: torch.Tensor,
    topk: int = NDP_SHN_TOPK,
    tau: float = NDP_SHN_TAU,
    max_anchors: int = NDP_SHN_MAX_ANCHORS_PER_BATCH,
    return_info: bool = False,
) -> list[tuple[torch.Tensor, float, int, int]] | tuple[list[tuple[torch.Tensor, float, int, int]], dict[str, Any]]:
    """Balanced multi-positive SHN loss used by the NDP_SHN_MP arm."""
    return _select_ndp_shn_losses(
        representation,
        pids,
        topk=topk,
        tau=tau,
        max_anchors=max_anchors,
        positive_aggregation="balanced_mean",
        return_info=return_info,
    )


def select_ndp_shn_mp_capr_losses(
    representation: torch.Tensor,
    pids: torch.Tensor,
    clothes: torch.Tensor,
    variant: str = "C",
    topk: int = NDP_SHN_TOPK,
    tau: float = NDP_SHN_TAU,
    max_extra_anchors: int = NDP_SHN_CAPR_MAX_EXTRA_ANCHORS,
    hard_beta: float = NDP_SHN_CAPR_HARD_BETA,
    original_info: dict[str, Any] | None = None,
    return_info: bool = False,
) -> list[tuple[torch.Tensor, float, int, int]] | tuple[list[tuple[torch.Tensor, float, int, int]], dict[str, Any]]:
    """Return the extra CAPR loss while leaving the original SHN-MP intact.

    CAPR-A adds at most two anchors from PIDs not covered by the unchanged
    eight-anchor SHN-MP selector and uses that selector's live balanced loss.
    CAPR-B keeps the same extra anchors but ranks every legal cross-clothes
    positive separately.  CAPR-C adds a bounded image-level hard-negative
    mixture to B; the original center-based SHN-MP negative path is never
    replaced.  All topology/risk weights are detached, while current-batch
    positive and negative image features stay live for backward.
    """
    if variant not in {"A", "B", "C"}:
        raise ValueError(f"unsupported CAPR variant: {variant}")
    empty_info: dict[str, Any] = {
        "variant": variant,
        "original_anchor_count": 0,
        "original_pid_count": 0,
        "original_anchor_indices": [],
        "original_anchor_pids": [],
        "candidate_extra_anchor_count": 0,
        "candidate_uncovered_pid_count": 0,
        "selected_extra_anchor_count": 0,
        "selected_extra_pid_count": 0,
        "selected_extra_anchor_indices": [],
        "selected_extra_anchor_pids": [],
        "extra_topk_by_anchor": [],
        "extra_cross_clothes_positive_count": 0,
        "extra_total_positive_count": 0,
        "extra_risk_weight_mean": 0.0,
        "extra_triplet_hard_overlap_count": 0,
        "extra_triplet_hard_count": 0,
    }
    if representation.ndim != 2 or representation.shape[0] < 4 or tau <= 0.0:
        return ([], empty_info) if return_info else []
    if pids.ndim != 1 or clothes.ndim != 1 or pids.shape[0] != representation.shape[0] or clothes.shape[0] != representation.shape[0]:
        return ([], empty_info) if return_info else []
    if max_extra_anchors <= 0 or not 0.0 <= float(hard_beta) <= 1.0:
        return ([], empty_info) if return_info else []

    if original_info is None:
        _original_losses, original_info = _select_ndp_shn_losses(
            representation,
            pids,
            topk=topk,
            tau=tau,
            max_anchors=NDP_SHN_MAX_ANCHORS_PER_BATCH,
            positive_aggregation="balanced_mean",
            return_info=True,
        )
    original_pids = {int(value) for value in original_info.get("selected_anchor_pids", [])}
    info = dict(empty_info)
    info.update({
        "original_anchor_count": int(original_info.get("selected_anchor_count", 0)),
        "original_pid_count": int(original_info.get("selected_pid_count", 0)),
        "original_anchor_indices": [int(value) for value in original_info.get("selected_anchor_indices", [])],
        "original_anchor_pids": [int(value) for value in original_info.get("selected_anchor_pids", [])],
    })
    unique_pids, inverse = torch.unique(pids.detach(), sorted=True, return_inverse=True)
    num_ids = int(unique_pids.numel())
    if num_ids < 3:
        return ([], info) if return_info else []

    with torch.no_grad():
        detached = representation.detach().float()
        detached_centers: list[torch.Tensor] = []
        members_by_pid: list[torch.Tensor] = []
        for class_index in range(num_ids):
            members = torch.nonzero(inverse == class_index, as_tuple=False).flatten()
            members_by_pid.append(members)
            detached_centers.append(F.normalize(detached[members].mean(dim=0), dim=0))
        detached_centers_tensor = torch.stack(detached_centers, dim=0)
        negative_count = min(max(1, int(topk)), num_ids - 1)
        candidate_rows: list[dict[str, Any]] = []
        uncovered_pid_indices: set[int] = set()
        for class_index, members in enumerate(members_by_pid):
            pid = int(unique_pids[class_index].item())
            if pid in original_pids or members.numel() < 2:
                continue
            for anchor_tensor in members:
                anchor = int(anchor_tensor.item())
                anchor_clothes = clothes[anchor].detach()
                cross_positive = members[
                    (members != anchor) & (clothes[members].detach() != anchor_clothes)
                ]
                if cross_positive.numel() == 0:
                    continue
                uncovered_pid_indices.add(class_index)
                positive_mean = float((detached[cross_positive] @ detached[anchor]).mean().item())
                negative_scores_all = detached_centers_tensor @ detached[anchor]
                negative_scores_all[class_index] = -float("inf")
                negative_scores, negative_classes = negative_scores_all.topk(negative_count)
                hard_negative_scores: list[torch.Tensor] = []
                for negative_class in negative_classes:
                    negative_members = members_by_pid[int(negative_class.item())]
                    negative_logits = detached[negative_members] @ detached[anchor]
                    hard_negative_scores.append(
                        float(tau) * (
                            torch.logsumexp(negative_logits / float(tau), dim=0)
                            - math.log(max(1, int(negative_logits.numel())))
                        )
                    )
                hard_scores_tensor = torch.stack(hard_negative_scores)
                mixed_scores = (1.0 - float(hard_beta)) * negative_scores + float(hard_beta) * hard_scores_tensor
                hardest = float(mixed_scores.max().item())
                local_margin = positive_mean - hardest
                risk = float(torch.sigmoid(torch.as_tensor((hardest - positive_mean) / float(tau))).item())
                candidate_rows.append({
                    "anchor": anchor,
                    "class_index": class_index,
                    "positive_indices": cross_positive,
                    "all_positive_indices": members[members != anchor],
                    "positive_mean": positive_mean,
                    "local_margin": local_margin,
                    "risk": risk,
                    "negative_scores": negative_scores,
                    "negative_classes": negative_classes,
                    "hard_negative_scores": hard_scores_tensor,
                })
        if not candidate_rows:
            info["candidate_uncovered_pid_count"] = len(uncovered_pid_indices)
            return ([], info) if return_info else []
        # This is a small, local risk ranking rather than ABM's global
        # median/MAD band.  At most one row per uncovered PID is retained, so
        # adding CAPR cannot let repeated views of one identity consume slots.
        best_by_pid: dict[int, dict[str, Any]] = {}
        for row in candidate_rows:
            class_index = int(row["class_index"])
            previous = best_by_pid.get(class_index)
            if previous is None or (float(row["risk"]), len(row["positive_indices"]), -int(row["anchor"])) > (
                float(previous["risk"]), len(previous["positive_indices"]), -int(previous["anchor"])
            ):
                best_by_pid[class_index] = row
        candidates = list(best_by_pid.values())
        candidates.sort(key=lambda row: (-float(row["risk"]), -len(row["positive_indices"]), int(row["anchor"])))
        chosen = candidates[: int(max_extra_anchors)]
        info["candidate_extra_anchor_count"] = len(candidate_rows)
        info["candidate_uncovered_pid_count"] = len(candidates)

    live_centers = [
        F.normalize(representation[members].mean(dim=0), dim=0)
        for members in members_by_pid
    ]
    live_centers_tensor = torch.stack(live_centers, dim=0)
    losses: list[tuple[torch.Tensor, float, int, int]] = []
    risk_weight_sum = 0.0
    risk_weight_count = 0
    cross_positive_count = 0
    total_positive_count = 0
    for row in chosen:
        anchor = int(row["anchor"])
        anchor_feature = representation[anchor]
        positive_indices = row["all_positive_indices"] if variant == "A" else row["positive_indices"]
        if positive_indices.numel() == 0:
            continue
        positive_logits = representation.index_select(0, positive_indices) @ anchor_feature
        negative_classes = row["negative_classes"]
        negative_scores = row["negative_scores"]
        center_logits = live_centers_tensor.index_select(0, negative_classes) @ anchor_feature
        hard_live_scores: list[torch.Tensor] = []
        for negative_class in negative_classes:
            negative_members = members_by_pid[int(negative_class.item())]
            negative_logits = representation.index_select(0, negative_members) @ anchor_feature
            hard_live_scores.append(
                float(tau) * (
                    torch.logsumexp(negative_logits / float(tau), dim=0)
                    - math.log(max(1, int(negative_logits.numel())))
                )
            )
        hard_live_tensor = torch.stack(hard_live_scores)
        if variant == "C":
            mixed_negative = (1.0 - float(hard_beta)) * center_logits + float(hard_beta) * hard_live_tensor
        else:
            mixed_negative = center_logits
        weights = F.softmax(negative_scores.float() / float(tau), dim=0).detach()
        negative_logsum = torch.logsumexp(
            mixed_negative / float(tau) + torch.log(weights.clamp_min(1e-6)), dim=0
        )
        if variant == "A":
            relation_loss = F.softplus(negative_logsum - positive_logits / float(tau)).mean()
            risk_weight_sum += float(positive_logits.numel())
            risk_weight_count += int(positive_logits.numel())
        else:
            detached_peak = float((
                (1.0 - float(hard_beta)) * negative_scores
                + float(hard_beta) * row["hard_negative_scores"]
                if variant == "C" else negative_scores
            ).max().item())
            risk_weights = torch.sigmoid((detached_peak - positive_logits.detach().float()) / float(tau))
            risk_weights = (
                float(NDP_SHN_CAPR_RISK_WEIGHT_MIN)
                + (float(NDP_SHN_CAPR_RISK_WEIGHT_MAX) - float(NDP_SHN_CAPR_RISK_WEIGHT_MIN)) * risk_weights
            ).detach()
            relation_loss = (F.softplus(negative_logsum - positive_logits / float(tau)) * risk_weights).mean()
            risk_weight_sum += float(risk_weights.sum().item())
            risk_weight_count += int(risk_weights.numel())
        margin = float((positive_logits.mean() - mixed_negative.mean()).detach().float().cpu())
        losses.append((relation_loss, margin, int(negative_classes.numel()), int(positive_logits.numel())))
        cross_positive_count += int(row["positive_indices"].numel())
        total_positive_count += int(positive_indices.numel())
    info.update({
        "selected_extra_anchor_count": len(losses),
        "selected_extra_pid_count": len({int(row["class_index"]) for row in chosen[: len(losses)]}),
        "selected_extra_anchor_indices": [int(row["anchor"]) for row in chosen[: len(losses)]],
        "selected_extra_anchor_pids": [int(unique_pids[int(row["class_index"])].item()) for row in chosen[: len(losses)]],
        "extra_topk_by_anchor": [
            {
                "anchor": int(row["anchor"]),
                "pid": int(unique_pids[int(row["class_index"])].item()),
                "negative_pids": [int(unique_pids[int(value)].item()) for value in row["negative_classes"]],
                "risk": float(row["risk"]),
                "local_margin": float(row["local_margin"]),
            }
            for row in chosen[: len(losses)]
        ],
        "extra_cross_clothes_positive_count": cross_positive_count,
        "extra_total_positive_count": total_positive_count,
        "extra_risk_weight_mean": risk_weight_sum / max(1, risk_weight_count),
    })
    return (losses, info) if return_info else losses


def _select_ndp_shn_mp_abm_losses(
    representation: torch.Tensor,
    pids: torch.Tensor,
    topk: int = NDP_SHN_TOPK,
    tau: float = NDP_SHN_TAU,
    max_anchors: int = NDP_SHN_MAX_ANCHORS_PER_BATCH,
) -> tuple[list[tuple[torch.Tensor, float, int, int]], dict[str, Any]]:
    """Select an Anchor-aware, non-outlier subset for the SHN-MP objective.

    The original SHN-MP selector ranks every image of a PID using the same
    PID-center neighborhood.  ABM keeps its loss untouched, but computes the
    Top-3 negative PID centers from each *anchor feature* instead.  A detached
    local margin (mean same-PID positive similarity minus the nearest negative
    center similarity) identifies a boundary band.  The band is centered a
    small robust-MAD step below the batch median, so the selector does not
    collapse onto the single hardest/outlier image.  At most one anchor is
    selected per PID; the final budget therefore maximizes identity coverage.

    ``info`` is deliberately small and detached.  It is used by unit/train
    diagnostics only and never changes the gradient path.
    """
    empty_info: dict[str, Any] = {
        "candidate_anchor_count": 0,
        "eligible_pid_count": 0,
        "selected_anchor_count": 0,
        "selected_pid_count": 0,
        "boundary_violation_count": 0,
        "topk_by_anchor": [],
        "selected_anchor_pids": [],
    }
    if representation.ndim != 2 or representation.shape[0] < 4 or tau <= 0.0 or max_anchors <= 0:
        return [], empty_info
    unique_pids, inverse = torch.unique(pids.detach(), sorted=True, return_inverse=True)
    num_ids = int(unique_pids.numel())
    if num_ids < 3:
        return [], empty_info

    with torch.no_grad():
        detached = representation.detach().float()
        detached_centers: list[torch.Tensor] = []
        members_by_pid: list[torch.Tensor] = []
        for class_index in range(num_ids):
            members = torch.nonzero(inverse == class_index, as_tuple=False).flatten()
            members_by_pid.append(members)
            detached_centers.append(F.normalize(detached[members].mean(dim=0), dim=0))
        detached_centers_tensor = torch.stack(detached_centers, dim=0)

        # Store only detached scalar selection evidence and the anchor-specific
        # Top-3 center indices.  Every anchor gets its own negative ordering.
        candidate_rows: list[dict[str, Any]] = []
        for class_index, members in enumerate(members_by_pid):
            if members.numel() < 2:
                continue
            for anchor_tensor in members:
                anchor = int(anchor_tensor.item())
                positive_members = members[members != anchor]
                positive_logits = detached[positive_members] @ detached[anchor]
                positive_mean = float(positive_logits.mean().item())
                negative_scores_all = detached_centers_tensor @ detached[anchor]
                negative_scores_all[class_index] = -float("inf")
                negative_count = min(max(1, int(topk)), num_ids - 1)
                negative_scores, negative_classes = negative_scores_all.topk(negative_count)
                nearest_negative = float(negative_scores[0].item())
                local_margin = positive_mean - nearest_negative
                candidate_rows.append(
                    {
                        "anchor": anchor,
                        "class_index": class_index,
                        "positive_mean": positive_mean,
                        "local_margin": local_margin,
                        "negative_scores": negative_scores,
                        "negative_classes": negative_classes,
                    }
                )
        if not candidate_rows:
            return [], empty_info

        margins = sorted(float(row["local_margin"]) for row in candidate_rows)
        median_margin = margins[len(margins) // 2]
        deviations = sorted(abs(value - median_margin) for value in margins)
        mad = deviations[len(deviations) // 2]
        margin_scale = max(float(mad), 1e-4)
        target_margin = median_margin - NDP_SHN_ABM_TARGET_MAD * margin_scale
        lower_margin = median_margin - NDP_SHN_ABM_OUTLIER_MAD * margin_scale
        upper_margin = median_margin + NDP_SHN_ABM_UPPER_MAD * margin_scale

        positive_scores = sorted(float(row["positive_mean"]) for row in candidate_rows)
        positive_median = positive_scores[len(positive_scores) // 2]
        positive_deviations = sorted(abs(value - positive_median) for value in positive_scores)
        positive_mad = max(positive_deviations[len(positive_deviations) // 2], 1e-4)
        positive_floor = positive_median - NDP_SHN_ABM_OUTLIER_MAD * positive_mad

        rows_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            rows_by_pid[int(row["class_index"])].append(row)

        # First choose one boundary-band representative per PID.  If a PID
        # has no in-band point, keep its closest non-positive-outlier point;
        # the fallback preserves coverage without silently selecting an
        # arbitrary hardest example.
        per_pid: list[dict[str, Any]] = []
        for class_index, rows in rows_by_pid.items():
            reliable = [row for row in rows if float(row["positive_mean"]) >= positive_floor]
            band = [
                row for row in reliable
                if lower_margin <= float(row["local_margin"]) <= upper_margin
            ]
            pool = band or reliable or rows
            selected = min(
                pool,
                key=lambda row: (
                    abs(float(row["local_margin"]) - target_margin),
                    float(row["local_margin"]),
                    int(row["anchor"]),
                ),
            )
            per_pid.append(selected)
        per_pid.sort(
            key=lambda row: (
                abs(float(row["local_margin"]) - target_margin),
                float(row["local_margin"]),
                int(row["anchor"]),
            )
        )
        chosen = per_pid[: int(max_anchors)]
        info = {
            "candidate_anchor_count": len(candidate_rows),
            "eligible_pid_count": len(per_pid),
            "selected_anchor_count": len(chosen),
            "selected_pid_count": len({int(row["class_index"]) for row in chosen}),
            "boundary_violation_count": sum(float(row["local_margin"]) <= 0.0 for row in chosen),
            "topk_by_anchor": [
                {
                    "anchor": int(row["anchor"]),
                    "pid": int(unique_pids[int(row["class_index"])].item()),
                    "negative_pids": [int(unique_pids[int(item)].item()) for item in row["negative_classes"]],
                    "local_margin": float(row["local_margin"]),
                }
                for row in chosen
            ],
            "selected_anchor_pids": [int(unique_pids[int(row["class_index"])].item()) for row in chosen],
            "median_margin": median_margin,
            "target_margin": target_margin,
            "lower_margin": lower_margin,
            "upper_margin": upper_margin,
        }

    live_centers = [
        F.normalize(representation[members].mean(dim=0), dim=0)
        for members in members_by_pid
    ]
    live_centers_tensor = torch.stack(live_centers, dim=0)
    losses: list[tuple[torch.Tensor, float, int, int]] = []
    for row in chosen:
        class_index = int(row["class_index"])
        anchor = int(row["anchor"])
        members = members_by_pid[class_index]
        positives = members[members != anchor]
        if positives.numel() == 0:
            continue
        anchor_feature = representation[anchor]
        positive_logits = representation[positives] @ anchor_feature
        negative_classes = row["negative_classes"]
        negative_scores = row["negative_scores"]
        negative_logits = live_centers_tensor[negative_classes] @ anchor_feature
        weights = F.softmax(negative_scores.float() / float(tau), dim=0).detach()
        negative_logsum = torch.logsumexp(
            negative_logits / float(tau) + torch.log(weights.clamp_min(1e-6)), dim=0
        )
        # This is byte-for-byte the SHN-MP positive aggregation rule: one
        # live softplus term per same-PID positive, then an equal mean.
        loss = F.softplus(negative_logsum - positive_logits / float(tau)).mean()
        margin = float((positive_logits.mean() - negative_logits.mean()).detach().float().cpu())
        losses.append((loss, margin, int(negative_classes.numel()), int(positives.numel())))
    info["selected_anchor_count"] = len(losses)
    info["selected_pid_count"] = len({int(item["class_index"]) for item in chosen[: len(losses)]})
    return losses, info


def select_ndp_shn_mp_abm_losses(
    representation: torch.Tensor,
    pids: torch.Tensor,
    topk: int = NDP_SHN_TOPK,
    tau: float = NDP_SHN_TAU,
    max_anchors: int = NDP_SHN_MAX_ANCHORS_PER_BATCH,
    return_info: bool = False,
) -> list[tuple[torch.Tensor, float, int, int]] | tuple[list[tuple[torch.Tensor, float, int, int]], dict[str, Any]]:
    """Anchor-aware Boundary Mining arm; original SHN-MP stays unchanged."""
    losses, info = _select_ndp_shn_mp_abm_losses(
        representation, pids, topk=topk, tau=tau, max_anchors=max_anchors
    )
    return (losses, info) if return_info else losses


def build_shn_bank_centers(
    bank_features: torch.Tensor,
    bank_pids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build detached train-only identity centers for NDP_SHN_BANK."""
    if bank_features.ndim != 2 or bank_pids.ndim != 1 or bank_features.shape[0] != bank_pids.shape[0]:
        raise ValueError("SHN bank feature/PID shape mismatch")
    if bank_features.shape[0] < 2:
        raise ValueError("SHN bank requires at least two train images")
    features = bank_features.detach().float()
    pids = bank_pids.detach().to(device=features.device, dtype=torch.long)
    unique_pids, inverse, counts = torch.unique(pids, sorted=True, return_inverse=True, return_counts=True)
    sums = torch.zeros(
        (int(unique_pids.numel()), int(features.shape[1])),
        dtype=features.dtype,
        device=features.device,
    )
    sums.index_add_(0, inverse, features)
    centers = F.normalize(sums / counts.to(features.dtype).unsqueeze(1).clamp_min(1.0), dim=1)
    return centers.detach(), unique_pids.detach()


def select_ndp_shn_bank_losses(
    representation: torch.Tensor,
    pids: torch.Tensor,
    bank_features: torch.Tensor,
    bank_pids: torch.Tensor,
    batch_bank_indices: torch.Tensor,
    bank_centers: torch.Tensor | None = None,
    bank_center_pids: torch.Tensor | None = None,
    bank_clothes: torch.Tensor | None = None,
    batch_clothes: torch.Tensor | None = None,
    topk: int = NDP_SHN_TOPK,
    tau: float = NDP_SHN_TAU,
    max_anchors: int = NDP_SHN_MAX_ANCHORS_PER_BATCH,
    max_cross_positives: int = NDP_SHN_BANK_MAX_CROSS_POSITIVES,
) -> list[tuple[torch.Tensor, float, int, int]]:
    """Return SHN_MP losses with a refreshed, train-only cross-batch bank.

    The bank is a detached student snapshot.  It supplies extra same-PID
    positives and PID-level negative centers, while current-batch features
    remain live so the objective still backpropagates through the student.
    Cross-batch positives prefer a different clothing label when that
    train-only metadata is available, then rank by bank-center reliability.
    No query/gallery feature is involved.
    """
    if representation.ndim != 2 or representation.shape[0] < 4 or tau <= 0.0:
        return []
    if bank_features.ndim != 2 or bank_features.shape[1] != representation.shape[1]:
        return []
    if bank_pids.ndim != 1 or bank_features.shape[0] != bank_pids.shape[0]:
        return []
    if batch_bank_indices.ndim != 1 or batch_bank_indices.shape[0] != representation.shape[0]:
        return []
    if max_anchors <= 0 or max_cross_positives < 0:
        return []
    bank_features = bank_features.detach().to(device=representation.device, dtype=torch.float32).contiguous()
    bank_pids = bank_pids.detach().to(device=representation.device, dtype=torch.long).contiguous()
    batch_bank_indices = batch_bank_indices.detach().to(device=representation.device, dtype=torch.long)
    if bank_clothes is not None:
        bank_clothes = bank_clothes.detach().to(device=representation.device, dtype=torch.long)
    if batch_clothes is not None:
        batch_clothes = batch_clothes.detach().to(device=representation.device, dtype=torch.long)
    if bank_centers is None or bank_center_pids is None:
        bank_centers, bank_center_pids = build_shn_bank_centers(bank_features, bank_pids)
    else:
        bank_centers = bank_centers.detach().to(device=representation.device, dtype=torch.float32)
        bank_center_pids = bank_center_pids.detach().to(device=representation.device, dtype=torch.long)
    if bank_centers.ndim != 2 or bank_center_pids.ndim != 1 or bank_centers.shape[0] != bank_center_pids.shape[0]:
        return []
    if bank_center_pids.numel() < 3:
        return []

    batch_index_set = set(int(value) for value in batch_bank_indices.detach().cpu().tolist())
    detached_representation = representation.detach().float()
    priorities: list[tuple[float, int, torch.Tensor, torch.Tensor, list[int], list[int]]] = []
    for anchor in range(int(representation.shape[0])):
        pid = int(pids[anchor].detach().cpu())
        center_matches = torch.nonzero(bank_center_pids == pid, as_tuple=False).flatten()
        if center_matches.numel() != 1:
            continue
        center_index = int(center_matches[0].item())
        negative_mask = bank_center_pids != pid
        negative_scores_all = bank_centers @ detached_representation[anchor]
        candidate_scores = negative_scores_all.masked_fill(~negative_mask, -float("inf"))
        negative_count = min(max(1, int(topk)), int(candidate_scores.numel() - 1))
        negative_scores, negative_classes = candidate_scores.topk(negative_count)
        if not bool(torch.isfinite(negative_scores).all().item()):
            continue

        local_positive_indices = [
            int(index)
            for index in torch.nonzero(pids == pid, as_tuple=False).flatten().detach().cpu().tolist()
            if int(index) != anchor
        ]
        bank_positive_indices = [
            int(index)
            for index in torch.nonzero(bank_pids == pid, as_tuple=False).flatten().detach().cpu().tolist()
            if int(index) not in batch_index_set
        ]
        if bank_positive_indices and bank_clothes is not None and batch_clothes is not None:
            anchor_clothes = int(batch_clothes[anchor].item())
            cross_clothes = [
                index for index in bank_positive_indices
                if int(bank_clothes[index].item()) != anchor_clothes
            ]
            if cross_clothes:
                bank_positive_indices = cross_clothes
        if max_cross_positives == 0:
            bank_positive_indices = []
        elif len(bank_positive_indices) > max_cross_positives:
            # Select centered/reliable bank positives, but retain deterministic
            # ordering and never let this become a nearest-positive miner.
            pid_center = bank_centers[center_index]
            bank_positive_indices.sort(
                key=lambda index: float(torch.dot(bank_features[index], pid_center).detach().cpu())
            , reverse=True)
            bank_positive_indices = bank_positive_indices[:max_cross_positives]
        if not local_positive_indices and not bank_positive_indices:
            continue
        priorities.append(
            (
                float(negative_scores[0].detach().cpu()),
                anchor,
                negative_scores,
                negative_classes,
                local_positive_indices,
                bank_positive_indices,
            )
        )
    priorities.sort(key=lambda item: (-item[0], item[1]))

    losses: list[tuple[torch.Tensor, float, int, int]] = []
    for _priority, anchor, negative_scores, negative_classes, local_positive_indices, bank_positive_indices in priorities[: int(max_anchors)]:
        anchor_feature = representation[anchor]
        positive_logits: list[torch.Tensor] = []
        if local_positive_indices:
            local_positive = torch.as_tensor(local_positive_indices, dtype=torch.long, device=representation.device)
            positive_logits.append(representation.index_select(0, local_positive) @ anchor_feature)
        if bank_positive_indices:
            bank_positive = torch.as_tensor(bank_positive_indices, dtype=torch.long, device=representation.device)
            positive_logits.append(bank_features.index_select(0, bank_positive) @ anchor_feature)
        if not positive_logits:
            continue
        positive_values = torch.cat(positive_logits, dim=0)
        negative_values = bank_centers.index_select(0, negative_classes) @ anchor_feature
        weights = F.softmax(negative_scores.float() / float(tau), dim=0).detach()
        negative_logsum = torch.logsumexp(
            negative_values / float(tau) + torch.log(weights.clamp_min(1e-6)), dim=0
        )
        loss = F.softplus(negative_logsum - positive_values / float(tau)).mean()
        margin = float((positive_values.mean() - negative_values.mean()).detach().float().cpu())
        losses.append((loss, margin, int(negative_classes.numel()), int(positive_values.numel())))
    return losses


def select_ndp_shn_reliable_bank_aux_losses(
    representation: torch.Tensor,
    pids: torch.Tensor,
    bank_features: torch.Tensor,
    bank_pids: torch.Tensor,
    batch_bank_indices: torch.Tensor,
    bank_centers: torch.Tensor | None = None,
    bank_center_pids: torch.Tensor | None = None,
    bank_clothes: torch.Tensor | None = None,
    batch_clothes: torch.Tensor | None = None,
    topk: int = NDP_SHN_TOPK,
    tau: float = NDP_SHN_TAU,
    max_anchors: int = NDP_SHN_RELIABLE_BANK_MAX_ANCHORS,
    risk_aware: bool = False,
    anchor_boundary: bool = False,
) -> list[tuple[torch.Tensor, float, int, int, float]]:
    """Return a sparse, low-weight bank-positive auxiliary loss.

    The main NDP term remains the live ``NDP_SHN_MP`` objective.  This helper
    only selects at most two reliable cross-batch positives.  Bank features
    are used for positive reliability and as detached positive endpoints; all
    negative identities and their soft mixture are computed from the current
    live batch, so a stale bank cannot replace the current student ordering.

    ``risk_aware=True`` keeps the same reliability gate but ranks candidates
    by a moderate local-confusion band.  The bank positive is compared with
    the current live negative identity centers; candidates close to the
    lower-middle risk quantile are preferred, while the single hardest
    relation is not forced.  Selection is greedy with distinct anchors and
    PIDs so the two auxiliary slots cover different identities whenever the
    batch provides them.

    ``anchor_boundary=True`` uses an Anchor-view margin and selects the
    second-lowest reliable bank margin within each Anchor.  Requiring at
    least two candidates skips an Anchor whose only candidate is the hardest
    positive, so the auxiliary path does not force a single extreme relation.
    """
    if representation.ndim != 2 or representation.shape[0] < 4 or tau <= 0.0:
        return []
    if bank_features.ndim != 2 or bank_features.shape[1] != representation.shape[1]:
        return []
    if bank_pids.ndim != 1 or bank_features.shape[0] != bank_pids.shape[0]:
        return []
    if batch_bank_indices.ndim != 1 or batch_bank_indices.shape[0] != representation.shape[0]:
        return []
    if max_anchors <= 0:
        return []

    live = F.normalize(representation.detach().float(), dim=1)
    bank_features = F.normalize(
        bank_features.detach().to(device=representation.device, dtype=torch.float32), dim=1
    ).contiguous()
    bank_pids = bank_pids.detach().to(device=representation.device, dtype=torch.long).contiguous()
    batch_bank_indices = batch_bank_indices.detach().to(device=representation.device, dtype=torch.long)
    if bank_clothes is not None:
        bank_clothes = bank_clothes.detach().to(device=representation.device, dtype=torch.long)
    if batch_clothes is not None:
        batch_clothes = batch_clothes.detach().to(device=representation.device, dtype=torch.long)
    if bank_centers is None or bank_center_pids is None:
        bank_centers, bank_center_pids = build_shn_bank_centers(bank_features, bank_pids)
    else:
        bank_centers = F.normalize(
            bank_centers.detach().to(device=representation.device, dtype=torch.float32), dim=1
        )
        bank_center_pids = bank_center_pids.detach().to(device=representation.device, dtype=torch.long)
    if bank_centers.ndim != 2 or bank_center_pids.ndim != 1 or bank_centers.shape[0] != bank_center_pids.shape[0]:
        return []

    unique_pids, inverse = torch.unique(pids.detach(), sorted=True, return_inverse=True)
    num_ids = int(unique_pids.numel())
    if num_ids < 3:
        return []
    live_centers: list[torch.Tensor] = []
    live_members: list[torch.Tensor] = []
    with torch.no_grad():
        for class_index in range(num_ids):
            members = torch.nonzero(inverse == class_index, as_tuple=False).flatten()
            live_members.append(members)
            live_centers.append(F.normalize(live[members].mean(dim=0), dim=0))
        detached_live_centers_tensor = torch.stack(live_centers, dim=0)
        center_pair = detached_live_centers_tensor @ detached_live_centers_tensor.t()
        center_pair.fill_diagonal_(-float("inf"))
        negative_count = min(max(1, int(topk)), num_ids - 1)
        batch_index_set = set(int(value) for value in batch_bank_indices.cpu().tolist())
        candidates: list[tuple[float, float, int, int, int, torch.Tensor, torch.Tensor]] = []
        for anchor in range(int(representation.shape[0])):
            class_index = int(inverse[anchor].item())
            local_positive_indices = [
                int(index) for index in live_members[class_index].cpu().tolist() if int(index) != anchor
            ]
            if not local_positive_indices:
                continue
            negative_scores, negative_classes = center_pair[class_index].topk(negative_count)
            if not bool(torch.isfinite(negative_scores).all().item()):
                continue
            # The nearest negative identity is Anchor-specific and is shared
            # by every bank positive candidate for this Anchor.  Compute it
            # once here rather than repeating the same center projection for
            # every cross-batch image.
            anchor_negative_scores = detached_live_centers_tensor @ live[anchor]
            anchor_negative_scores[class_index] = -float("inf")
            anchor_negative_peak = float(anchor_negative_scores.max().item())
            pid = int(unique_pids[class_index].item())
            bank_pid_indices = [
                int(index)
                for index in torch.nonzero(bank_pids == pid, as_tuple=False).flatten().cpu().tolist()
                if int(index) not in batch_index_set
            ]
            if bank_pid_indices and bank_clothes is not None and batch_clothes is not None:
                anchor_clothes = int(batch_clothes[anchor].item())
                cross_clothes = [
                    index for index in bank_pid_indices
                    if int(bank_clothes[index].item()) != anchor_clothes
                ]
                if cross_clothes:
                    bank_pid_indices = cross_clothes
            bank_center_matches = torch.nonzero(bank_center_pids == pid, as_tuple=False).flatten()
            if bank_center_matches.numel() != 1:
                continue
            bank_center = bank_centers[int(bank_center_matches[0].item())]
            local_positive_mean = float((live[local_positive_indices] @ live[anchor].detach()).mean().item())
            for bank_index in bank_pid_indices:
                bank_positive = bank_features[bank_index]
                live_alignment = float(torch.dot(live[anchor], bank_positive).item())
                center_support = float(torch.dot(bank_positive, bank_center).item())
                # A bank endpoint is accepted only when it is supported by
                # its own identity center and is not clearly worse than the
                # live in-batch positives.  This is a detached reliability
                # gate, not a historical margin target.
                if center_support < NDP_SHN_RELIABLE_BANK_RELIABILITY_FLOOR:
                    continue
                if live_alignment < local_positive_mean - 0.05:
                    continue
                reliability = 0.5 * (live_alignment + center_support)
                local_risk_margin = live_alignment - anchor_negative_peak
                candidates.append(
                    (reliability, local_risk_margin, anchor, pid, bank_index, negative_scores, negative_classes)
                )
        if anchor_boundary:
            candidates_by_anchor: dict[int, list[tuple[float, float, int, int, int, torch.Tensor, torch.Tensor]]] = {}
            for candidate in candidates:
                candidates_by_anchor.setdefault(candidate[2], []).append(candidate)
            boundary_candidates = []
            for anchor_candidates in candidates_by_anchor.values():
                if len(anchor_candidates) < NDP_SHN_RELIABLE_BANK_MIN_BOUNDARY_CANDIDATES:
                    continue
                anchor_candidates.sort(key=lambda item: (item[1], -item[0], item[3], item[4]))
                boundary_candidates.append(anchor_candidates[1])
            ranked = sorted(
                boundary_candidates,
                key=lambda item: (item[1], -item[0], item[3], item[2], item[4]),
            )
            chosen = []
            used_anchors: set[int] = set()
            used_pids: set[int] = set()
            for candidate in ranked:
                if candidate[2] in used_anchors or candidate[3] in used_pids:
                    continue
                chosen.append(candidate)
                used_anchors.add(candidate[2])
                used_pids.add(candidate[3])
                if len(chosen) >= int(max_anchors):
                    break
        elif not risk_aware:
            candidates.sort(key=lambda item: (-item[0], item[2], item[4]))
            chosen = candidates[: int(max_anchors)]
        elif candidates:
            risk_values = torch.tensor([item[1] for item in candidates], dtype=torch.float32)
            target_risk_margin = float(torch.quantile(risk_values, NDP_SHN_RELIABLE_BANK_RISK_QUANTILE).item())
            ranked = sorted(
                candidates,
                key=lambda item: (abs(item[1] - target_risk_margin), -item[0], item[3], item[2], item[4]),
            )
            chosen = []
            used_anchors: set[int] = set()
            used_pids: set[int] = set()
            for candidate in ranked:
                if candidate[2] in used_anchors or candidate[3] in used_pids:
                    continue
                chosen.append(candidate)
                used_anchors.add(candidate[2])
                used_pids.add(candidate[3])
                if len(chosen) >= int(max_anchors):
                    break
        else:
            chosen = []

    # Rebuild the selected negative centers from the live representation so
    # the auxiliary term contributes gradients to current negative identities;
    # only topology and reliability decisions above are detached.
    live_centers_tensor = torch.stack(
        [F.normalize(representation[members].mean(dim=0), dim=0) for members in live_members], dim=0
    )
    losses: list[tuple[torch.Tensor, float, int, int, float]] = []
    for reliability, _local_risk_margin, anchor, _pid, bank_index, negative_scores, negative_classes in chosen:
        anchor_feature = representation[anchor]
        negative_values = live_centers_tensor.index_select(0, negative_classes) @ anchor_feature
        weights = F.softmax(negative_scores.float() / float(tau), dim=0).detach()
        negative_logsum = torch.logsumexp(
            negative_values / float(tau) + torch.log(weights.clamp_min(1e-6)), dim=0
        )
        bank_positive_value = bank_features[bank_index] @ anchor_feature
        loss = F.softplus(negative_logsum - bank_positive_value / float(tau))
        margin = float((bank_positive_value - negative_values.mean()).detach().float().cpu())
        losses.append((loss, margin, int(negative_classes.numel()), 1, float(reliability)))
    return losses


def select_ndp_gpc_confusions(
    representation: torch.Tensor,
    pids: torch.Tensor,
    selected_gpc: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]],
) -> list[tuple[torch.Tensor, int]]:
    """Select current batch risks attached to the frozen GPC positive pairs.

    GPC supplies the positive relation ``(left, right)``.  For each such
    relation, choose the closest different-PID image to the pair in the
    current student space, excluding the original Batch-Hard negative PID for
    either endpoint so this branch complements rather than duplicates Triplet.
    The returned margin keeps its live autograd path through all three student
    features; only the candidate index and the confusion gate are detached.
    """
    if not selected_gpc:
        return []
    hard_negative_pids, _valid = batch_hard_negative_pid(representation, pids)
    selected: list[tuple[torch.Tensor, int]] = []
    for left, right, _image_key, _state_key in selected_gpc:
        left_pid = int(pids[int(left)].detach().cpu())
        if int(pids[int(right)].detach().cpu()) != left_pid:
            continue
        candidate_mask = pids != left_pid
        excluded = {
            int(hard_negative_pids[int(left)].detach().cpu()),
            int(hard_negative_pids[int(right)].detach().cpu()),
        }
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        keep = [
            int(index)
            for index in candidate_indices.detach().cpu().tolist()
            if int(pids[index].detach().cpu()) not in excluded
        ]
        if not keep:
            continue
        candidate_indices = torch.as_tensor(keep, dtype=torch.long, device=representation.device)
        with torch.no_grad():
            pair_scores = 0.5 * (
                representation[candidate_indices] @ representation[int(left)]
                + representation[candidate_indices] @ representation[int(right)]
            )
            negative_index = int(candidate_indices[int(torch.argmax(pair_scores).item())].item())
            negative_pid = int(pids[negative_index].detach().cpu())
        positive_similarity = torch.sum(representation[int(left)] * representation[int(right)])
        negative_similarity = 0.5 * (
            torch.sum(representation[negative_index] * representation[int(left)])
            + torch.sum(representation[negative_index] * representation[int(right)])
        )
        current_margin = positive_similarity - negative_similarity
        if float(current_margin.detach().float().cpu()) <= NDP_GPC_CONFUSION_MARGIN:
            selected.append((current_margin, negative_pid))
    return selected


def select_ndp_gpc_impacts(
    representation: torch.Tensor,
    pids: torch.Tensor,
    selected_gpc: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]],
    gpc_loss: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Keep only relations whose first-order margin is harmed by GPC.

    The candidate is the closest different-PID image to each actual GPC
    positive pair in the current student space.  Unlike NDP_GPC, the original
    Batch-Hard PID is not discarded here: it is retained only when the live
    GPC descent direction projects negatively onto this local margin.  This
    makes the branch a GPC-impact diagnostic/protection path rather than a
    second static hard-negative miner.
    """
    if not selected_gpc or not gpc_loss.requires_grad or not representation.requires_grad:
        return []
    try:
        gpc_grad = torch.autograd.grad(gpc_loss, representation, retain_graph=True, allow_unused=True)[0]
    except (RuntimeError, ValueError, TypeError):
        return []
    if gpc_grad is None:
        return []
    gpc_grad = gpc_grad.detach().float()
    selected: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for left, right, _image_key, _state_key in selected_gpc:
        left = int(left); right = int(right)
        pid = int(pids[left].detach().cpu())
        if int(pids[right].detach().cpu()) != pid:
            continue
        candidate_indices = torch.nonzero(pids != pid, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        with torch.no_grad():
            pair_scores = 0.5 * (
                representation[candidate_indices] @ representation[left]
                + representation[candidate_indices] @ representation[right]
            )
            negative_index = int(candidate_indices[int(torch.argmax(pair_scores).item())].item())
            negative_pid = int(pids[negative_index].detach().cpu())
        positive_similarity = torch.sum(representation[left] * representation[right])
        negative_similarity = 0.5 * (
            torch.sum(representation[negative_index] * representation[left])
            + torch.sum(representation[negative_index] * representation[right])
        )
        current_margin = positive_similarity - negative_similarity
        # Analytic feature-space gradient of the local margin.  Multiplying by
        # -grad(GPC) estimates the first-order margin change under a GPC-only
        # descent step without building a second-order graph.
        margin_grad_left = representation[right] - 0.5 * representation[negative_index]
        margin_grad_right = representation[left] - 0.5 * representation[negative_index]
        margin_grad_negative = -0.5 * (representation[left] + representation[right])
        impact = (
            torch.sum(margin_grad_left * (-gpc_grad[left]))
            + torch.sum(margin_grad_right * (-gpc_grad[right]))
            + torch.sum(margin_grad_negative * (-gpc_grad[negative_index]))
        ).detach()
        if bool(torch.isfinite(impact).item()) and float(impact.cpu()) < 0.0:
            selected.append((current_margin, impact, negative_pid))
    return selected


def gpc_relation_key(image_key: Any, state_key: Any) -> str:
    """Return a stable, train-only identifier for a reusable GPC state relation.

    ``image_key`` identifies one sampled image pair and almost never repeats
    across epochs.  The frozen selector's ``state_key`` identifies the actual
    same-PID/clothes relation, so it is the correct granularity for carrying a
    measured post-step risk to a later batch.
    """
    return json.dumps(
        ["state", json_safe(state_key)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def select_ndp_gpc_delta_probes(
    representation: torch.Tensor,
    pids: torch.Tensor,
    selected_gpc: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]],
) -> list[dict[str, Any]]:
    """Choose the local relation whose *actual* post-step change is measured.

    This is deliberately a probe only: it adds no gradient and does not use a
    threshold.  The nearest different-PID image is fixed before the optimizer
    step; the same three current-batch features are evaluated again after the
    step by ``measure_gpc_delta_after``.
    """
    if not selected_gpc:
        return []
    probes: list[dict[str, Any]] = []
    for left, right, image_key, state_key in selected_gpc:
        left = int(left)
        right = int(right)
        pid = int(pids[left].detach().cpu())
        if int(pids[right].detach().cpu()) != pid:
            continue
        candidate_indices = torch.nonzero(pids != pid, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        with torch.no_grad():
            pair_scores = 0.5 * (
                representation[candidate_indices] @ representation[left]
                + representation[candidate_indices] @ representation[right]
            )
            negative_index = int(candidate_indices[int(torch.argmax(pair_scores).item())].item())
            negative_pid = int(pids[negative_index].detach().cpu())
        positive_similarity = torch.sum(representation[left] * representation[right])
        negative_similarity = 0.5 * (
            torch.sum(representation[negative_index] * representation[left])
            + torch.sum(representation[negative_index] * representation[right])
        )
        probes.append({
            "left": left,
            "right": right,
            "negative": negative_index,
            "negative_pid": negative_pid,
            "key": gpc_relation_key(image_key, state_key),
            "before_margin": float((positive_similarity - negative_similarity).detach().float().cpu()),
        })
    return probes


def select_ndp_gpc_delta_protections(
    representation: torch.Tensor,
    pids: torch.Tensor,
    selected_gpc: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]],
    observed_risks: dict[str, dict[str, Any]] | None,
) -> list[tuple[torch.Tensor, float, int]]:
    """Protect only relations previously observed to lose margin in reality.

    The target comes from the relation's own pre-step margin.  There is no
    global Margin gate and no teacher-space target; a relation is eligible only
    after a real optimizer update has decreased its current-batch margin.
    """
    if not selected_gpc or not observed_risks:
        return []
    protections: list[tuple[torch.Tensor, float, int]] = []
    for probe in select_ndp_gpc_delta_probes(representation, pids, selected_gpc):
        risk = observed_risks.get(str(probe["key"]))
        if not risk:
            continue
        left = int(probe["left"])
        right = int(probe["right"])
        negative = int(probe["negative"])
        current_margin = (
            torch.sum(representation[left] * representation[right])
            - 0.5 * (
                torch.sum(representation[negative] * representation[left])
                + torch.sum(representation[negative] * representation[right])
            )
        )
        target = float(risk.get("target_margin", risk.get("before_margin", 0.0)))
        if math.isfinite(target) and float(current_margin.detach().float().cpu()) < target:
            protections.append((current_margin, target, int(probe["negative_pid"])))
    return protections


@torch.no_grad()
def measure_gpc_delta_after(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    probes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Measure the same local margins after the actual optimizer update.

    The extra forward is executed only for batches containing a GPC probe.  It
    uses the raw module (not DDP communication), and restores the device RNG so
    the diagnostic cannot change the training trajectory.
    """
    if not probes:
        return []
    raw_model = model.module if hasattr(model, "module") else model
    rng = base.rng_state(device)
    unique_indices = sorted({int(probe[key]) for probe in probes for key in ("left", "right", "negative")})
    local_index = {original: position for position, original in enumerate(unique_indices)}
    index_tensor = torch.as_tensor(unique_indices, dtype=torch.long, device=device)
    images = batch["image"].to(device, non_blocking=False).index_select(0, index_tensor)
    camids = batch["camid"].to(device, non_blocking=False).index_select(0, index_tensor)
    with amp.autocast(enabled=True):
        _score, feat = raw_model(images, cam_label=camids)
    representation = F.normalize(feat[0].float(), dim=-1)
    base.restore_rng(rng, device)
    observed: list[dict[str, Any]] = []
    for probe in probes:
        left = local_index[int(probe["left"])]
        right = local_index[int(probe["right"])]
        negative = local_index[int(probe["negative"])]
        after_margin = float((
            torch.sum(representation[left] * representation[right])
            - 0.5 * (
                torch.sum(representation[negative] * representation[left])
                + torch.sum(representation[negative] * representation[right])
            )
        ).float().cpu())
        before_margin = float(probe["before_margin"])
        observed.append({
            "key": str(probe["key"]),
            "before_margin": before_margin,
            "after_margin": after_margin,
            "delta": after_margin - before_margin,
            "negative_pid": int(probe["negative_pid"]),
        })
    return observed


def gpc_delta_fix_relation_key(state_key: Any, negative_pid: int) -> str:
    """Identify one fixed positive-state/negative-PID relation.

    The old DELTA key stopped at ``state_key`` and therefore allowed the
    negative identity to change between observations.  DELTA_FIX makes the
    negative PID part of the key and stores the serialized state separately
    for an explicit equality check during protection.
    """
    return json.dumps(
        ["state_negative_pid", json_safe(state_key), int(negative_pid)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _local_identity_margin(
    representation: torch.Tensor,
    left: int,
    right: int,
    negative: int,
) -> torch.Tensor:
    return torch.sum(representation[left] * representation[right]) - 0.5 * (
        torch.sum(representation[negative] * representation[left])
        + torch.sum(representation[negative] * representation[right])
    )


@torch.no_grad()
def _eval_batch_representation(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    indices: list[int],
) -> torch.Tensor:
    """Encode selected batch images in deterministic eval mode.

    DELTA_FIX uses this same mode for both pre-step and post-step snapshots.
    The model's training state and RNG are restored so the snapshot cannot
    alter the optimizer trajectory or stochastic training stream.
    """
    raw_model = model.module if hasattr(model, "module") else model
    rng = base.rng_state(device)
    was_training = bool(raw_model.training)
    raw_model.eval()
    try:
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        images = batch["image"].to(device, non_blocking=False).index_select(0, index_tensor)
        camids = batch["camid"].to(device, non_blocking=False).index_select(0, index_tensor)
        with amp.autocast(enabled=True):
            outputs = raw_model(images, cam_label=camids)
        if isinstance(outputs, (tuple, list)):
            feature_output = outputs[1]
            feature_output = feature_output[0] if isinstance(feature_output, (tuple, list)) else feature_output
        else:
            # The deployed Bio model returns the feature tensor directly in
            # eval mode and the ``(score, [feature])`` pair in train mode.
            feature_output = outputs
        return F.normalize(feature_output.float(), dim=-1)
    finally:
        base.restore_rng(rng, device)
        raw_model.train(was_training)


def _nearest_negative_index(
    representation: torch.Tensor,
    pids: torch.Tensor,
    left: int,
    right: int,
    negative_pid: int | None = None,
) -> int | None:
    positive_pid = int(pids[left].detach().cpu())
    if int(pids[right].detach().cpu()) != positive_pid:
        return None
    if negative_pid is None:
        candidate_indices = torch.nonzero(pids != positive_pid, as_tuple=False).flatten()
    else:
        candidate_indices = torch.nonzero(pids == int(negative_pid), as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return None
    with torch.no_grad():
        pair_scores = 0.5 * (
            representation[candidate_indices] @ representation[left]
            + representation[candidate_indices] @ representation[right]
        )
        return int(candidate_indices[int(torch.argmax(pair_scores).item())].item())


def select_ndp_delta_fix_probes(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    pids: torch.Tensor,
    selected_gpc: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]],
) -> list[dict[str, Any]]:
    """Snapshot exact GPC relations before the real optimizer update.

    Candidate identity selection and the pre-step margin are both measured in
    deterministic eval mode.  The post-step function below uses the same
    mode and the same three batch indices, removing the train/eval and
    negative-PID mismatch in the original DELTA arm.
    """
    if not selected_gpc:
        return []
    all_indices = list(range(int(pids.shape[0])))
    snapshot = _eval_batch_representation(model, batch, device, all_indices)
    probes: list[dict[str, Any]] = []
    for left, right, _image_key, state_key in selected_gpc:
        left = int(left)
        right = int(right)
        negative = _nearest_negative_index(snapshot, pids, left, right)
        if negative is None:
            continue
        negative_pid = int(pids[negative].detach().cpu())
        probes.append({
            "left": left,
            "right": right,
            "negative": negative,
            "negative_pid": negative_pid,
            "positive_pid": int(pids[left].detach().cpu()),
            "state_key": json_safe(state_key),
            "key": gpc_delta_fix_relation_key(state_key, negative_pid),
            "before_margin": float(_local_identity_margin(snapshot, left, right, negative).float().cpu()),
        })
    return probes


def select_ndp_delta_fix_protections(
    representation: torch.Tensor,
    pids: torch.Tensor,
    selected_gpc: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]],
    observed_risks: dict[str, dict[str, Any]] | None,
    max_relations: int = NDP_DELTA_FIX_MAX_PROTECTIONS_PER_BATCH,
) -> list[tuple[torch.Tensor, float, int, str]]:
    """Return at most the single largest deficit for an armed exact relation.

    A risk is eligible only after one observed decrease for the same
    positive state and negative PID.  Its target is fixed at first evidence;
    it is never replaced by a later maximum and is consumed once after a
    successful optimizer update.
    """
    if not selected_gpc or not observed_risks or max_relations <= 0:
        return []
    candidates: list[tuple[float, torch.Tensor, float, int, str]] = []
    for left, right, _image_key, state_key in selected_gpc:
        left = int(left)
        right = int(right)
        state_value = json_safe(state_key)
        for key, risk in observed_risks.items():
            if risk.get("used", False) or not risk.get("armed", False):
                continue
            if int(risk.get("observations", 0)) < NDP_DELTA_FIX_MIN_OBSERVATIONS:
                continue
            if risk.get("state_key") != state_value:
                continue
            negative_pid = int(risk.get("negative_pid", -1))
            negative = _nearest_negative_index(representation, pids, left, right, negative_pid)
            if negative is None:
                continue
            current_margin = _local_identity_margin(representation, left, right, negative)
            target = float(risk.get("target_margin", float("nan")))
            if not math.isfinite(target):
                continue
            deficit = target - float(current_margin.detach().float().cpu())
            if deficit <= 0.0:
                continue
            candidates.append((deficit, current_margin, target, negative_pid, str(key)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(item[1], item[2], item[3], item[4]) for item in candidates[:max_relations]]


@torch.no_grad()
def measure_ndp_delta_fix_after(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    probes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Measure the exact pre-step probe indices after optimizer.step()."""
    if not probes:
        return []
    unique_indices = sorted({int(probe[name]) for probe in probes for name in ("left", "right", "negative")})
    local_index = {original: position for position, original in enumerate(unique_indices)}
    representation = _eval_batch_representation(model, batch, device, unique_indices)
    observed: list[dict[str, Any]] = []
    for probe in probes:
        left = local_index[int(probe["left"])]
        right = local_index[int(probe["right"])]
        negative = local_index[int(probe["negative"])]
        after_margin = float(_local_identity_margin(representation, left, right, negative).float().cpu())
        before_margin = float(probe["before_margin"])
        observed.append({
            "key": str(probe["key"]),
            "state_key": probe["state_key"],
            "positive_pid": int(probe["positive_pid"]),
            "negative_pid": int(probe["negative_pid"]),
            "before_margin": before_margin,
            "after_margin": after_margin,
            "delta": after_margin - before_margin,
        })
    return observed


def gradient_gate_from_grads(g_base: torch.Tensor | None, g_ndp: torch.Tensor | None) -> tuple[bool, float, str]:
    """Return a detached compatibility decision for two feature gradients.

    A missing, non-finite, or zero-norm gradient is conservatively rejected.
    The returned cosine and status are diagnostics only; no gate quantity is
    part of the autograd graph.
    """
    if g_base is None or g_ndp is None:
        return False, float("nan"), "missing_gradient"
    base_vec = g_base.detach().float().reshape(-1)
    ndp_vec = g_ndp.detach().float().reshape(-1)
    base_norm = torch.linalg.vector_norm(base_vec)
    ndp_norm = torch.linalg.vector_norm(ndp_vec)
    if not bool(torch.isfinite(base_norm).item()) or not bool(torch.isfinite(ndp_norm).item()):
        return False, float("nan"), "nonfinite_norm"
    if float(base_norm.item()) <= 0.0 or float(ndp_norm.item()) <= 0.0:
        return False, float("nan"), "zero_norm"
    cosine_tensor = F.cosine_similarity(base_vec.unsqueeze(0), ndp_vec.unsqueeze(0), dim=1)[0]
    cosine = float(cosine_tensor.detach().cpu())
    if not math.isfinite(cosine):
        return False, float("nan"), "nonfinite_cosine"
    return cosine >= 0.0, cosine, "kept" if cosine >= 0.0 else "conflict"


def gradient_gate_decision(main_total: torch.Tensor, ndp_loss: torch.Tensor, feature: torch.Tensor) -> tuple[bool, float, str]:
    """Compute the current-rank gate without differentiating through it."""
    if not main_total.requires_grad or not ndp_loss.requires_grad or not feature.requires_grad:
        return False, float("nan"), "missing_autograd"
    try:
        g_base = torch.autograd.grad(main_total, feature, retain_graph=True, allow_unused=True)[0]
        g_ndp = torch.autograd.grad(ndp_loss, feature, retain_graph=True, allow_unused=True)[0]
    except RuntimeError:
        return False, float("nan"), "autograd_error"
    return gradient_gate_from_grads(g_base, g_ndp)


def nrg_should_select(reference_margin: float, current_margin: float, epsilon: float = EPSILON) -> bool:
    """RiskGate: retain a historically safe but currently near-boundary PID relation."""
    return bool(reference_margin >= epsilon and 0.0 < current_margin < epsilon)


def listwise_local_rank_loss(
    positive_similarity: torch.Tensor,
    negative_similarities: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Return a temperature-scaled local listwise ranking loss.

    The positive and negative endpoint features are expected to be detached
    before their similarities are passed here.  The live anchor remains in
    the similarity graph, so the returned loss can still backpropagate to the
    current model feature.  Empty negative lists return a graph-connected zero
    when a tensor positive is supplied.
    """
    if tau <= 0.0 or not math.isfinite(float(tau)):
        raise ValueError(f"tau must be finite and positive, got {tau}")
    if positive_similarity.numel() != 1:
        raise ValueError("positive_similarity must contain exactly one value")
    if negative_similarities.numel() == 0:
        return positive_similarity.reshape(()).sum() * 0.0
    negatives = negative_similarities.reshape(-1)
    logits = torch.cat((positive_similarity.reshape(1), negatives), dim=0) / float(tau)
    return float(tau) * (torch.logsumexp(logits, dim=0) - logits[0])


def ndyn_should_select(margin0: float, margin1: float, delta: float = NDYN_DELTA) -> bool:
    """Select a relation correct at both nodes but measurably degraded."""
    return bool(margin0 > 0.0 and margin1 > 0.0 and margin1 < margin0 - delta and margin1 < 0.05)


def compose_total_loss(main_total: torch.Tensor, ndp_loss: torch.Tensor, arm: str, ndp_lambda: float | None = None) -> torch.Tensor:
    """Compose the fixed B0 loss or the B0+NDP loss without cross-contamination."""
    if arm == "B0":
        total = main_total
        if total is not main_total or not torch.equal(total, main_total):
            raise AssertionError("B0 total loss must be exactly main_total")
        return total
    if arm in {"N5", "NF0", "NRG", "NCAP", "NDYN", "NDYN_TOP5", "NGATE", "NFRESH", "NCOMP", "NONLINE", "NLIST", "NDP_SAFE", "NDP_SELECT", "NDP_V2", "NTEACH", "NTEACH_SAFE", "NDP_RANK", "NDP_SR", "NDP_LATE", "NDP_GPC", "NDP_GPC_IMPACT", "NDP_GPC_DELTA", "NDP_DELTA_FIX", "NDP_AD", "NDP_SHN", "NDP_SHN_MP", "NDP_SHN_MP_ABM", "NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C", "NDP_SHN_BANK", "NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"}:
        coefficient = (LAMBDA_NDP_GPC if arm == "NDP_GPC" else (0.001 if arm in {"NDP_GPC_IMPACT", "NDP_GPC_DELTA", "NDP_DELTA_FIX", "NDP_AD", "NDP_SHN", "NDP_SHN_MP", "NDP_SHN_MP_ABM", "NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C", "NDP_SHN_BANK", "NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"} else LAMBDA_NDP)) if ndp_lambda is None else float(ndp_lambda)
        return main_total + coefficient * ndp_loss
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


def forward_batch(model, loss_fn, batch, device, path_to_index, bank, reference_rows, reference_features, arm, epoch, rank, iteration, global_step, stats, diagnostic_steps, ndp_epsilon, ndp_lambda=None, gpc_delta_risks=None, shn_bank_features=None, shn_bank_pids=None, shn_bank_indices=None, shn_bank_centers=None, shn_bank_center_pids=None, shn_bank_clothes=None, shn_batch_clothes=None):
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
    hard_negative_pids = None
    hard_negative_valid = None
    if arm == "NCOMP" or arm in {"NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C"}:
        # Only the detached filter uses the raw identity feature.  The live
        # feat[0] remains untouched for the original Triplet and NDP paths.
        hard_negative_pids, hard_negative_valid = batch_hard_negative_pid(feat[0], pids)
    selected_gpc = chosen["id"]
    gpc_loss = torch.stack([1.0 - torch.sum(representation[i] * representation[j]) for i, j, _key, _state in selected_gpc]).mean() if selected_gpc else representation.sum() * 0.0
    main_total = base_total + LAMBDA_GPC * gpc_loss

    selected_ndp = active_ndp = current_error_pid = correct_pid = 0
    triplet_hard_excluded = 0
    triplet_filter_candidates = 0
    margin_sum = 0.0
    gpc_delta_probes = select_ndp_gpc_delta_probes(representation, pids, selected_gpc) if arm == "NDP_GPC_DELTA" else []
    delta_fix_probes = select_ndp_delta_fix_probes(model, batch, device, pids, selected_gpc) if arm == "NDP_DELTA_FIX" else []
    ndp_ad_relations = (
        select_ndp_ad_relations(representation, pids, selected_gpc)
        if arm == "NDP_AD" and epoch >= 3 and stats.get("ndp_ad_protection_batches", 0) < NDP_AD_MAX_PROTECTION_BATCHES
        else []
    )
    abm_info: dict[str, Any] = {}
    capr_info: dict[str, Any] = {}
    capr_losses: list[tuple[torch.Tensor, float, int, int]] = []
    shn_selection_info: dict[str, Any] = {}
    if arm in {"NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C"} and epoch >= 3:
        ndp_shn_losses, shn_selection_info = select_ndp_shn_mp_losses(representation, pids, return_info=True)
        capr_losses, capr_info = select_ndp_shn_mp_capr_losses(
            representation,
            pids,
            clothes,
            variant=arm[-1],
            original_info=shn_selection_info,
            return_info=True,
        )
    elif arm == "NDP_SHN_MP_ABM" and epoch >= 3:
        ndp_shn_losses, abm_info = select_ndp_shn_mp_abm_losses(representation, pids, return_info=True)
    elif arm in {"NDP_SHN", "NDP_SHN_MP", "NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"} and epoch >= 3:
        ndp_shn_losses = (select_ndp_shn_mp_losses if arm in {"NDP_SHN_MP", "NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"} else select_ndp_shn_losses)(representation, pids)
    else:
        ndp_shn_losses = []
    ndp_shn_bank_losses = (
        select_ndp_shn_bank_losses(
            representation,
            pids,
            shn_bank_features,
            shn_bank_pids,
            shn_bank_indices,
            bank_centers=shn_bank_centers,
            bank_center_pids=shn_bank_center_pids,
            bank_clothes=shn_bank_clothes,
            batch_clothes=shn_batch_clothes,
        )
        if arm == "NDP_SHN_BANK" and epoch >= 3 and shn_bank_features is not None and shn_bank_pids is not None and shn_bank_indices is not None
        else []
    )
    ndp_shn_reliable_bank_aux_losses = (
        select_ndp_shn_reliable_bank_aux_losses(
            representation,
            pids,
            shn_bank_features,
            shn_bank_pids,
            shn_bank_indices,
            bank_centers=shn_bank_centers,
            bank_center_pids=shn_bank_center_pids,
            bank_clothes=shn_bank_clothes,
            batch_clothes=shn_batch_clothes,
            risk_aware=arm == "NDP_SHN_RELIABLE_RISK",
            anchor_boundary=arm == "NDP_SHN_RELIABLE_ANCHOR",
        )
        if arm in {"NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"} and epoch >= 3 and shn_bank_features is not None and shn_bank_pids is not None and shn_bank_indices is not None
        else []
    )
    shn_losses = ndp_shn_losses + ndp_shn_bank_losses
    ndp_enabled = arm in {"N5", "NF0", "NRG", "NCAP", "NDYN", "NDYN_TOP5", "NGATE", "NFRESH", "NCOMP", "NONLINE", "NLIST", "NDP_SAFE", "NDP_SELECT", "NDP_V2", "NTEACH", "NTEACH_SAFE", "NDP_RANK", "NDP_SR", "NDP_LATE", "NDP_GPC", "NDP_GPC_IMPACT", "NDP_GPC_DELTA", "NDP_DELTA_FIX", "NDP_AD", "NDP_SHN", "NDP_SHN_MP", "NDP_SHN_MP_ABM", "NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C", "NDP_SHN_BANK", "NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"}
    if ndp_enabled:
        ndp_values = []
        if arm == "NDP_GPC":
            for current_margin, _negative_pid in select_ndp_gpc_confusions(representation, pids, selected_gpc):
                selected_ndp += 1
                active_ndp += 1
                current_error_pid += 1
                margin_sum += float(current_margin.detach().float().cpu())
                ndp_values.append(
                    F.relu(
                        torch.as_tensor(NDP_GPC_TARGET_MARGIN, dtype=current_margin.dtype, device=device)
                        - current_margin
                    )
                )
        if arm == "NDP_GPC_IMPACT":
            for current_margin, _impact, _negative_pid in select_ndp_gpc_impacts(representation, pids, selected_gpc, gpc_loss):
                selected_ndp += 1
                current_error_pid += int(float(current_margin.detach().float().cpu()) <= 0.0)
                correct_pid += int(float(current_margin.detach().float().cpu()) > 0.0)
                margin_sum += float(current_margin.detach().float().cpu())
                hinge = F.relu(
                    torch.as_tensor(NDP_GPC_IMPACT_TARGET_MARGIN, dtype=current_margin.dtype, device=device)
                    - current_margin
                )
                active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                ndp_values.append(hinge)
        if arm == "NDP_GPC_DELTA":
            for current_margin, target_margin, _negative_pid in select_ndp_gpc_delta_protections(
                representation, pids, selected_gpc, gpc_delta_risks
            ):
                selected_ndp += 1
                current_value = float(current_margin.detach().float().cpu())
                current_error_pid += int(current_value <= 0.0)
                correct_pid += int(current_value > 0.0)
                margin_sum += current_value
                hinge = F.relu(
                    torch.as_tensor(target_margin, dtype=current_margin.dtype, device=device)
                    - current_margin
                )
                active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                ndp_values.append(hinge)
        delta_fix_keys: list[str] = []
        if arm == "NDP_DELTA_FIX" and stats.get("delta_fix_protection_batches", 0) < NDP_DELTA_FIX_MAX_PROTECTION_BATCHES:
            for current_margin, target_margin, _negative_pid, risk_key in select_ndp_delta_fix_protections(
                representation, pids, selected_gpc, gpc_delta_risks, max_relations=1
            ):
                selected_ndp += 1
                current_value = float(current_margin.detach().float().cpu())
                current_error_pid += int(current_value <= 0.0)
                correct_pid += int(current_value > 0.0)
                margin_sum += current_value
                hinge = F.relu(
                    torch.as_tensor(target_margin, dtype=current_margin.dtype, device=device)
                    - current_margin
                )
                active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                ndp_values.append(hinge)
                delta_fix_keys.append(risk_key)
            if delta_fix_keys:
                stats["delta_fix_protection_batches"] += 1
        if arm == "NDP_AD" and ndp_ad_relations:
            for positive_similarity, negative_similarity, adaptive_weight, _negative_pid, positive_common, negative_common in ndp_ad_relations:
                selected_ndp += 1
                current_margin = positive_similarity - negative_similarity
                detached_margin = float(current_margin.detach().float().cpu())
                current_error_pid += int(detached_margin <= 0.0)
                correct_pid += int(detached_margin > 0.0)
                margin_sum += detached_margin
                # ANFI-inspired discrepancy weighting is detached, while the
                # live pairwise similarities remain differentiable.  This is
                # a target-free local ranking penalty rather than restoration
                # to a historical margin.
                relation_loss = adaptive_weight * (NDP_AD_TAU * F.softplus((negative_similarity - positive_similarity) / NDP_AD_TAU))
                ndp_values.append(relation_loss)
                active_ndp += int(float(relation_loss.detach().float().cpu()) > 0.0)
            stats["ndp_ad_protection_batches"] += 1
        if arm in {"NDP_SHN", "NDP_SHN_MP", "NDP_SHN_MP_ABM", "NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C", "NDP_SHN_BANK", "NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"} and shn_losses:
            for shn_loss, detached_margin, negative_count, positive_count in shn_losses:
                selected_ndp += 1
                active_ndp += int(float(shn_loss.detach().float().cpu()) > 0.0)
                current_error_pid += int(detached_margin <= 0.0)
                correct_pid += int(detached_margin > 0.0)
                margin_sum += detached_margin
                if arm not in {"NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C"}:
                    ndp_values.append(shn_loss)
        if arm in {"NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C"} and capr_losses:
            for capr_loss, detached_margin, negative_count, positive_count in capr_losses:
                selected_ndp += 1
                active_ndp += int(float(capr_loss.detach().float().cpu()) > 0.0)
                current_error_pid += int(detached_margin <= 0.0)
                correct_pid += int(detached_margin > 0.0)
                margin_sum += detached_margin
        if arm in {"NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"} and ndp_shn_reliable_bank_aux_losses:
            for aux_loss, detached_margin, negative_count, positive_count, _reliability in ndp_shn_reliable_bank_aux_losses:
                selected_ndp += 1
                active_ndp += int(float(aux_loss.detach().float().cpu()) > 0.0)
                current_error_pid += int(detached_margin <= 0.0)
                correct_pid += int(detached_margin > 0.0)
                margin_sum += detached_margin
        for local_index, entry in enumerate(entries):
            if arm in {"NDP_GPC", "NDP_GPC_IMPACT", "NDP_GPC_DELTA", "NDP_DELTA_FIX", "NDP_AD", "NDP_SHN", "NDP_SHN_MP", "NDP_SHN_MP_ABM", "NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C", "NDP_SHN_BANK", "NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"}:
                continue
            anchor_id = int(path_to_index[entry["image_path"]])
            anchor = representation[local_index]
            anchor_values = []
            listwise_negative_similarities = []
            sr_candidates = []
            for relation in reference_rows.get(anchor_id, []):
                positive = reference_features[relation["positive_local"]]
                negative = reference_features[relation["negative_local"]]
                positive_similarity = (positive @ anchor).mean()
                for pid_index, is_valid in enumerate(relation["historical_valid_pid"]):
                    if not is_valid:
                        continue
                    if arm == "NCOMP":
                        triplet_filter_candidates += 1
                        if bool(hard_negative_valid[local_index].item()) and ndp_relation_excluded_by_triplet_pid(
                            relation["negative_pid"][pid_index],
                            int(hard_negative_pids[local_index].item()),
                        ):
                            triplet_hard_excluded += 1
                            continue
                    current_margin = positive_similarity - negative[pid_index] @ anchor
                    reference_margin = float(relation["reference_pid_margin"][pid_index])
                    if arm == "NDP_SAFE":
                        # Low-interference local identity hinge.  The graph
                        # endpoint bank is detached; only the live anchor is
                        # connected to the current EVA02 computation.
                        selected_ndp += 1
                        hinge = F.relu(
                            torch.as_tensor(NDP_SAFE_MARGIN, dtype=current_margin.dtype, device=device)
                            - current_margin
                        )
                        anchor_values.append(hinge)
                        active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                        current_error_pid += int(float(current_margin.detach().float().cpu()) <= 0.0)
                        correct_pid += int(float(current_margin.detach().float().cpu()) > 0.0)
                        margin_sum += float(current_margin.detach().float().cpu())
                        continue
                    if arm == "NDP_SELECT":
                        # Selective low-interference gate: only a reliable
                        # cross-clothes positive and a currently correct but
                        # near-boundary negative relation are eligible.  The
                        # detached predicate never participates in backward.
                        if not bool(relation.get("safe_positive_cross_clothes", False)):
                            continue
                        detached_margin = float(current_margin.detach().float().cpu())
                        if not (0.0 < detached_margin < 0.05):
                            continue
                        selected_ndp += 1
                        hinge = F.relu(
                            torch.as_tensor(0.02, dtype=current_margin.dtype, device=device)
                            - current_margin
                        )
                        anchor_values.append(hinge)
                        active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                        current_error_pid += int(detached_margin <= 0.0)
                        correct_pid += int(detached_margin > 0.0)
                        margin_sum += detached_margin
                        continue
                    if arm == "NDP_RANK":
                        # Teacher-guided local ranking.  The teacher graph is
                        # fixed and detached; only the live student anchor is
                        # connected to the loss.  We keep reliable but
                        # genuinely dangerous teacher relations, rather than
                        # forcing recovery of every historical neighbor.
                        teacher_margin = float(relation.get("teacher_pid_margin", [reference_margin] * len(relation["negative_pid"]))[pid_index])
                        detached_margin = float(current_margin.detach().float().cpu())
                        if not (NDP_RANK_TEACHER_MIN_MARGIN <= teacher_margin <= NDP_RANK_TEACHER_MAX_MARGIN and detached_margin < NDP_RANK_CURRENT_MAX_MARGIN):
                            continue
                        selected_ndp += 1
                        target_margin = min(NDP_RANK_TARGET_MAX_MARGIN, max(NDP_RANK_TARGET_MIN_MARGIN, 0.5 * teacher_margin))
                        hinge = F.relu(
                            torch.as_tensor(target_margin, dtype=current_margin.dtype, device=device)
                            - current_margin
                        )
                        anchor_values.append(hinge)
                        active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                        current_error_pid += int(detached_margin <= 0.0)
                        correct_pid += int(detached_margin > 0.0)
                        margin_sum += detached_margin
                        continue
                    if arm == "NDP_SR":
                        # Only the teacher-selected relation/index set is
                        # reused.  The endpoints in reference_features are a
                        # refreshed student train bank.
                        teacher_margin = float(relation.get("teacher_pid_margin", [reference_margin] * len(relation["negative_pid"]))[pid_index])
                        detached_margin = float(current_margin.detach().float().cpu())
                        if not (NDP_SR_TEACHER_MIN_MARGIN <= teacher_margin <= NDP_SR_TEACHER_MAX_MARGIN and detached_margin < NDP_SR_CONFUSION_MARGIN):
                            continue
                        sr_candidates.append((detached_margin, current_margin))
                        continue
                    if arm in {"NTEACH", "NTEACH_SAFE"}:
                        # The relation endpoints and teacher margin were
                        # captured once from the B0 initialization.  Protect
                        # only a reliable teacher ordering that has actually
                        # crossed into a mild current identity confusion.
                        teacher_margin = float(relation.get("teacher_pid_margin", [reference_margin] * len(relation["negative_pid"]))[pid_index])
                        detached_margin = float(current_margin.detach().float().cpu())
                        if teacher_margin < 0.02 or not (-0.05 < detached_margin < 0.0):
                            continue
                        # NTEACH_SAFE additionally requires the teacher's
                        # positive endpoint to be supported by two distinct
                        # clothes identities.  This is a detached reliability
                        # gate; the live anchor remains the only gradient path.
                        if arm == "NTEACH_SAFE" and not bool(relation.get("safe_positive_reliable", False)):
                            continue
                        selected_ndp += 1
                        target_margin = min(0.05, max(0.01, 0.25 * teacher_margin))
                        hinge = F.relu(
                            torch.as_tensor(target_margin, dtype=current_margin.dtype, device=device)
                            - current_margin
                        )
                        anchor_values.append(hinge)
                        active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                        current_error_pid += 1
                        margin_sum += detached_margin
                        continue
                    if arm == "NDP_LATE":
                        # Delay intervention until the base Bio+GPC model has
                        # learned its identity space.  After the delay, only
                        # a teacher-reliable relation that has crossed into a
                        # real current confusion (margin <= 0) is eligible.
                        # The endpoint bank remains in the live student space.
                        if epoch < NDP_LATE_START_EPOCH:
                            continue
                        teacher_margin = float(relation.get("teacher_pid_margin", [reference_margin] * len(relation["negative_pid"]))[pid_index])
                        detached_margin = float(current_margin.detach().float().cpu())
                        if not (
                            NDP_LATE_TEACHER_MIN_MARGIN <= teacher_margin <= NDP_LATE_TEACHER_MAX_MARGIN
                            and detached_margin <= NDP_LATE_CONFUSION_MARGIN
                        ):
                            continue
                        sr_candidates.append((detached_margin, current_margin))
                        continue
                    if arm == "NDP_V2":
                        # V2 keeps only a robust, multi-clothes positive
                        # reference.  Unlike NDP_SELECT it also admits a
                        # modest negative current margin, but only in a
                        # narrow near-boundary band and with a small positive
                        # target to avoid forcing full historical recovery.
                        if not bool(relation.get("safe_positive_reliable", False)):
                            continue
                        detached_margin = float(current_margin.detach().float().cpu())
                        if not (-0.05 < detached_margin < 0.05):
                            continue
                        selected_ndp += 1
                        target_margin = 0.005 if detached_margin <= 0.0 else 0.02
                        hinge = F.relu(
                            torch.as_tensor(target_margin, dtype=current_margin.dtype, device=device)
                            - current_margin
                        )
                        anchor_values.append(hinge)
                        active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                        current_error_pid += int(detached_margin <= 0.0)
                        correct_pid += int(detached_margin > 0.0)
                        margin_sum += detached_margin
                        continue
                    if arm == "NLIST":
                        selected_ndp += 1
                        listwise_negative_similarities.append(negative[pid_index] @ anchor)
                        current_error_pid += int(float(current_margin.detach().float().cpu()) <= 0.0)
                        correct_pid += int(float(current_margin.detach().float().cpu()) > 0.0)
                        margin_sum += float(current_margin.detach().float().cpu())
                        continue
                    if arm == "NONLINE":
                        # NONLINE has no historical target.  Its endpoint
                        # references were mined from the current epoch bank;
                        # the only fixed target is the small live boundary.
                        target_margin = 0.02
                    elif arm == "NRG" and not nrg_should_select(
                        reference_margin,
                        float(current_margin.detach().float().cpu()),
                        ndp_epsilon,
                    ):
                        continue
                    selected_ndp += 1
                    if arm == "NONLINE":
                        target_margin = 0.02
                    elif arm in {"NDYN_TOP5", "NGATE", "NFRESH", "NCOMP"}:
                        target_margin = ndyn_top5_target_margin(reference_margin, float(relation["dynamic_drop_pid"][pid_index]))
                    else:
                        target_margin = ndp_target_margin(reference_margin, ndp_epsilon, arm)
                    hinge = F.relu(torch.as_tensor(target_margin, dtype=current_margin.dtype, device=device) - current_margin)
                    anchor_values.append(hinge)
                    active_ndp += int(float(hinge.detach().float().cpu()) > 0.0)
                    current_error_pid += int(float(current_margin.detach().float().cpu()) <= 0.0)
                    correct_pid += int(float(current_margin.detach().float().cpu()) > 0.0)
                    margin_sum += float(current_margin.detach().float().cpu())
            if arm == "NDP_SR" and sr_candidates:
                _, current_margin = min(sr_candidates, key=lambda item: item[0])
                selected_ndp += 1
                hinge = F.relu(
                    torch.as_tensor(NDP_SR_CONFUSION_MARGIN, dtype=current_margin.dtype, device=device)
                    - current_margin
                )
                anchor_values.append(hinge)
                detached_margin = float(current_margin.detach().float().cpu())
                active_ndp += int(detached_margin < NDP_SR_CONFUSION_MARGIN)
                current_error_pid += int(detached_margin <= 0.0)
                correct_pid += int(detached_margin > 0.0)
                margin_sum += detached_margin
            if arm == "NDP_LATE" and epoch >= NDP_LATE_START_EPOCH and sr_candidates:
                _, current_margin = min(sr_candidates, key=lambda item: item[0])
                selected_ndp += 1
                hinge = F.relu(
                    torch.as_tensor(NDP_LATE_TARGET_MARGIN, dtype=current_margin.dtype, device=device)
                    - current_margin
                )
                anchor_values.append(hinge)
                detached_margin = float(current_margin.detach().float().cpu())
                active_ndp += int(detached_margin <= NDP_LATE_CONFUSION_MARGIN)
                current_error_pid += int(detached_margin <= 0.0)
                margin_sum += detached_margin
            if arm == "NLIST" and listwise_negative_similarities:
                listwise_loss = listwise_local_rank_loss(
                    positive_similarity,
                    torch.stack(listwise_negative_similarities),
                    tau=0.1,
                )
                ndp_values.append(listwise_loss)
                if float(listwise_loss.detach().float().cpu()) > 0.0:
                    active_ndp += len(listwise_negative_similarities)
            if anchor_values:
                ndp_values.append(torch.stack(anchor_values).mean())
        if arm in {"NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"}:
            live_loss = (
                torch.stack([item[0] for item in ndp_shn_losses]).mean()
                if ndp_shn_losses
                else representation.sum() * 0.0
            )
            bank_aux_loss = (
                torch.stack([item[0] for item in ndp_shn_reliable_bank_aux_losses]).mean()
                if ndp_shn_reliable_bank_aux_losses
                else representation.sum() * 0.0
            )
            ndp_loss = live_loss + NDP_SHN_RELIABLE_BANK_AUX_WEIGHT * bank_aux_loss
        elif arm in {"NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C"}:
            live_loss = (
                torch.stack([item[0] for item in ndp_shn_losses]).mean()
                if ndp_shn_losses
                else representation.sum() * 0.0
            )
            capr_loss = (
                torch.stack([item[0] for item in capr_losses]).mean()
                if capr_losses
                else representation.sum() * 0.0
            )
            # The original eight-anchor SHN-MP objective is not averaged
            # together with the supplement.  CAPR is an extra, bounded term.
            ndp_loss = live_loss + NDP_SHN_CAPR_GAMMA * capr_loss
        else:
            ndp_loss = torch.stack(ndp_values).mean() if ndp_values else representation.sum() * 0.0
    else:
        # B0 must not inspect or backpropagate through any NDP relation.
        ndp_loss = representation.sum() * 0.0
    raw_ndp_loss = ndp_loss
    gate_cosine = float("nan")
    gate_status = "not_applicable"
    gate_evaluated = False
    if arm == "NGATE":
        # DDP ranks must execute the same autograd.grad call sequence. A local
        # active-only gate can otherwise hang when one rank has a relation and
        # the other rank does not. The decision remains local after this
        # synchronization; inactive ranks only participate with their zero NDP
        # graph and never add an NDP term.
        local_active = int(active_ndp > 0)
        global_active_tensor = torch.tensor([local_active], dtype=torch.int64, device=device)
        dist.all_reduce(global_active_tensor, op=dist.ReduceOp.MAX)
        if int(global_active_tensor.item()) > 0:
            keep_ndp, candidate_cosine, candidate_status = gradient_gate_decision(main_total, raw_ndp_loss, feat[0])
            if local_active:
                gate_evaluated = True
                gate_cosine = candidate_cosine
                gate_status = candidate_status
                # The decision is detached. A rejected batch keeps a zero-valued
                # graph so later diagnostic autograd calls remain well-defined,
                # but it adds no NDP gradient or loss to the optimizer objective.
                ndp_loss = raw_ndp_loss if keep_ndp else raw_ndp_loss * 0.0
            else:
                gate_status = "inactive_rank"
    total = compose_total_loss(main_total, ndp_loss, arm, ndp_lambda=ndp_lambda)

    # Batch-local Purity@10 is a training diagnostic, not the official PRCC metric.
    with torch.no_grad():
        pair = representation @ representation.t()
        pair.fill_diagonal_(-float("inf"))
        k = min(10, max(1, pair.shape[0] - 1))
        top = pair.topk(k=k, dim=1).indices
        purity = (pids[top] == pids[:, None]).float().mean()
    # Probe the first few batches as before, but also probe the first batch in
    # which the newly observed GPC-delta protection is actually active.  The
    # risk table is populated only after an optimizer step, so a first-N-only
    # probe can report a zero NDP gradient even when the loss is live later in
    # the epoch.  This is diagnostics only and does not alter the objective.
    need_diag = (
        ndp_enabled
        and (
            stats["batches"] < diagnostic_steps
            or (arm in {"NDP_GPC_DELTA", "NDP_DELTA_FIX", "NDP_AD", "NDP_SHN", "NDP_SHN_MP", "NDP_SHN_MP_ABM", "NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C", "NDP_SHN_BANK", "NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"} and active_ndp > 0 and ndp_loss.requires_grad)
        )
    )
    if need_diag:
        try:
            ndp_feature_grad = torch.autograd.grad(
                ndp_loss, feat[0], retain_graph=True, allow_unused=True
            )[0]
            main_feature_grad = torch.autograd.grad(
                main_total, feat[0], retain_graph=True, allow_unused=True
            )[0]
        except (RuntimeError, ValueError, TypeError):
            ndp_feature_grad = None
            main_feature_grad = None
        ndp_feature_norm, ndp_probe_valid = finite_gradient_norm(ndp_feature_grad)
        main_feature_norm, main_probe_valid = finite_gradient_norm(main_feature_grad)
        ndp_main_grad_cosine, _grad_cosine_valid = finite_gradient_cosine(ndp_feature_grad, main_feature_grad)
    else:
        # Keep aggregate CSV fields finite.  These zeros mean "not probed";
        # the first diagnostic batches carry the actual probe values.
        ndp_feature_norm = 0.0
        main_feature_norm = 0.0
        ndp_main_grad_cosine = 0.0
        ndp_probe_valid = False
        main_probe_valid = False
    components = getattr(loss_fn, "last_components", {})
    stats["batches"] += 1
    stats["active_batches"] += int(active_ndp > 0 and ndp_enabled)
    stats["selected_relations"] += selected_ndp if ndp_enabled else 0
    stats["active_relations"] += active_ndp if ndp_enabled else 0
    stats["triplet_hard_excluded"] += triplet_hard_excluded if arm == "NCOMP" else 0
    stats["triplet_filter_candidates"] += triplet_filter_candidates if arm == "NCOMP" else 0
    stats["current_error_pid"] += current_error_pid if ndp_enabled else 0
    stats["correct_pid"] += correct_pid if ndp_enabled else 0
    stats["margin_sum"] += margin_sum if ndp_enabled else 0.0
    stats["purity_sum"] += float(purity.detach().cpu())
    stats["ndp_loss_sum"] += float(ndp_loss.detach().float().cpu()) if ndp_enabled else 0.0
    stats["ndp_raw_loss_sum"] += float(raw_ndp_loss.detach().float().cpu()) if ndp_enabled else 0.0
    stats["gate_evaluated"] += int(gate_evaluated and arm == "NGATE")
    stats["gate_kept"] += int(gate_status == "kept")
    stats["gate_conflict"] += int(gate_status == "conflict")
    stats["gate_invalid"] += int(gate_evaluated and gate_status not in {"kept", "conflict"})
    # This counter is an explicit sparse-budget diagnostic for NDP-AD.  It is
    # incremented only when a live discrepancy relation was actually added.
    # The selection helper is batch-local and never sees query/gallery data.
    stats.setdefault("ndp_ad_protection_batches", 0)
    if math.isfinite(gate_cosine):
        stats["gate_cosine_sum"] += gate_cosine
    stats["gpc_selected"] += len(selected_gpc)
    gpc_info = dict(gpc_info)
    stats.setdefault("shn_live_relations", 0)
    stats.setdefault("shn_bank_aux_relations", 0)
    stats.setdefault("shn_bank_aux_reliability_sum", 0.0)
    if arm in {"NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"}:
        stats["shn_live_relations"] += len(ndp_shn_losses)
        stats["shn_bank_aux_relations"] += len(ndp_shn_reliable_bank_aux_losses)
        stats["shn_bank_aux_reliability_sum"] += sum(item[4] for item in ndp_shn_reliable_bank_aux_losses)
    if arm == "NDP_SHN_MP_ABM":
        stats.setdefault("abm_candidate_anchors", 0)
        stats.setdefault("abm_selected_pids", 0)
        stats.setdefault("abm_selected_anchors", 0)
        stats.setdefault("abm_boundary_violations", 0)
        stats.setdefault("abm_topk_variation_batches", 0)
        stats["abm_candidate_anchors"] += int(abm_info.get("candidate_anchor_count", 0))
        stats["abm_selected_pids"] += int(abm_info.get("selected_pid_count", 0))
        stats["abm_selected_anchors"] += int(abm_info.get("selected_anchor_count", 0))
        stats["abm_boundary_violations"] += int(abm_info.get("boundary_violation_count", 0))
        signatures = {tuple(item.get("negative_pids", [])) for item in abm_info.get("topk_by_anchor", [])}
        stats["abm_topk_variation_batches"] += int(len(signatures) > 1)
    if arm in {"NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C"}:
        stats.setdefault("capr_candidate_extra_anchors", 0)
        stats.setdefault("capr_candidate_uncovered_pids", 0)
        stats.setdefault("capr_selected_extra_anchors", 0)
        stats.setdefault("capr_selected_extra_pids", 0)
        stats.setdefault("capr_cross_clothes_positives", 0)
        stats.setdefault("capr_total_positives", 0)
        stats.setdefault("capr_risk_weight_sum", 0.0)
        stats.setdefault("capr_risk_weight_count", 0)
        stats.setdefault("capr_triplet_hard_overlap", 0)
        stats.setdefault("capr_triplet_hard_count", 0)
        stats["capr_candidate_extra_anchors"] += int(capr_info.get("candidate_extra_anchor_count", 0))
        stats["capr_candidate_uncovered_pids"] += int(capr_info.get("candidate_uncovered_pid_count", 0))
        stats["capr_selected_extra_anchors"] += int(capr_info.get("selected_extra_anchor_count", 0))
        stats["capr_selected_extra_pids"] += int(capr_info.get("selected_extra_pid_count", 0))
        stats["capr_cross_clothes_positives"] += int(capr_info.get("extra_cross_clothes_positive_count", 0))
        stats["capr_total_positives"] += int(capr_info.get("extra_total_positive_count", 0))
        risk_count = int(capr_info.get("extra_total_positive_count", 0))
        stats["capr_risk_weight_sum"] += float(capr_info.get("extra_risk_weight_mean", 0.0)) * risk_count
        stats["capr_risk_weight_count"] += risk_count
        extra_anchors = [int(value) for value in capr_info.get("selected_extra_anchor_indices", [])]
        for item in capr_info.get("extra_topk_by_anchor", []):
            anchor = int(item.get("anchor", -1))
            hard_pid = int(hard_negative_pids[anchor].item()) if hard_negative_pids is not None and 0 <= anchor < int(hard_negative_pids.numel()) else -1
            if hard_negative_valid is not None and 0 <= anchor < int(hard_negative_valid.numel()) and bool(hard_negative_valid[anchor].item()):
                stats["capr_triplet_hard_count"] += 1
                if hard_pid in {int(value) for value in item.get("negative_pids", [])}:
                    stats["capr_triplet_hard_overlap"] += 1
    capr_arms = {"NDP_SHN_MP_CAPR_A", "NDP_SHN_MP_CAPR_B", "NDP_SHN_MP_CAPR_C"}
    shn_arms = {"NDP_SHN", "NDP_SHN_MP", "NDP_SHN_MP_ABM", *capr_arms, "NDP_SHN_BANK", "NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"}
    gpc_info.update({"ndp_selected_count": selected_ndp if ndp_enabled else 0, "ndp_active_count": active_ndp if ndp_enabled else 0, "ndp_active": bool(active_ndp > 0 and ndp_enabled), "ndp_ad_selected_count": len(ndp_ad_relations) if arm == "NDP_AD" else 0, "ndp_ad_budget_batches": stats.get("ndp_ad_protection_batches", 0), "ndp_shn_anchor_count": len(shn_losses) if arm in shn_arms else 0, "ndp_shn_negative_count": int(sum(item[2] for item in shn_losses)) if arm in shn_arms else 0, "ndp_shn_positive_count": int(sum(item[3] for item in shn_losses)) if arm in shn_arms else 0, "ndp_shn_bank_cross_positive_count": int(sum(max(0, item[3] - 3) for item in ndp_shn_bank_losses)) if arm == "NDP_SHN_BANK" else (len(ndp_shn_reliable_bank_aux_losses) if arm in {"NDP_SHN_RELIABLE", "NDP_SHN_RELIABLE_RISK", "NDP_SHN_RELIABLE_ANCHOR"} else 0), "abm_candidate_anchors": int(abm_info.get("candidate_anchor_count", 0)) if arm == "NDP_SHN_MP_ABM" else 0, "abm_selected_pids": int(abm_info.get("selected_pid_count", 0)) if arm == "NDP_SHN_MP_ABM" else 0, "abm_topk_variation": bool(len({tuple(item.get("negative_pids", [])) for item in abm_info.get("topk_by_anchor", [])}) > 1) if arm == "NDP_SHN_MP_ABM" else False, "capr_candidate_extra_anchors": int(capr_info.get("candidate_extra_anchor_count", 0)) if arm in capr_arms else 0, "capr_selected_extra_anchors": int(capr_info.get("selected_extra_anchor_count", 0)) if arm in capr_arms else 0, "capr_selected_extra_pids": int(capr_info.get("selected_extra_pid_count", 0)) if arm in capr_arms else 0, "capr_cross_clothes_positives": int(capr_info.get("extra_cross_clothes_positive_count", 0)) if arm in capr_arms else 0, "capr_triplet_hard_overlap": int(stats.get("capr_triplet_hard_overlap", 0)) if arm in capr_arms else 0, "current_error_pid": current_error_pid if ndp_enabled else 0, "correct_pid": correct_pid if ndp_enabled else 0, "mean_current_margin": margin_sum / max(correct_pid + current_error_pid, 1), "batch_purity10": float(purity.detach().cpu()), "ndp_gate_evaluated": gate_evaluated, "ndp_gate_cosine": gate_cosine, "ndp_gate_status": gate_status, "triplet_hard_excluded_count": triplet_hard_excluded, "triplet_filter_candidate_count": triplet_filter_candidates})
    return {"total": total, "base_total": base_total, "ce": components.get("ce", torch.tensor(float("nan"), device=device)), "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)), "bio": components.get("bio", torch.tensor(float("nan"), device=device)), "gpc": gpc_loss, "ndp": ndp_loss, "ndp_raw": raw_ndp_loss, "main_total": main_total, "ndp_lambda": (float(ndp_lambda) if ndp_enabled and ndp_lambda is not None else (LAMBDA_NDP if ndp_enabled else 0.0)), "relation_info": gpc_info, "gpc_delta_probes": gpc_delta_probes, "delta_fix_probes": delta_fix_probes, "delta_fix_keys": delta_fix_keys if arm == "NDP_DELTA_FIX" else [], "ndp_feature_grad_norm": ndp_feature_norm, "main_feature_grad_norm": main_feature_norm, "ndp_main_grad_cosine": ndp_main_grad_cosine, "ndp_feature_grad_valid": ndp_probe_valid, "main_feature_grad_valid": main_probe_valid, "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(), "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1]}


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
    if args.arm in {"B0", "NONLINE", "NLIST"}:
        # B0 is the frozen Bio+GPC control.  NONLINE also does not load a
        # fixed historical graph; it constructs its graph from each current
        # epoch bank below.
        reference_rows = {}
        reference_features = torch.empty((0, 1024), dtype=torch.float32, device=device)
        reference_meta = {
            "reference_source": "not used by B0 control" if args.arm == "B0" else "current_epoch_online_train_bank",
            "reference_update": None,
            "epsilon": None if args.arm == "B0" else 0.02,
            "future_updates_read_for_selection": [],
        }
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
        ndp_version = {"N5": "NDP-V5", "NF0": "NDP-Floor0", "NRG": "NDP-RiskGate", "NCAP": "NDP-Cap", "NDYN": "NDP-Dynamic", "NDYN_TOP5": "NDP-Dynamic-TOP5", "NGATE": "NDP-GradientGate", "NFRESH": "NDP-Fresh", "NCOMP": "NDP-Triplet-Complement-Fresh", "NONLINE": "NDP-Online", "NLIST": "NDP-Listwise", "NDP_LATE": "NDP-Late"}.get(args.arm)
        ndp_mode = {"N5": "reference_margin_target", "NF0": "floor_zero_target", "NRG": "historically_safe_current_near_boundary", "NCAP": "capped_reference_target", "NDYN": "dynamic_margin_degradation", "NDYN_TOP5": "top5_risk_small_recovery", "NGATE": "top5_risk_small_recovery_with_main_ndp_gradient_gate", "NFRESH": "top5_risk_small_recovery_with_epoch_fresh_endpoints", "NCOMP": "fresh_top5_excluding_current_triplet_hard_negative_pid", "NONLINE": "epoch_online_dangerous_neighbors_fixed_margin", "NLIST": "epoch_online_local_listwise_rank_tau_0.1", "NDP_LATE": "teacher_gated_student_space_delayed_confusion"}.get(args.arm)
        write_json(run_dir / "training_config.json", {"run_id": args.run_id, "arm": args.arm, "seed": args.seed, "initialization": "warm-start from B0 checkpoint with complete per-rank model/optimizer/scheduler/scaler/RNG state" if args.resume_checkpoint else "PRCC_COMMON_INIT fresh optimizer; no historical PRCC E60 state exists", "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None, "resume_epoch": start_epoch, "resume_global_step": start_global_step, "continuation_updates": int(args.steps) if args.steps else int(args.epochs) * EPOCH_STEPS, "raw_epoch_mode": args.steps is None, "evaluation_segment_epochs": sorted(int(x) for x in args.eval_epochs.split(",") if x.strip()), "sampler_policy": "new explicit continuation segment; sampler.set_epoch(absolute_epoch) before every epoch", "common_init": audit, "config": cfg.dump(), "dataset": {"train_images": len(dataset.train), "train_pids": int(dataset.num_train_pids), "train_clothes": int(dataset.num_train_clothes), "steps_per_epoch": len(train_loader), "batch_size": BATCH_SIZE, "num_instances": NUM_INSTANCES}, "base_loss": "CE + Triplet + clipBio + 0.1 * original GPC-v1 matched-budget relation loss", "gpc_source_sha256": sha256_file(ENGINE / "scripts/gpc_v2_train.py"), "gpc_pair_phase": gpc_phase, "gpc_selector": "gpc_v2_train.selected_v2 with arm B0; unchanged GPC matched-budget candidate/positive logic", "ndp": {"version": ndp_version, "mode": ndp_mode, "lambda": LAMBDA_NDP, "epsilon": float(args.ndp_epsilon), "per_negative_pid": True, "reference_meta": reference_meta, "reference_stop_gradient": True, "current_anchor_grad_path": True, "risk_gate": "reference_pid_margin >= epsilon and 0 < current_pid_margin < epsilon" if args.arm == "NRG" else None}, "future_updates_read_for_selection": [], "query_gallery_used_for_training_selection": False, "probe_parameter": probe_name})
    if rank == 0 and args.arm in {"NF0", "NRG", "NCAP", "NDYN", "NDYN_TOP5", "NGATE", "NFRESH", "NCOMP", "NONLINE", "NLIST", "NDP_LATE"}:
        variant = {
            "NF0": ("NDP-Floor0", "floor_zero_target"),
            "NRG": ("NDP-RiskGate", "historically_safe_current_near_boundary"),
            "NCAP": ("NDP-Cap", "capped_reference_target"),
            "NDYN": ("NDP-Dynamic", "dynamic_margin_degradation"),
            "NDYN_TOP5": ("NDP-Dynamic-TOP5", "top5_risk_small_recovery"),
            "NGATE": ("NDP-GradientGate", "top5_risk_small_recovery_with_main_ndp_gradient_gate"),
            "NFRESH": ("NDP-Fresh", "top5_risk_small_recovery_with_epoch_fresh_endpoints"),
            "NCOMP": ("NDP-Triplet-Complement-Fresh", "fresh_top5_excluding_current_triplet_hard_negative_pid"),
            "NONLINE": ("NDP-Online", "epoch_online_dangerous_neighbors_fixed_margin"),
            "NLIST": ("NDP-Listwise", "epoch_online_local_listwise_rank_tau_0.1"),
            "NDP_LATE": ("NDP-Late", "teacher_gated_student_space_delayed_confusion"),
        }[args.arm]
        write_json(run_dir / "ndp_variant.json", {"version": variant[0], "mode": variant[1], "lambda": LAMBDA_NDP, "epsilon": float(args.ndp_epsilon), "reference_correct_only": True})
    if rank == 0 and args.resume_model:
        write_json(run_dir / "model_only_warm_start.json", {"status": "MODEL_ONLY_WARM_START", "model": str(args.resume_model), "model_sha256": sha256_file(args.resume_model), "epoch": int(args.resume_epoch), "global_step": int(args.resume_global_step), "optimizer_scheduler_scaler_rng": "fresh; complete rank state was unavailable"})
        training_config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
        training_config["initialization"] = "model-only B0 warm start + fresh optimizer/scheduler/scaler/RNG"
        training_config["resume_model"] = str(args.resume_model)
        training_config["ndp"]["version"] = {"NF0": "NDP-Floor0", "NRG": "NDP-RiskGate", "NCAP": "NDP-Cap", "NDYN": "NDP-Dynamic", "NDYN_TOP5": "NDP-Dynamic-TOP5", "NFRESH": "NDP-Fresh", "NCOMP": "NDP-Triplet-Complement-Fresh", "NONLINE": "NDP-Online", "NLIST": "NDP-Listwise", "NDP_LATE": "NDP-Late"}.get(args.arm, "NDP-V5")
        training_config["ndp"]["mode"] = {"NF0": "floor_zero_target", "NRG": "historically_safe_current_near_boundary", "NCAP": "capped_reference_target", "NDYN": "dynamic_margin_degradation", "NDYN_TOP5": "top5_risk_small_recovery", "NGATE": "top5_risk_small_recovery_with_main_ndp_gradient_gate", "NFRESH": "top5_risk_small_recovery_with_epoch_fresh_endpoints", "NCOMP": "fresh_top5_excluding_current_triplet_hard_negative_pid", "NONLINE": "epoch_online_dangerous_neighbors_fixed_margin", "NLIST": "epoch_online_local_listwise_rank_tau_0.1", "NDP_LATE": "teacher_gated_student_space_delayed_confusion"}.get(args.arm, "reference_margin_target")
        write_json(run_dir / "training_config.json", training_config)
    fields = ["batch_index", "epoch", "iteration", "global_step", "effective_updates", "total", "base_total", "ce", "triplet", "bio", "gpc", "ndp", "ndp_raw", "lambda_ndp", "gpc_selected", "ndp_selected", "ndp_active", "triplet_hard_excluded", "triplet_filter_candidates", "ndp_gate_evaluated", "ndp_gate_cosine", "ndp_gate_status", "current_error_pid", "correct_pid", "batch_purity10", "ndp_feature_grad_norm", "main_feature_grad_norm", "total_model_grad_norm", "ndp_probe_param_grad_norm", "lr", "scale_before", "scale_after", "found_inf", "step_executed", "fresh_bank_epoch", "fresh_bank_sha256", "fresh_feature_tensor_sha256", "fresh_valid_pid_count", "fresh_old_margin_mean", "fresh_margin_mean", "fresh_margin_delta_mean", "fresh_margin_delta_std", "fresh_margin_delta_min", "fresh_margin_delta_max", "online_neighbor_count", "online_valid_anchor_count", "online_neighbor_change_rate"]
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
    stats = {"batches": 0, "active_batches": 0, "selected_relations": 0, "active_relations": 0, "triplet_hard_excluded": 0, "triplet_filter_candidates": 0, "current_error_pid": 0, "correct_pid": 0, "margin_sum": 0.0, "purity_sum": 0.0, "ndp_loss_sum": 0.0, "ndp_raw_loss_sum": 0.0, "gate_evaluated": 0, "gate_kept": 0, "gate_conflict": 0, "gate_invalid": 0, "gate_cosine_sum": 0.0, "gpc_selected": 0}
    saved_nodes = []
    global_step = start_global_step
    effective_updates = 0
    batch_index = 0
    log = logger(run_dir) if rank == 0 else None
    epoch_sums = defaultdict(float)
    epoch_stats = defaultdict(lambda: {"batches": 0, "active_batches": 0, "selected_relations": 0, "active_relations": 0, "triplet_hard_excluded": 0, "triplet_filter_candidates": 0, "current_error_pid": 0, "correct_pid": 0, "margin_sum": 0.0, "purity_sum": 0.0, "ndp_loss_sum": 0.0, "ndp_raw_loss_sum": 0.0, "gate_evaluated": 0, "gate_kept": 0, "gate_conflict": 0, "gate_invalid": 0, "gate_cosine_sum": 0.0, "gpc_selected": 0})
    fresh_epoch_info: dict[str, Any] = {}
    online_previous_signature: list[tuple[int, ...]] | None = None
    started = time.time()
    try:
        epoch = start_epoch
        segment_epoch = 0
        while (segment_epoch < int(args.epochs)) if raw_epoch_mode else (effective_updates < steps_to_run):
            segment_epoch += 1
            epoch = start_epoch + segment_epoch
            if args.arm in {"NFRESH", "NCOMP", "NONLINE", "NLIST"}:
                # The graph, selected PID mask, and t1 target margins remain
                # fixed for NFRESH/NCOMP. NONLINE instead regenerates its
                # dangerous-neighbor graph from this epoch's bank.
                fresh_rng_before = rng_state_digest(base.rng_state(device))
                fresh_features = extract_train_feature_bank(model, cfg, dataset, device)
                fresh_rng_after = rng_state_digest(base.rng_state(device))
                fresh_rng_preserved = fresh_rng_before == fresh_rng_after
                if not fresh_rng_preserved:
                    raise RuntimeError(
                        f"fresh-bank RNG changed training state at epoch={epoch}: "
                        f"before={fresh_rng_before} after={fresh_rng_after}"
                    )
                if args.arm in {"NONLINE", "NLIST"}:
                    train_pid_tensor = torch.as_tensor(
                        [int(item["pid"]) for item in dataset.train], dtype=torch.long, device=device
                    )
                    reference_rows, reference_features, online_summary, online_previous_signature = build_online_neighbor_graph(
                        fresh_features,
                        train_pid_tensor,
                        device,
                        max_negative_pids=int(args.online_max_negative_pids),
                        block_size=int(args.online_block_size),
                        previous_signature=online_previous_signature,
                    )
                    fresh_summary = {
                        "fresh_valid_pid_count": int(online_summary["online_relation_count"]),
                        "fixed_selected_pid_count": int(online_summary["online_relation_count"]),
                        "old_margin_mean": 0.0,
                        "fresh_margin_mean": 0.0,
                        "margin_delta_mean": 0.0,
                        "margin_delta_std": 0.0,
                        "margin_delta_min": 0.0,
                        "margin_delta_max": 0.0,
                        **online_summary,
                    }
                    # Store a compact, auditable graph.  The dense endpoint
                    # bank is stored separately as the fresh NPZ.
                    online_graph_path = run_dir / f"online_neighbor_graph_epoch{segment_epoch:02d}.json"
                    if rank == 0:
                        write_json(online_graph_path, {
                            "version": "ndp-online-prcc-train-only",
                            "epoch": int(epoch),
                            "train_only": True,
                            "query_gallery_used": False,
                            "max_negative_pids": int(args.online_max_negative_pids),
                            "block_size": int(args.online_block_size),
                            "summary": online_summary,
                            "relations": [
                                {
                                    "anchor_image_index": int(anchor_id),
                                    "anchor_pid": int(rows[0]["anchor_pid"]),
                                    "negative_indices": [int(value) for value in rows[0]["negative_local"]],
                                    "negative_pid": [int(value) for value in rows[0]["negative_pid"]],
                                    "positive_reference_index": int(rows[0]["positive_local"][0]),
                                }
                                for anchor_id, rows in sorted(reference_rows.items())
                            ],
                        })
                else:
                    fresh_summary = summarize_fresh_bank(reference_rows, fresh_features)
                fresh_cpu = fresh_features.detach().cpu().numpy().astype(np.float32, copy=True)
                fresh_feature_hash = hashlib.sha256(fresh_cpu.tobytes()).hexdigest()
                image_index = np.arange(len(dataset.train), dtype=np.int32)
                image_index_hash = hashlib.sha256(image_index.tobytes()).hexdigest()
                fresh_path = run_dir / f"fresh_bank_epoch{segment_epoch:02d}.npz"
                fresh_manifest_path = run_dir / f"fresh_bank_epoch{segment_epoch:02d}.json"
                if rank == 0:
                    np.savez_compressed(fresh_path, features=fresh_cpu, image_index=image_index, epoch=np.asarray(epoch, dtype=np.int32))
                    fresh_epoch_info = {
                        "fresh_bank_epoch": int(epoch),
                        "fresh_bank_path": str(fresh_path),
                        "fresh_bank_sha256": sha256_file(fresh_path),
                        "fresh_feature_tensor_sha256": fresh_feature_hash,
                        "fresh_image_index_sha256": image_index_hash,
                        "rng_state_before_sha256": fresh_rng_before,
                        "rng_state_after_sha256": fresh_rng_after,
                        "rng_state_preserved": fresh_rng_preserved,
                        "fresh_bank_shape": list(fresh_cpu.shape),
                        **fresh_summary,
                    }
                    write_json(fresh_manifest_path, fresh_epoch_info)
                else:
                    fresh_epoch_info = {
                        "fresh_bank_epoch": int(epoch),
                        "fresh_feature_tensor_sha256": fresh_feature_hash,
                        "fresh_image_index_sha256": image_index_hash,
                        "rng_state_before_sha256": fresh_rng_before,
                        "rng_state_after_sha256": fresh_rng_after,
                        "rng_state_preserved": fresh_rng_preserved,
                        "fresh_bank_shape": list(fresh_cpu.shape),
                        **fresh_summary,
                    }
                dist.barrier()
                # NONLINE uses the combined [fresh samples, per-anchor
                # positive references] tensor returned by the online graph
                # builder.  NFRESH/NCOMP use the fresh sample bank directly.
                if args.arm not in {"NONLINE", "NLIST"}:
                    reference_features = fresh_features
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
                if stats["batches"] <= args.diagnostic_steps and args.arm in {"N5", "NF0", "NRG", "NCAP", "NDYN", "NDYN_TOP5", "NGATE", "NFRESH", "NCOMP", "NONLINE", "NLIST"}:
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
                    step_row = {"batch_index": batch_index, "epoch": epoch, "iteration": iteration, "global_step": global_step, "effective_updates": effective_updates, "total": scalar(result["total"]), "base_total": scalar(result["base_total"]), "ce": scalar(result["ce"]), "triplet": scalar(result["triplet"]), "bio": scalar(result["bio"]), "gpc": scalar(result["gpc"]), "ndp": scalar(result["ndp"]), "ndp_raw": scalar(result["ndp_raw"]), "lambda_ndp": result["ndp_lambda"], "gpc_selected": len(info.get("keys", [])), "ndp_selected": info["ndp_selected_count"], "ndp_active": info["ndp_active_count"], "triplet_hard_excluded": info["triplet_hard_excluded_count"], "triplet_filter_candidates": info["triplet_filter_candidate_count"], "ndp_gate_evaluated": int(info["ndp_gate_evaluated"]), "ndp_gate_cosine": info["ndp_gate_cosine"], "ndp_gate_status": info["ndp_gate_status"], "current_error_pid": info["current_error_pid"], "correct_pid": info["correct_pid"], "batch_purity10": info["batch_purity10"], "ndp_feature_grad_norm": result["ndp_feature_grad_norm"], "main_feature_grad_norm": result["main_feature_grad_norm"], "total_model_grad_norm": total_norm, "ndp_probe_param_grad_norm": float(torch.linalg.vector_norm(diag_probe.float()).detach().cpu()) if diag_probe is not None else float("nan"), "lr": float(optimizer.param_groups[0]["lr"]), "scale_before": scale_before, "scale_after": scale_after, "found_inf": found_inf, "step_executed": step_executed}
                    if args.arm in {"NFRESH", "NCOMP", "NONLINE", "NLIST"}:
                        step_row.update({"fresh_bank_epoch": fresh_epoch_info.get("fresh_bank_epoch", epoch), "fresh_bank_sha256": fresh_epoch_info.get("fresh_bank_sha256", ""), "fresh_feature_tensor_sha256": fresh_epoch_info.get("fresh_feature_tensor_sha256", ""), "fresh_valid_pid_count": fresh_epoch_info.get("fresh_valid_pid_count", 0), "fresh_old_margin_mean": fresh_epoch_info.get("old_margin_mean", 0.0), "fresh_margin_mean": fresh_epoch_info.get("fresh_margin_mean", 0.0), "fresh_margin_delta_mean": fresh_epoch_info.get("margin_delta_mean", 0.0), "fresh_margin_delta_std": fresh_epoch_info.get("margin_delta_std", 0.0), "fresh_margin_delta_min": fresh_epoch_info.get("margin_delta_min", 0.0), "fresh_margin_delta_max": fresh_epoch_info.get("margin_delta_max", 0.0)})
                    else:
                        step_row.update({"fresh_bank_epoch": "", "fresh_bank_sha256": "", "fresh_feature_tensor_sha256": "", "fresh_valid_pid_count": 0, "fresh_old_margin_mean": 0.0, "fresh_margin_mean": 0.0, "fresh_margin_delta_mean": 0.0, "fresh_margin_delta_std": 0.0, "fresh_margin_delta_min": 0.0, "fresh_margin_delta_max": 0.0})
                    step_row.update({"online_neighbor_count": fresh_epoch_info.get("online_relation_count", 0), "online_valid_anchor_count": fresh_epoch_info.get("online_valid_anchor_count", 0), "online_neighbor_change_rate": fresh_epoch_info.get("online_neighbor_change_rate", None)})
                    mw.writerow(step_row); mf.flush()
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
                counts = torch.tensor([stats["active_batches"], stats["selected_relations"], stats["active_relations"], stats["triplet_hard_excluded"], stats["triplet_filter_candidates"], stats["current_error_pid"], stats["correct_pid"], stats["margin_sum"], stats["purity_sum"], stats["ndp_loss_sum"], stats["gpc_selected"], stats["batches"], stats["ndp_raw_loss_sum"], stats["gate_evaluated"], stats["gate_kept"], stats["gate_conflict"], stats["gate_invalid"], stats["gate_cosine_sum"]], dtype=torch.float64, device=device)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                if rank == 0:
                    row = {"arm": args.arm, "seed": args.seed, "segment_epoch": segment_epoch, "epoch": epoch, "raw_batches": batch_index, "effective_updates": effective_updates, "train_total": float(values[0].cpu()), "ce": float(values[1].cpu()), "triplet": float(values[2].cpu()), "bio": float(values[3].cpu()), "gpc": float(values[4].cpu()), "ndp": float(values[5].cpu()), "id_accuracy": float(values[6].cpu()), "bio_accuracy": float(values[7].cpu()), "ndp_active_batches": int(counts[0].item()), "ndp_selected_relations": int(counts[1].item()), "ndp_active_relations": int(counts[2].item()), "triplet_hard_excluded_relations": int(counts[3].item()), "triplet_filter_candidates": int(counts[4].item()), "triplet_hard_exclusion_rate": float(counts[3].item() / max(1.0, counts[4].item())), "ndp_active_rate": float(counts[0].item() / max(1.0, counts[11].item())), "current_error_pid": int(counts[5].item()), "correct_pid": int(counts[6].item()), "mean_current_margin": float(counts[7].item() / max(1.0, counts[5].item() + counts[6].item())), "purity10_batch_mean": float(counts[8].item() / max(1.0, counts[11].item())), "ndp_mean_loss": float(counts[9].item() / max(1.0, counts[11].item())), "ndp_raw_mean_loss": float(counts[12].item() / max(1.0, counts[11].item())), "ndp_gate_evaluated": int(counts[13].item()), "ndp_gate_kept": int(counts[14].item()), "ndp_gate_conflict": int(counts[15].item()), "ndp_gate_invalid": int(counts[16].item()), "ndp_gate_cosine_mean": float(counts[17].item() / max(1.0, counts[13].item())), "ndp_gate_negative_rate": float(counts[15].item() / max(1.0, counts[13].item())), "ndp_gate_keep_rate": float(counts[14].item() / max(1.0, counts[13].item())), "gpc_selected_relations": int(counts[10].item()), "lr": float(optimizer.param_groups[0]["lr"]), "cc_rank1": eval_metrics.get("cc_rank1"), "cc_mAP": eval_metrics.get("cc_mAP"), "general_rank1": eval_metrics.get("general_rank1"), "general_mAP": eval_metrics.get("general_mAP"), "mINP": None, "mINP_status": "not exposed by repository R1_mAP_eval"}
                    if args.arm in {"NFRESH", "NCOMP", "NONLINE", "NLIST"}:
                        row.update({"fresh_bank_epoch": fresh_epoch_info.get("fresh_bank_epoch", epoch), "fresh_bank_sha256": fresh_epoch_info.get("fresh_bank_sha256", ""), "fresh_feature_tensor_sha256": fresh_epoch_info.get("fresh_feature_tensor_sha256", ""), "fresh_valid_pid_count": fresh_epoch_info.get("fresh_valid_pid_count", 0), "fresh_old_margin_mean": fresh_epoch_info.get("old_margin_mean", 0.0), "fresh_margin_mean": fresh_epoch_info.get("fresh_margin_mean", 0.0), "fresh_margin_delta_mean": fresh_epoch_info.get("margin_delta_mean", 0.0), "fresh_margin_delta_std": fresh_epoch_info.get("margin_delta_std", 0.0), "fresh_margin_delta_min": fresh_epoch_info.get("margin_delta_min", 0.0), "fresh_margin_delta_max": fresh_epoch_info.get("margin_delta_max", 0.0)})
                    row.update({"online_neighbor_count": fresh_epoch_info.get("online_relation_count", 0), "online_valid_anchor_count": fresh_epoch_info.get("online_valid_anchor_count", 0), "online_neighbor_change_rate": fresh_epoch_info.get("online_neighbor_change_rate", None)})
                    with (run_dir / "metrics_per_epoch.csv").open("a", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader() if stream.tell() == 0 else None; writer.writerow(row)
                    write_json(run_dir / "run_progress.json", {"last_epoch": epoch, "effective_updates": effective_updates, "batch_index": batch_index, "saved_nodes": saved_nodes})
                # The next epoch must report its own activation and local
                # margin statistics, not a cumulative sum from epoch 1.
                stats = {"batches": 0, "active_batches": 0, "selected_relations": 0, "active_relations": 0, "triplet_hard_excluded": 0, "triplet_filter_candidates": 0, "current_error_pid": 0, "correct_pid": 0, "margin_sum": 0.0, "purity_sum": 0.0, "ndp_loss_sum": 0.0, "ndp_raw_loss_sum": 0.0, "gate_evaluated": 0, "gate_kept": 0, "gate_conflict": 0, "gate_invalid": 0, "gate_cosine_sum": 0.0, "gpc_selected": 0}
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
    parser.add_argument("--seed", type=int, choices=(1234, 3456, 4321), default=1234)
    parser.add_argument("--arm", choices=("B0", "N5", "NF0", "NRG", "NCAP", "NDYN", "NDYN_TOP5", "NGATE", "NFRESH", "NCOMP", "NONLINE", "NLIST", "NDP_LATE", "NDP_GPC", "NDP_GPC_IMPACT"), default="N5")
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
    parser.add_argument("--online-max-negative-pids", type=int, default=3, help="NONLINE distinct dangerous negative PIDs per anchor")
    parser.add_argument("--online-block-size", type=int, default=512, help="NONLINE GPU similarity query block size")
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
