#!/usr/local/bin/python3.10
"""Formal 60-epoch Bio+GPC+NDP-Listwise runner.

This entry point intentionally does not call the historical recovery writer.
It uses the formal model/loss/sampler/optimizer path, adds only the already
implemented epoch-local NLIST graph, and writes no per-rank checkpoint.  The
only resumable artifact is the atomic, single-file ``resume_latest.pth``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
FORMAL_SOURCE = ROOT / "github_uploads/test2_20261016_141306/scripts/train_final_gpc.py"
WORLD_SIZE = 2
SEED = 1234
EPOCHS = 60
WARMUP_EPOCHS = 20
LAMBDA_NDP = 0.05
LAMBDA_NDP_SAFE = 0.005
NDP_SAFE_MARGIN = 0.05
NDP_SAFE_WARMUP_EPOCHS = 10
NDP_SAFE_GRAPH_UPDATE_EVERY = 5
LAMBDA_NDP_SELECT = 0.001
NDP_SELECT_START_EPOCH = 3
NDP_SELECT_WARMUP_EPOCHS = 2
NDP_SELECT_GRAPH_UPDATE_EVERY = 5
LAMBDA_NDP_V2 = 0.001
NDP_V2_START_EPOCH = 3
NDP_V2_WARMUP_EPOCHS = 2
NDP_V2_GRAPH_UPDATE_EVERY = 5
LAMBDA_NDP_TEACH = 0.001
NTEACH_START_EPOCH = 2
TAU = 0.1
GPC_CONTEXT = "phase7i_relation_specificity"
EXPECTED_DEVICE_NAME = "PPU-ZW810E"

sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

# Preload the deployed engine modules before loading the historical formal
# wrapper.  The wrapper then reuses these module objects instead of importing
# an old model implementation from the archive snapshot.
from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from model.build import factory as MODEL_FACTORY  # noqa: E402
from processor.processor import test as processor_test  # noqa: E402
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC  # noqa: E402


def preload_formal_gpc_loss_module() -> None:
    """Provide the formal wrapper's audit-only GPC helper without changing loss."""
    module_name = "loss.gpc_relation"
    if module_name in sys.modules:
        return
    source = ROOT / "github_uploads/test2_20261016_141306/loss/gpc_relation.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load formal GPC helper: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def load_formal_module():
    if not FORMAL_SOURCE.is_file():
        raise FileNotFoundError(FORMAL_SOURCE)
    spec = importlib.util.spec_from_file_location("formal_train_final_gpc", FORMAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load formal source: {FORMAL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    preload_formal_gpc_loss_module()
    spec.loader.exec_module(module)
    return module


FORMAL = load_formal_module()
# The archived formal module computes its fixed LTCC candidate-table path from
# its snapshot root.  Point only that path resolver at the deployed engine;
# the candidate-selection code and hash namespace remain unchanged.
FORMAL.ROOT = ENGINE
import ndp_v5_prcc_runner as NDP  # noqa: E402
import ndp_safe as NDP_SAFE  # noqa: E402


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def beijing_stamp() -> str:
    return dt.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    hashes = {}
    for key, value in sorted(state.items()):
        hashes[key] = sha256_bytes(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return sha256_bytes(json.dumps(hashes, sort_keys=True).encode("utf-8"))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runtime_gate() -> tuple[int, torch.device]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("NLIST full60 requires exactly two visible PPU/GPU devices")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if any(name != EXPECTED_DEVICE_NAME for name in names):
        raise RuntimeError(f"unexpected devices: {names}")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != WORLD_SIZE or rank not in (0, 1):
        raise RuntimeError(f"torchrun requires WORLD_SIZE=2 and LOCAL_RANK in {{0,1}}, got {world}/{rank}")
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, torch.device(f"cuda:{rank}")


def cleanup_dist() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cfg_for(dataset_name: str, output_dir: Path):
    # The formal source is the released 60-epoch protocol.  Its imported
    # build_dataset/build_model/loss objects are the deployed /data engine
    # modules preloaded above.
    cfg = FORMAL.cfg_for(dataset_name, output_dir)
    cfg.defrost()
    # The mounted train-only caption assets live under /data/data on this
    # server; the historical formal wrapper used an older /home/data mount.
    cfg.DATA.CAPTION_DIR = f"/data/data/DIFFER_TextCaption/{'LTCC_ReID' if dataset_name == 'ltcc' else 'PRCC'}"
    cfg.SOLVER.SEED = SEED
    cfg.SOLVER.MAX_EPOCHS = EPOCHS
    cfg.SOLVER.WARMUP_EPOCHS = WARMUP_EPOCHS
    cfg.freeze()
    return cfg


def spec_for(dataset_name: str) -> dict[str, Any]:
    return FORMAL.SPEC[dataset_name]


def init_paths(output_root: Path, dataset_name: str) -> tuple[Path, Path]:
    directory = output_root / dataset_name.upper() / "common_init"
    return directory / f"{dataset_name.upper()}_EVA02_BIO_INIT_SEED{SEED}.pth", directory / "init_manifest.json"


FORMAL_COMMON_INIT = {
    "ltcc": ENGINE / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth",
    "prcc": ENGINE / "outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT.pth",
}
FORMAL_COMMON_META = {
    "ltcc": ENGINE / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT_HASHES.json",
    "prcc": ENGINE / "outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT_HASHES.json",
}


def assert_no_rank_state_files(output_root: Path) -> None:
    forbidden = []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if (name.startswith("rank") and "state.pt" in name) or (name.startswith("rank") and "state.pth" in name) or (name.startswith("checkpoint_e") and "_rank" in name):
            forbidden.append(str(path))
    if forbidden:
        raise RuntimeError("forbidden per-rank checkpoint detected: " + ", ".join(forbidden[:10]))


def prepare_initialization(output_root: Path, dataset_name: str) -> dict[str, Any]:
    path, manifest_path = init_paths(output_root, dataset_name)
    if path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(path) != manifest["file_sha256"]:
            raise RuntimeError(f"initialization SHA256 mismatch: {path}")
        return manifest
    if path.exists() or manifest_path.exists():
        raise RuntimeError(f"partial initialization artifact: {path}")
    source = FORMAL_COMMON_INIT[dataset_name]
    source_meta_path = FORMAL_COMMON_META[dataset_name]
    if not source.is_file() or not source_meta_path.is_file():
        raise FileNotFoundError(f"missing historical formal common initialization: {source}")
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    state = torch.load(source, map_location="cpu", weights_only=False)
    if sha256_file(source) != source_meta.get("state_sha256"):
        raise RuntimeError(f"formal common-init file hash mismatch: {source}")
    expected_state_hash = source_meta.get("state_hash")
    if expected_state_hash is not None and state_hash(state) != expected_state_hash:
        raise RuntimeError(f"formal common-init tensor hash mismatch: {source}")
    if any("reverse" in key.lower() for key in state):
        raise RuntimeError("formal Bio initialization unexpectedly contains reverse parameters")
    cfg = cfg_for(dataset_name, output_root / dataset_name.upper() / "common_init")
    dataset = build_dataset(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    manifest = {
        "dataset": dataset_name,
        "seed": SEED,
        "source": "historical formal Bio common initialization",
        "source_path": str(source),
        "source_file_sha256": sha256_file(source),
        "source_meta_path": str(source_meta_path),
        "source_meta_sha256": sha256_file(source_meta_path),
        "formal_source": str(FORMAL_SOURCE),
        "formal_source_sha256": sha256_file(FORMAL_SOURCE),
        "state_hash": state_hash(state),
        "file_sha256": sha256_file(path),
        "state_key_count": len(state),
        "state_numel": int(sum(v.numel() for v in state.values())),
        "train_count": len(dataset.train),
        "query_gallery_used": False,
        "created_utc": utc_now(),
    }
    write_json(manifest_path, manifest)
    del dataset, state
    return manifest


def load_initialization(output_root: Path, dataset_name: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path, manifest_path = init_paths(output_root, dataset_name)
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"missing formal initialization; run --mode prepare for {dataset_name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["file_sha256"]:
        raise RuntimeError(f"initialization hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state_hash(state) != manifest["state_hash"]:
        raise RuntimeError(f"initialization tensor hash mismatch: {path}")
    return state, manifest


def build_model_for_evaluation(cfg, dataset):
    """Build the deployed architecture without downloading pretrained weights.

    The deployed ``model.build_model`` wrapper hard-codes ``pretrained=True``.
    That is appropriate for creating the formal initialization, but wrong when
    evaluating a fully materialized local checkpoint: evaluation must not make
    an unrelated HuggingFace/network request.  Construct the same factory
    entry with ``pretrained=False`` and then load the final state dict.
    """
    model_type = cfg.MODEL.TYPE
    if model_type in ["eva02"]:
        # This formal run uses MODEL.TYPE=eva02_cloth.  Keep the guard explicit
        # rather than silently constructing a different transformer variant.
        raise RuntimeError("evaluation without download is not defined for MODEL.TYPE=eva02")
    if model_type == "" or model_type == "eva02_cloth":
        constructor = MODEL_FACTORY[cfg.MODEL.NAME]
    else:
        constructor = MODEL_FACTORY[f"{cfg.MODEL.NAME}_{model_type}"]
    return constructor(
        pretrained=False,
        config=cfg,
        num_classes=dataset.num_train_pids,
        camera_num=dataset.num_camera,
    )


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def capture_rank_state(optimizer, scheduler, scaler, device: torch.device) -> dict[str, Any]:
    # These are CPU objects so one shared resume file can contain both rank
    # optimizer/scaler states and both rank RNG states without per-rank files.
    return {
        "optimizer": FORMAL.to_cpu(optimizer.state_dict()),
        "scheduler": FORMAL.to_cpu(scheduler.state_dict()),
        "scaler": FORMAL.to_cpu(scaler.state_dict()),
        "rng": FORMAL.to_cpu(NDP.base.rng_state(device)),
    }


def save_resume(output_dir: Path, raw, optimizer, scheduler, scaler, epoch: int, global_step: int, device: torch.device, dataset_name: str) -> None:
    local = capture_rank_state(optimizer, scheduler, scaler, device)
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local)
    dist.barrier()
    if int(os.environ.get("RANK", "0")) == 0:
        payload = {
            "version": "ndp-nlist-full60-single-file-resume-v1",
            "dataset": dataset_name,
            "seed": SEED,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model": FORMAL.cpu_state_dict(raw),
            "rank_states": gathered,
            "saved_utc": utc_now(),
            "gpc_context": GPC_CONTEXT,
            "no_per_rank_checkpoint": True,
        }
        atomic_torch_save(payload, output_dir / "resume_latest.pth")
        write_json(output_dir / "resume_latest.json", {
            "dataset": dataset_name,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "file": str(output_dir / "resume_latest.pth"),
            "file_sha256": sha256_file(output_dir / "resume_latest.pth"),
            "rank_state_count": len(gathered),
            "saved_utc": utc_now(),
        })
    dist.barrier()


def load_resume(path: Path, raw, optimizer, scheduler, scaler, rank: int, device: torch.device) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("dataset") is None or payload.get("seed") != SEED or len(payload.get("rank_states", [])) != WORLD_SIZE:
        raise RuntimeError(f"invalid single-file resume payload: {path}")
    raw.load_state_dict(payload["model"], strict=True)
    local = payload["rank_states"][rank]
    optimizer.load_state_dict(local["optimizer"])
    scheduler.load_state_dict(local["scheduler"])
    scaler.load_state_dict(local["scaler"])
    NDP.base.restore_rng(local["rng"], device)
    return int(payload["epoch"]), int(payload["global_step"])


def logger_for(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"NDP_NLIST_FULL60.{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def identity_feature_bank(model, cfg, dataset, device) -> torch.Tensor:
    # NDP's validated helper preserves Python/NumPy/CPU/PPU RNG and uses
    # deterministic train-only evaluation preprocessing.
    return NDP.extract_train_feature_bank(model, cfg, dataset, device)


def make_online_graph(model, cfg, dataset, device, previous_signature, arm):
    bank = identity_feature_bank(model, cfg, dataset, device)
    if arm in {"NDP_SAFE", "NDP_SELECT", "NDP_V2", "NTEACH"}:
        rows, reference_features, summary, signature = NDP_SAFE.build_safe_online_neighbor_graph(
            bank, dataset.train, device, max_negative_pids=3, block_size=512, previous_signature=previous_signature
            , robust_positive=(arm in {"NDP_V2", "NTEACH"})
        )
    else:
        pids = torch.tensor([int(item["pid"]) for item in dataset.train], dtype=torch.long, device=device)
        rows, reference_features, summary, signature = NDP.build_online_neighbor_graph(
            bank, pids, device, max_negative_pids=3, block_size=512, previous_signature=previous_signature
        )
    return rows, reference_features, summary, signature


def annotate_teacher_graph(rows: dict[int, list[dict[str, Any]]], reference_features: torch.Tensor) -> dict[str, int]:
    """Attach fixed teacher margins and validity flags to a detached graph.

    The graph is made once from the B0 initialization.  A relation is trusted
    only when the teacher itself separates the same-PID reference from the
    candidate negative PID by at least 0.02.  Current training features are
    never used to rewrite this teacher selection.
    """
    reliable = 0
    relations = 0
    for anchor_id, anchor_rows in rows.items():
        anchor = reference_features[int(anchor_id)]
        for relation in anchor_rows:
            positive = reference_features[relation["positive_local"]]
            negative = reference_features[relation["negative_local"]]
            teacher_positive = (positive @ anchor).mean()
            teacher_margins = (teacher_positive - negative @ anchor).detach().float().cpu().tolist()
            relation["teacher_pid_margin"] = [float(value) for value in teacher_margins]
            relation["reference_pid_margin"] = [float(value) for value in teacher_margins]
            relation["historical_valid_pid"] = [bool(value >= 0.02) for value in teacher_margins]
            relations += len(teacher_margins)
            reliable += sum(bool(value >= 0.02) for value in teacher_margins)
    return {"teacher_relation_count": int(relations), "teacher_reliable_relation_count": int(reliable)}


def zero_stats() -> dict[str, float]:
    return {
        "batches": 0, "active_batches": 0, "selected_relations": 0,
        "active_relations": 0, "current_error_pid": 0, "correct_pid": 0,
        "margin_sum": 0.0, "purity_sum": 0.0, "ndp_loss_sum": 0.0,
        "ndp_raw_loss_sum": 0.0, "gpc_selected": 0,
        # The shared NDP forward helper keeps these counters for other arms;
        # initialize them here even though NLIST does not use their gates.
        "triplet_hard_excluded": 0, "triplet_filter_candidates": 0,
        "gate_evaluated": 0, "gate_kept": 0, "gate_conflict": 0,
        "gate_invalid": 0, "gate_cosine_sum": 0.0,
    }


def reduce_vec(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return [float(x) for x in tensor.detach().cpu().tolist()]


def eval_protocol(output_dir: Path, dataset_name: str, model_path: Path, cfg, device: torch.device) -> dict[str, Any]:
    dataset = build_dataset(cfg)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model_for_evaluation(cfg, dataset)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _, transform_test = build_img_transforms(cfg)
    if dataset_name == "ltcc":
        protocols = {
            "general": (dataset.query, dataset.gallery, R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=True), False),
            "clothes_changing": (dataset.query, dataset.gallery, R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=True), True),
        }
    else:
        protocols = {
            "same_clothes": (dataset.query_same, dataset.gallery, R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=True), False),
            "clothes_changing_C_to_A": (dataset.query_diff, dataset.gallery, R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=True), False),
        }
    results: dict[str, Any] = {}
    eval_logger = logger_for(output_dir / "eval.log")
    for name, (query, gallery, evaluator, ltcc_flag) in protocols.items():
        loader = DataLoader(
            ConcatDataset([ImageDataset(query, transform=transform_test), ImageDataset(gallery, transform=transform_test)]),
            batch_size=int(cfg.TEST.TEST_BATCH if hasattr(cfg.TEST, "TEST_BATCH") else 128),
            shuffle=False, num_workers=int(cfg.DATA.NUM_WORKERS), pin_memory=False, drop_last=False,
        )
        evaluator.reset()
        processor_test(cfg, model, evaluator, loader, eval_logger, device, test=True, ltcc=ltcc_flag, clothChanging=True)
        cmc, mAP, *_ = evaluator.compute()
        results[name] = {
            "rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9]),
            "mAP": float(mAP), "mINP": None, "mINP_status": "Unavailable: repository evaluator does not expose mINP",
            "query_count": len(query), "gallery_count": len(gallery),
        }
    return {"dataset": dataset_name, "model": str(model_path), "model_sha256": sha256_file(model_path), "protocols": results, "evaluated_utc": utc_now()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    rank, device = runtime_gate()
    output_dir = Path(args.output_dir).resolve() / args.dataset.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_no_rank_state_files(output_dir)
    artifact_dir = output_dir / "smoke" if args.smoke_steps else output_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = logger_for(artifact_dir / "train.log") if rank == 0 else None
    run_dir = output_dir / "runtime"
    cfg = cfg_for(args.dataset, run_dir)
    set_seed(SEED)
    dataset = build_dataset(cfg)
    if args.init_model:
        init_path = Path(args.init_model).resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"missing model-only warm-start: {init_path}")
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        init_manifest = {"state_hash": state_hash(state), "source": str(init_path), "model_only_warm_start": True}
    else:
        state, init_manifest = load_initialization(Path(args.output_dir).resolve(), args.dataset)
    # The exact formal common init is already materialized locally.  Build the
    # same architecture without invoking a fresh pretrained download.
    raw = build_model_for_evaluation(cfg, dataset)
    raw.load_state_dict(state, strict=True)
    raw.to(device).train()
    init_hash = state_hash(FORMAL.cpu_state_dict(raw))
    all_hashes: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(all_hashes, init_hash)
    if len(set(all_hashes)) != 1 or init_hash != init_manifest["state_hash"]:
        raise RuntimeError(f"initial parameter mismatch: {all_hashes}")
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, _center_optimizer = FORMAL.make_optimizer(cfg, raw, center)
    scheduler = FORMAL.create_scheduler(cfg, optimizer)
    initial_scale = int(FORMAL.SPEC[args.dataset]["initial_scale"])
    scaler = amp.GradScaler(init_scale=initial_scale)
    model = DDP(raw, device_ids=[rank], output_device=rank, find_unused_parameters=False, broadcast_buffers=False, bucket_cap_mb=25)
    gpc_bank = FORMAL.build_train_only_pair_bank(dataset, args.dataset)
    path_to_index = gpc_bank["path_to_index"]
    sampler = FORMAL.FinalBatchSampler(dataset.train, int(FORMAL.SPEC[args.dataset]["batch_size"]), int(FORMAL.SPEC[args.dataset]["num_instances"]), int(FORMAL.SPEC[args.dataset]["steps_per_epoch"]), SEED, rank)
    loader = FORMAL.build_train_loader(cfg, dataset, sampler)
    # The formal GPC namespace is part of the frozen selector hash.  It must
    # be literal and arm-independent for B0 and NDP_SAFE.
    gpc_phase = NDP.set_exact_gpc_pair_context(GPC_CONTEXT, SEED)
    resume = output_dir / "resume_latest.pth"
    start_epoch = 1
    global_step = 0
    if resume.is_file() and args.resume:
        previous_epoch, global_step = load_resume(resume, raw, optimizer, scheduler, scaler, rank, device)
        start_epoch = previous_epoch + 1
        raw.train()
        if logger:
            logger.info("从单文件恢复：epoch=%d global_step=%d", previous_epoch, global_step)
    elif resume.is_file() and not args.resume:
        raise RuntimeError(f"resume_latest.pth exists; pass --resume or move it before a fresh run: {resume}")

    steps = int(FORMAL.SPEC[args.dataset]["steps_per_epoch"])
    max_epochs = int(args.epochs)
    if max_epochs < 1 or max_epochs > EPOCHS:
        raise ValueError(f"epochs must be in 1..{EPOCHS}")
    end_epoch = min(max_epochs, EPOCHS)
    metrics_path = artifact_dir / "metrics_per_epoch.csv"
    summary_path = artifact_dir / "online_neighbor_summary.csv"
    fields = ["epoch", "global_step", "train_loss", "ce_loss", "triplet_loss", "bio_loss", "gpc_loss", "ndp_loss", "ndp_raw_loss", "ndp_feature_grad_norm", "main_feature_grad_norm", "ndp_main_grad_ratio", "ndp_lambda", "lr", "amp_scale_min", "amp_scale_max", "amp_scale_end", "skipped_steps", "optimizer_updates", "ndp_active_batches", "ndp_active_rate", "ndp_selected_relations", "ndp_active_relations", "mean_current_margin", "purity10", "online_relation_count", "online_valid_anchor_count", "online_neighbor_change_rate", "online_graph_sha256"]
    summary_fields = ["epoch", "online_relation_count", "online_valid_anchor_count", "online_neighbor_change_rate", "online_graph_sha256", "fresh_bank_shape", "fresh_bank_device", "train_only", "query_gallery_used"]
    if rank == 0 and not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
    if rank == 0 and not summary_path.exists():
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=summary_fields).writeheader()
    dist.barrier()
    previous_signature = None
    cached_reference_rows: dict[int, list[dict[str, Any]]] = {}
    cached_reference_features = torch.empty((0, 1024), dtype=torch.float32, device=device)
    cached_graph_summary = {"online_relation_count": 0, "online_valid_anchor_count": 0, "online_neighbor_change_rate": None, "online_neighbor_graph_sha256": "", "fresh_bank_shape": [], "online_similarity_device": str(device)}
    cached_graph_epoch = 0
    history: list[dict[str, Any]] = []
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            scheduler.step(epoch)
            reference_rows = cached_reference_rows
            reference_features = cached_reference_features
            graph_summary = cached_graph_summary
            if args.arm in {"NDP_SAFE", "NDP_SELECT", "NDP_V2", "NTEACH"} and (cached_graph_epoch == 0 or (args.arm != "NTEACH" and epoch - cached_graph_epoch >= (NDP_SAFE_GRAPH_UPDATE_EVERY if args.arm == "NDP_SAFE" else (NDP_SELECT_GRAPH_UPDATE_EVERY if args.arm == "NDP_SELECT" else NDP_V2_GRAPH_UPDATE_EVERY)))):
                cached_reference_rows, cached_reference_features, cached_graph_summary, previous_signature = make_online_graph(raw, cfg, dataset, device, previous_signature, args.arm)
                if args.arm == "NTEACH":
                    teacher_counts = annotate_teacher_graph(cached_reference_rows, cached_reference_features)
                    cached_graph_summary.update(teacher_counts)
                cached_graph_summary["fresh_bank_shape"] = [len(dataset.train), 1024]
                cached_graph_epoch = epoch
                reference_rows, reference_features, graph_summary = cached_reference_rows, cached_reference_features, cached_graph_summary
            elif args.arm == "NLIST":
                reference_rows, reference_features, graph_summary, previous_signature = make_online_graph(raw, cfg, dataset, device, previous_signature, args.arm)
                graph_summary["fresh_bank_shape"] = [len(dataset.train), 1024]
            sampler.prepare(epoch)
            stats = zero_stats()
            sums = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "gpc", "ndp", "ndp_raw", "ndp_feature_grad_norm", "main_feature_grad_norm")}
            local_skips = 0
            local_scale_min = float(scaler.get_scale())
            local_scale_max = local_scale_min
            plan = sampler.meta
            plans = [None] * WORLD_SIZE
            dist.all_gather_object(plans, plan)
            iterator = iter(loader)
            for iteration, batch in enumerate(iterator, start=1):
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                ndp_lambda = 0.0
                if args.arm == "NDP_SAFE":
                    ndp_lambda = LAMBDA_NDP_SAFE * min(1.0, float(epoch) / float(NDP_SAFE_WARMUP_EPOCHS))
                elif args.arm == "NDP_SELECT" and epoch >= NDP_SELECT_START_EPOCH:
                    ndp_lambda = LAMBDA_NDP_SELECT * min(1.0, float(epoch - NDP_SELECT_START_EPOCH + 1) / float(NDP_SELECT_WARMUP_EPOCHS))
                elif args.arm == "NDP_V2" and epoch >= NDP_V2_START_EPOCH:
                    ndp_lambda = LAMBDA_NDP_V2 * min(1.0, float(epoch - NDP_V2_START_EPOCH + 1) / float(NDP_V2_WARMUP_EPOCHS))
                elif args.arm == "NTEACH" and epoch >= NTEACH_START_EPOCH:
                    ndp_lambda = LAMBDA_NDP_TEACH
                result = NDP.forward_batch(model, loss_fn, batch, device, path_to_index, gpc_bank, reference_rows, reference_features, args.arm, epoch, rank, iteration, global_step, stats, 3, 0.02, ndp_lambda=ndp_lambda)
                if not bool(torch.isfinite(result["total"].detach()).item()):
                    raise RuntimeError(f"non-finite total loss at {args.dataset}/epoch{epoch}/iteration{iteration}")
                scaler.scale(result["total"]).backward()
                scaler.step(optimizer)
                inf = int(NDP.base.found_inf(scaler, optimizer) > 0)
                scaler.update()
                local_skips += inf
                local_scale_min = min(local_scale_min, float(scaler.get_scale()))
                local_scale_max = max(local_scale_max, float(scaler.get_scale()))
                for key in sums:
                    sums[key] += NDP.scalar(result[key])
                if args.smoke_steps and iteration >= args.smoke_steps:
                    break
            del iterator
            if hasattr(loader, "shutdown"):
                loader.shutdown()
            # Reduce epoch means and counts across ranks.  The formal batch
            # budget is fixed, while smoke mode deliberately reports its raw
            # prefix length separately.
            denom = max(1, iteration)
            reduced = reduce_vec([sums[key] / denom for key in sums], device)
            count_values = reduce_vec([stats["active_batches"], stats["selected_relations"], stats["active_relations"], stats["current_error_pid"], stats["correct_pid"], stats["margin_sum"], stats["purity_sum"], stats["batches"], local_skips], device)
            scale_values = torch.tensor([local_scale_min, local_scale_max, float(scaler.get_scale())], dtype=torch.float64, device=device)
            dist.all_reduce(scale_values[0], op=dist.ReduceOp.MIN); dist.all_reduce(scale_values[1], op=dist.ReduceOp.MAX); dist.all_reduce(scale_values[2], op=dist.ReduceOp.MAX)
            if rank == 0:
                total_batches = max(1.0, count_values[7])
                row = {
                    "epoch": epoch, "global_step": global_step, "train_loss": reduced[0], "ce_loss": reduced[1], "triplet_loss": reduced[2], "bio_loss": reduced[3], "gpc_loss": reduced[4], "ndp_loss": reduced[5], "ndp_raw_loss": reduced[6], "ndp_feature_grad_norm": reduced[7], "main_feature_grad_norm": reduced[8], "ndp_main_grad_ratio": reduced[7] / max(reduced[8], 1e-12), "ndp_lambda": (LAMBDA_NDP_SAFE * min(1.0, float(epoch) / float(NDP_SAFE_WARMUP_EPOCHS)) if args.arm == "NDP_SAFE" else (LAMBDA_NDP_SELECT * min(1.0, float(epoch - NDP_SELECT_START_EPOCH + 1) / float(NDP_SELECT_WARMUP_EPOCHS)) if args.arm == "NDP_SELECT" and epoch >= NDP_SELECT_START_EPOCH else (LAMBDA_NDP_V2 * min(1.0, float(epoch - NDP_V2_START_EPOCH + 1) / float(NDP_V2_WARMUP_EPOCHS)) if args.arm == "NDP_V2" and epoch >= NDP_V2_START_EPOCH else 0.0))), "lr": float(optimizer.param_groups[0]["lr"]), "amp_scale_min": float(scale_values[0]), "amp_scale_max": float(scale_values[1]), "amp_scale_end": float(scale_values[2]), "skipped_steps": int(count_values[8]), "optimizer_updates": int(count_values[7] - count_values[8]), "ndp_active_batches": int(count_values[0]), "ndp_active_rate": float(count_values[0] / total_batches), "ndp_selected_relations": int(count_values[1]), "ndp_active_relations": int(count_values[2]), "mean_current_margin": float(count_values[5] / max(1.0, count_values[3] + count_values[4])), "purity10": float(count_values[6] / total_batches), "online_relation_count": int(graph_summary.get("online_relation_count", 0)), "online_valid_anchor_count": int(graph_summary.get("online_valid_anchor_count", 0)), "online_neighbor_change_rate": graph_summary.get("online_neighbor_change_rate"), "online_graph_sha256": graph_summary.get("online_neighbor_graph_sha256", ""),
                }
                row["ndp_lambda"] = float(ndp_lambda)
                with metrics_path.open("a", newline="", encoding="utf-8") as stream:
                    csv.DictWriter(stream, fieldnames=fields).writerow(row)
                with summary_path.open("a", newline="", encoding="utf-8") as stream:
                    csv.DictWriter(stream, fieldnames=summary_fields).writerow({"epoch": epoch, "online_relation_count": graph_summary.get("online_relation_count", 0), "online_valid_anchor_count": graph_summary.get("online_valid_anchor_count", 0), "online_neighbor_change_rate": graph_summary.get("online_neighbor_change_rate"), "online_graph_sha256": graph_summary.get("online_neighbor_graph_sha256", ""), "fresh_bank_shape": graph_summary.get("fresh_bank_shape", []), "fresh_bank_device": graph_summary.get("online_similarity_device", str(device)), "train_only": True, "query_gallery_used": False})
                history.append(row)
                if logger:
                    logger.info("epoch %d/%d loss=%.6f ndp=%.6f active=%d/%d lr=%.3e skip=%d", epoch, end_epoch, row["train_loss"], row["ndp_loss"], row["ndp_active_relations"], row["ndp_selected_relations"], row["lr"], row["skipped_steps"])
            dist.barrier()
            if args.smoke_steps:
                break
            if epoch % 5 == 0:
                save_resume(output_dir, raw, optimizer, scheduler, scaler, epoch, global_step, device, args.dataset)
                assert_no_rank_state_files(output_dir)
            dist.barrier()
    finally:
        if hasattr(loader, "shutdown"):
            try:
                loader.shutdown()
            except Exception:
                pass
    if args.smoke_steps:
        if rank == 0:
            write_json(artifact_dir / "smoke_result.json", {"dataset": args.dataset, "arm": args.arm, "steps": int(iteration), "passed": True, "no_per_rank_checkpoint": True, "created_utc": utc_now()})
        dist.barrier()
        cleanup_dist()
        return {"dataset": args.dataset, "mode": "smoke", "steps": int(iteration), "passed": True}
    dist.barrier()
    final_path = output_dir / "model_final.pth"
    if rank == 0:
        state_final = FORMAL.cpu_state_dict(raw)
        atomic_torch_save(state_final, final_path)
        final_path.chmod(0o444)
        meta = {"dataset": args.dataset, "arm": args.arm, "seed": SEED, "epochs_completed": end_epoch, "global_step": global_step, "model_path": str(final_path), "model_sha256": sha256_file(final_path), "model_state_hash": state_hash(state_final), "model_bytes": final_path.stat().st_size, "initialization_source": init_manifest.get("source"), "model_only_warm_start": bool(init_manifest.get("model_only_warm_start", False)), "gpc_context": GPC_CONTEXT, "lambda_gpc": 0.1, "lambda_ndp": LAMBDA_NDP_SAFE if args.arm == "NDP_SAFE" else (LAMBDA_NDP_SELECT if args.arm == "NDP_SELECT" else (LAMBDA_NDP_V2 if args.arm == "NDP_V2" else 0.0)), "ndp_safe_margin": NDP_SAFE_MARGIN if args.arm == "NDP_SAFE" else None, "ndp_safe_warmup_epochs": NDP_SAFE_WARMUP_EPOCHS if args.arm == "NDP_SAFE" else None, "ndp_safe_graph_update_every": NDP_SAFE_GRAPH_UPDATE_EVERY if args.arm == "NDP_SAFE" else (NDP_SELECT_GRAPH_UPDATE_EVERY if args.arm == "NDP_SELECT" else (NDP_V2_GRAPH_UPDATE_EVERY if args.arm == "NDP_V2" else None)), "ndp_select_start_epoch": NDP_SELECT_START_EPOCH if args.arm == "NDP_SELECT" else None, "ndp_select_warmup_epochs": NDP_SELECT_WARMUP_EPOCHS if args.arm == "NDP_SELECT" else None, "ndp_v2_start_epoch": NDP_V2_START_EPOCH if args.arm == "NDP_V2" else None, "ndp_v2_warmup_epochs": NDP_V2_WARMUP_EPOCHS if args.arm == "NDP_V2" else None, "nlist_tau": TAU if args.arm == "NLIST" else None, "train_only_reference": True, "query_gallery_used_during_training": False, "per_rank_checkpoint_files_created": False, "created_utc": utc_now()}
        meta["lambda_ndp"] = LAMBDA_NDP_TEACH if args.arm == "NTEACH" else meta["lambda_ndp"]
        meta["nteach_start_epoch"] = NTEACH_START_EPOCH if args.arm == "NTEACH" else None
        meta["nteach_teacher"] = "frozen B0 initialization feature bank; graph captured once before epoch-1 updates" if args.arm == "NTEACH" else None
        meta["nteach_teacher_model_path"] = init_manifest.get("source") if args.arm == "NTEACH" else None
        meta["nteach_teacher_state_hash"] = init_manifest.get("state_hash") if args.arm == "NTEACH" else None
        write_json(output_dir / "final_model.json", meta)
        # RUN_COMPLETE is reserved for the requested formal 60-epoch run.
        # Short paired gates must not be mistaken for a completed full run.
        if end_epoch == EPOCHS:
            write_json(output_dir / "RUN_COMPLETE.json", meta)
        else:
            write_json(output_dir / "SHORT_RUN_COMPLETE.json", {**meta, "status": "SHORT_RUN_COMPLETE"})
    dist.barrier()
    if rank == 0 and end_epoch == EPOCHS and (output_dir / "resume_latest.pth").is_file():
        (output_dir / "resume_latest.pth").unlink()
        if (output_dir / "resume_latest.json").is_file():
            (output_dir / "resume_latest.json").unlink()
    dist.barrier()
    assert_no_rank_state_files(output_dir)
    cleanup_dist()
    return {"dataset": args.dataset, "epochs_completed": end_epoch, "model": str(final_path) if rank == 0 else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal 60-epoch DIFFER+GPC+NDP-Listwise training")
    parser.add_argument("--mode", choices=("prepare", "smoke", "train", "eval"), required=True)
    parser.add_argument("--dataset", choices=("ltcc", "prcc", "all"), required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--arm", choices=("B0", "NDP_SAFE", "NDP_SELECT", "NDP_V2", "NTEACH", "NLIST"), default="NTEACH")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-model", type=Path, default=None, help="model-only warm-start for a matched short continuation")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seed != SEED:
        raise RuntimeError("this formal run is frozen to seed=1234")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ("ltcc", "prcc") if args.dataset == "all" else (args.dataset,)
    if args.mode == "prepare":
        print(json.dumps([prepare_initialization(args.output_dir, name) for name in names], ensure_ascii=False, indent=2))
        return
    if args.mode in {"smoke", "train"}:
        if args.dataset == "all":
            raise RuntimeError("smoke/train require one dataset per torchrun invocation")
        if args.mode == "smoke":
            args.smoke_steps = 20
            args.epochs = 1
        print(json.dumps(train(args), ensure_ascii=False, indent=2))
        return
    if args.mode == "eval":
        if args.dataset == "all":
            raise RuntimeError("eval requires one dataset")
        model_path = args.output_dir.resolve() / args.dataset.upper() / "model_final.pth"
        if not model_path.is_file():
            raise RuntimeError(f"missing final model: {model_path}")
        cfg = cfg_for(args.dataset, args.output_dir / args.dataset.upper() / "eval")
        if not torch.cuda.is_available():
            raise RuntimeError("evaluation requires PPU/GPU")
        torch.cuda.set_device(0)
        result = eval_protocol(args.output_dir.resolve() / args.dataset.upper(), args.dataset, model_path, cfg, torch.device("cuda:0"))
        write_json(args.output_dir.resolve() / args.dataset.upper() / "eval_final.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
