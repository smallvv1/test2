#!/usr/local/bin/python3.10
"""Evaluation-only LTCC query/gallery metrics for B0 and N1 fixed nodes.

The loader, EVA02 model construction, clothes-changing filter and feature
normalization are inherited from the released DIFFER evaluator.  This script
adds only mINP calculation and loops over already saved model nodes.  It does
not train, select a checkpoint, or write any training configuration.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path("/data/DIFFER")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from model import eva_meta_bio  # noqa: E402
from torch import nn  # noqa: E402
import gpc_v2_train as v2  # noqa: E402


class EvaluationOnlyReverseAdapter(nn.Module):
    def __init__(self, in_num: int, class_num: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.grl = eva_meta_bio.GradientReversalLayer(alpha)
        self.head_fc = nn.Linear(in_num, class_num, bias=True)


def install_eval_only_reverse_adapter(model: nn.Module) -> int:
    heads = getattr(model, "head_clip_bioReverse", None)
    if heads is None:
        return 0
    count = 0
    for i, head in enumerate(heads):
        if head.head_fc.bias is None:
            heads[i] = EvaluationOnlyReverseAdapter(head.head_fc.in_features, head.head_fc.out_features, head.alpha)
            count += 1
    return count


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_config(path: Path):
    # Use the exact training-side Bio-only constructor so the checkpoint does
    # not acquire the optional reverse heads from the generic eval YAML.
    cfg = v2.p7.configure(); cfg.defrost()
    # The server keeps the caption sidecar on /data/data; this is an
    # evaluation-path correction and does not modify the saved training YAML.
    cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.NUM_WORKERS = 0; cfg.DATA.PIN_MEMORY = False
    cfg.DATA.TEST_BATCH = 128
    cfg.freeze()
    return cfg


def make_loader(dataset, transform, batch_size: int, workers: int):
    return DataLoader(ConcatDataset([
        ImageDataset(dataset.query, transform=transform),
        ImageDataset(dataset.gallery, transform=transform),
    ]), batch_size=batch_size, num_workers=workers, pin_memory=False, shuffle=False, drop_last=False)


def load_state(path: Path) -> dict:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and isinstance(value.get("model"), dict):
        value = value["model"]
    elif isinstance(value, dict) and isinstance(value.get("state_dict"), dict):
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise RuntimeError(f"unsupported model artifact: {path}")
    return {str(k).replace("module.", "", 1): v for k, v in value.items()}


@torch.no_grad()
def extract(model, loader, device):
    feats, pids, cams, clothes, paths = [], [], [], [], []
    model.eval()
    for index, batch in enumerate(loader, 1):
        images = batch["image"].to(device)
        camids = batch["camid"].to(device)
        out = model(images, cam_label=camids)
        feature = out[1][0] if isinstance(out, tuple) else out
        if feature.ndim != 2 or not bool(torch.isfinite(feature).all()):
            raise RuntimeError(f"non-finite/invalid feature at batch {index}: {type(out)}")
        feats.append(feature.float().cpu())
        pids.extend(np.asarray(batch["pid"]).tolist())
        cams.extend(np.asarray(batch["camid"]).tolist())
        clothes.extend(np.asarray(batch["clothes_id"]).tolist())
        paths.extend(list(batch["image_path"]))
    return (torch.cat(feats, 0).numpy(), np.asarray(pids), np.asarray(cams),
            np.asarray(clothes), paths)


def ltcc_metrics(features, pids, cams, clothes, num_query: int) -> dict:
    feat = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    qf, gf = feat[:num_query], feat[num_query:]
    qpid, gpid = pids[:num_query], pids[num_query:]
    qcam, gcam = cams[:num_query], cams[num_query:]
    qcloth, gcloth = clothes[:num_query], clothes[num_query:]
    dist = 2.0 - 2.0 * (qf @ gf.T)
    order = np.argsort(dist, axis=1, kind="stable")
    cmc_values, ap_values, inp_values = [], [], []
    for qi in range(len(qpid)):
        ranked = order[qi]
        same_pid = gpid[ranked] == qpid[qi]
        remove = same_pid & ((gcam[ranked] == qcam[qi]) | (gcloth[ranked] == qcloth[qi]))
        keep = ~remove
        matches = same_pid[keep].astype(np.int32)
        if not np.any(matches):
            continue
        pos = np.flatnonzero(matches)
        cmc = np.zeros(min(50, len(matches)), dtype=np.float32)
        first = int(pos[0])
        if first < len(cmc):
            cmc[first:] = 1.0
        cum = np.cumsum(matches)
        precision = cum / np.arange(1, len(matches) + 1)
        ap_values.append(float((precision * matches).sum() / matches.sum()))
        last = int(pos[-1])
        inp_values.append(float(cum[last] / (last + 1)))
        cmc_values.append(cmc)
    if not cmc_values:
        raise RuntimeError("no valid LTCC query")
    cmc_mat = np.stack(cmc_values)
    return {
        "rank1": float(cmc_mat[:, 0].mean()),
        "rank5": float(cmc_mat[:, min(4, cmc_mat.shape[1] - 1)].mean()),
        "rank10": float(cmc_mat[:, min(9, cmc_mat.shape[1] - 1)].mean()),
        "mAP": float(np.mean(ap_values)),
        "mINP": float(np.mean(inp_values)),
        "valid_query_count": int(len(cmc_values)),
        "query_count": int(len(qpid)),
        "gallery_count": int(len(gpid)),
        "distance_nan_count": int(np.isnan(dist).sum()),
        "distance_inf_count": int(np.isinf(dist).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--node", action="append", nargs=3, metavar=("ARM", "UPDATE", "WEIGHT"), required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("official LTCC evaluation requires PPU/CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed); device = torch.device(f"cuda:{args.device_index}"); torch.cuda.set_device(device)
    cfg = load_config(args.config)
    # Match the training-side constructor: pretrained=False, then load the
    # saved full model.  Using the generic build_model() here would attempt a
    # fresh HuggingFace download before the checkpoint is loaded.
    dataset = v2.base.build_dataset(cfg); _, transform = v2.base.build_img_transforms(cfg)
    loader = make_loader(dataset, transform, cfg.DATA.TEST_BATCH, cfg.DATA.NUM_WORKERS)
    model = v2.base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg,
                                            num_classes=dataset.num_train_pids,
                                            camera_num=v2.base.NUM_CAMERAS)
    adapter_count = install_eval_only_reverse_adapter(model)
    model.to(device)
    results = []
    for arm, update, weight_text in args.node:
        weight = Path(weight_text)
        state = load_state(weight)
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"strict load failed for {weight}: missing={missing}, unexpected={unexpected}")
        features, pids, cams, clothes, paths = extract(model, loader, device)
        metrics = ltcc_metrics(features, pids, cams, clothes, dataset.num_query_imgs)
        results.append({"arm": arm, "update": int(update), "weight": str(weight.resolve()),
                        "weight_sha256": sha256_file(weight), "metrics": metrics})
        print("LTCC_NODE", json.dumps(results[-1], ensure_ascii=False), flush=True)
        del features, pids, cams, clothes, paths
        gc.collect(); torch.cuda.empty_cache()
    payload = {
        "status": "PASS_OFFICIAL_LTCC_EVAL_WITH_MINP",
        "seed": args.seed, "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config), "dataset": "ltcc",
        "counts": {"train": len(dataset.train), "query": len(dataset.query), "gallery": len(dataset.gallery),
                    "num_train_pids": dataset.num_train_pids, "num_camera": dataset.num_camera},
        "eval_contract": "released evaluator clothes-changing filter; normalized feat[0]; euclidean distance; no reranking",
        "eval_only_reverse_adapters": adapter_count,
        "optimizer_step_executed": False, "hyperparameter_selection": False,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LTCC_EVALUATION_DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
