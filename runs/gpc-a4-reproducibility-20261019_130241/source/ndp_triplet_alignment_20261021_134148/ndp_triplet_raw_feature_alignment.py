#!/usr/bin/env python3
"""只读获取真实训练 Batch 的 EVA02 feat[0]，对齐原始 Triplet 语义。

只运行 eval/no_grad 模型前向；随后把 detached 特征作为叶子张量做
TripletLoss autograd，不反传到网络、不调用 optimizer.step、不写回 checkpoint。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
sys.path.insert(0, "/data/DIFFER/scripts")
sys.path.insert(0, "/home/DIFFER/scripts")
import gpc_v2_train as v2  # noqa: E402
import ndp_real_trajectory_activation as act  # noqa: E402

from loss.triplet_loss import TripletLoss, euclidean_dist, hard_example_mining  # noqa: E402


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def state_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def jsonable_tensor(x):
    return {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device), "finite": bool(torch.isfinite(x).all().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epoch", type=int, default=60)
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("PPU/CUDA device required; CPU model fallback forbidden")
    device = torch.device("cuda:0")

    class Args:
        output = OLD_OUTPUT; run_time = "20261015_155822"; arm = "B0"; stage = "two_seed"
        epochs = 60; full_data = False; seed = args.seed
    split = json.loads((OLD_OUTPUT / "development_split_manifest.json").read_text(encoding="utf-8"))
    protocol = v2.install_protocol(Args, split, args.output_dir, None)
    cfg = v2.p7.configure(); dataset = v2.base.build_dataset(cfg)
    _train_transform, transform_test = v2.base.build_img_transforms(cfg)
    active_state = v2.b6.full_active_state(dataset); index_map = v2.base.build_index_map(dataset)
    loader, _meta = protocol["restore"][2](dataset, None, active_state, index_map, args.epoch, args.rank)
    batch_positions = next(iter(loader.batch_sampler))
    entries = [loader.dataset.samples[pos][0] for pos in batch_positions]
    batch_ids = [int(next(i for i, x in enumerate(dataset.train) if str(x["image_path"]) == str(item["image_path"]))) for item in entries]
    images, cams = [], []
    for idx in batch_ids:
        with Image.open(Path(str(dataset.train[idx]["image_path"]))) as image:
            images.append(transform_test(image.convert("RGB")))
        cams.append(int(dataset.train[idx]["camid"]))
    image_tensor = torch.stack(images).to(device, non_blocking=False)
    cam_tensor = torch.tensor(cams, dtype=torch.long, device=device)
    labels = torch.tensor([int(dataset.train[idx]["pid"]) for idx in batch_ids], dtype=torch.long, device=device)
    model_path = OLD_OUTPUT / "training" / "two_seed" / f"B0_seed{args.seed}" / "rank0_state_epoch060.pt"
    state = act.restore_numpy_pickle(model_path)
    state = state["model"] if isinstance(state, dict) and isinstance(state.get("model"), dict) else state
    model = act.load_model(cfg, state, device); model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True):
        out = model(image_tensor, cam_label=cam_tensor)
    raw = out[1][0] if isinstance(out, tuple) else out
    if raw.ndim != 2 or raw.shape[1] != 1024:
        raise RuntimeError(f"unexpected feat[0] shape {tuple(raw.shape)}")
    model_output_dtype = str(raw.dtype)
    raw_detached = raw.detach().float()
    normed = torch.nn.functional.normalize(raw_detached, dim=-1)
    raw_norm = torch.linalg.vector_norm(raw_detached, dim=-1)

    def triplet_case(features):
        leaf = features.detach().clone().requires_grad_(True)
        loss_fn = TripletLoss()
        loss, dist_ap, dist_an = loss_fn(leaf, labels, normalize_feature=False)
        loss.backward()
        dap, dan, pidx, nidx = hard_example_mining(euclidean_dist(leaf, leaf), labels, return_inds=True)
        grad = leaf.grad.detach().float()
        return {
            "loss": float(loss.detach()), "dist_ap": [float(x) for x in dist_ap.detach().cpu()],
            "dist_an": [float(x) for x in dist_an.detach().cpu()],
            "hard_positive_local": [int(x) for x in pidx.detach().cpu()],
            "hard_negative_local": [int(x) for x in nidx.detach().cpu()],
            "gradient_norm": float(grad.norm()), "gradient_finite": bool(torch.isfinite(grad).all()),
            "gradient_sha256": sha(grad.cpu().numpy().tobytes()),
        }
    raw_case = triplet_case(raw_detached); norm_case = triplet_case(normed)
    raw_d = euclidean_dist(raw_detached, raw_detached); norm_d = euclidean_dist(normed, normed)
    _, _, raw_p, raw_n = hard_example_mining(raw_d, labels, return_inds=True)
    _, _, norm_p, norm_n = hard_example_mining(norm_d, labels, return_inds=True)
    batch_meta = [{"local": i, "image_index": idx, "pid": int(dataset.train[idx]["pid"]), "clothes": int(dataset.train[idx]["clothes_id"]), "camid": int(dataset.train[idx]["camid"]), "image_path": str(dataset.train[idx]["image_path"])} for i, idx in enumerate(batch_ids)]
    summary = {
        "status": "PASS_PPU_READONLY_TRIPLET_RAW_FEATURE_ALIGNMENT", "seed": args.seed, "epoch": args.epoch, "rank": args.rank,
        "device": str(device), "torch": torch.__version__, "checkpoint": str(model_path), "checkpoint_sha256": state_sha(model_path),
        "batch_size": len(batch_ids), "batch_ids": batch_ids, "batch_meta": batch_meta,
        "model_output_dtype": model_output_dtype,
        "raw_feature": jsonable_tensor(raw_detached), "normalized_feature": jsonable_tensor(normed),
        "raw_norm_min": float(raw_norm.min()), "raw_norm_mean": float(raw_norm.mean()), "raw_norm_max": float(raw_norm.max()),
        "raw_feature_sha256": sha(raw_detached.cpu().numpy().tobytes()), "normalized_feature_sha256": sha(normed.cpu().numpy().tobytes()),
        "triplet_raw": raw_case, "triplet_normalized": norm_case,
        "raw_vs_norm_hard_positive_same": bool(torch.equal(raw_p, norm_p)),
        "raw_vs_norm_hard_negative_same": bool(torch.equal(raw_n, norm_n)),
        "original_training_input_contract": "make_loss calls TripletLoss(feat[0], target, normalize_feature=False); this probe preserves that raw path",
        "normalized_bank_contract": "update0 full feature bank is a separate eval/no_grad normalized retrieval export, not assumed identical",
        "optimizer_step_executed": False, "model_parameters_gradients_requested": False, "official_query_gallery_used": False,
    }
    (args.output_dir / f"triplet_raw_feature_alignment_seed{args.seed}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()
