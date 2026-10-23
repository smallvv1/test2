#!/usr/local/bin/python3.10
"""Export detached train-only feat[0] for an NDP-V2 endpoint bank.

The update0 bank is reused from the existing 9576-image train-only export;
only the new update596 model is encoded.  No gradient or optimizer step is
performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE / "scripts"))
import gpc_v2_train as v2  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_state(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict) and isinstance(value.get("model"), dict):
        value = value["model"]
    return {str(k).replace("module.", "", 1): v for k, v in value.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", choices=("B0", "N2"), required=True)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--update0-feature", type=Path, required=True)
    ap.add_argument("--weight", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V2 feature export requires PPU/CUDA")
    started = time.time()
    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    ref = np.load(args.update0_feature, allow_pickle=False)
    if "update" not in ref.files or int(np.asarray(ref["update"]).item()) != 0:
        raise RuntimeError("update0 feature export is not explicit update0")
    image_index = np.asarray(ref["image_index"], dtype=np.int64)
    image_path = np.asarray(ref["image_path"])
    pid = np.asarray(ref["pid"], dtype=np.int64)
    clothes = np.asarray(ref["clothes"], dtype=np.int64)
    camid = np.asarray(ref["camid"], dtype=np.int64)
    update0 = np.asarray(ref["features"], dtype=np.float32)
    update0 /= np.maximum(np.linalg.norm(update0, axis=1, keepdims=True), 1e-12)
    if update0.shape != (len(image_index), 1024):
        raise RuntimeError(f"unexpected update0 shape {update0.shape}")

    cfg = v2.p7.configure()
    cfg.defrost(); cfg.DATA.ROOT = "/home/data"; cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"; cfg.DATA.NUM_WORKERS = 0; cfg.DATA.PIN_MEMORY = False; cfg.freeze()
    dataset = v2.base.build_dataset(cfg)
    if len(dataset.train) != len(image_index):
        raise RuntimeError(f"train count mismatch {len(dataset.train)} != {len(image_index)}")
    _, transform_test = v2.base.build_img_transforms(cfg)
    state = load_state(args.weight)
    model = v2.base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=dataset.num_train_pids, camera_num=v2.base.NUM_CAMERAS)
    model.load_state_dict(state, strict=True); model.to(device); model.eval()

    values = []
    from PIL import Image
    for start in range(0, len(image_index), 16):
        ids = list(range(start, min(start + 16, len(image_index))))
        tensors = []
        cams = []
        for idx in ids:
            with Image.open(Path(str(dataset.train[idx]["image_path"]))) as image:
                tensors.append(transform_test(image.convert("RGB")))
            cams.append(int(dataset.train[idx]["camid"]))
        images = torch.stack(tensors, dim=0).to(device, non_blocking=False)
        cams_t = torch.tensor(cams, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True):
            out = model(images, cam_label=cams_t)
        feat = out[1][0] if isinstance(out, tuple) else out
        if feat.ndim != 2 or feat.shape != (len(ids), 1024) or not bool(torch.isfinite(feat).all().item()):
            raise RuntimeError(f"invalid feat at batch {start // 16}: {tuple(feat.shape)}")
        values.append(F.normalize(feat.float(), dim=-1).cpu().numpy().astype(np.float32))
    current = np.concatenate(values, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack([update0, current]), updates=np.asarray([0, 596], dtype=np.int64), image_index=image_index, image_path=image_path, pid=pid, clothes=clothes, camid=camid)
    manifest = {
        "status": "PASS_NDP_V2_TRAIN_ONLY_FEATURE_EXPORT",
        "seed": args.seed, "arm": args.arm, "device": str(device),
        "update0_feature": str(args.update0_feature), "update0_sha256": sha256_file(args.update0_feature),
        "weight": str(args.weight), "weight_sha256": sha256_file(args.weight),
        "output": str(args.output), "output_sha256": sha256_file(args.output),
        "image_count": len(image_index), "feature_dim": 1024, "updates": [0, 596],
        "optimizer_step_executed": False, "official_query_gallery_used": False,
        "elapsed_seconds": time.time() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
