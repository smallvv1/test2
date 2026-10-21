#!/usr/bin/env python3
"""在 PPU 上导出固定 continuation 的 update0 全部 LTCC train-only 特征。

只执行 eval/no_grad forward，不恢复 optimizer、不调用 optimizer.step。用于构建
真正覆盖多个异身份 PID 的 update0 参考图；后续节点另行按冻结关系抽取。
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

ROOT = Path("/home/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
sys.path.insert(0, str(ROOT / "scripts"))
import ndp_real_trajectory_activation as act  # noqa: E402
import gpc_v2_train as v2  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--resume-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("需要两张 PPU；禁止 CPU model fallback")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda:0")

    class Args:
        output = OLD_OUTPUT
        run_time = "20261015_155822"
        arm = "B0"
        stage = "two_seed"
        seed = args.seed
        epochs = 60
        full_data = False

    split = json.loads((OLD_OUTPUT / "development_split_manifest.json").read_text(encoding="utf-8"))
    protocol = v2.install_protocol(Args, split, args.output_dir, None)
    cfg = v2.p7.configure()
    dataset = v2.base.build_dataset(cfg)
    _transform_train, transform_test = v2.base.build_img_transforms(cfg)
    indices = list(range(len(dataset.train)))
    source_path = args.resume_dir / "rank0_state_epoch060.pt"
    source = act.restore_numpy_pickle(source_path)
    model = act.load_model(cfg, source["model"], device)
    model.eval()
    with torch.no_grad():
        values = act.encode(model, dataset, indices, transform_test, device)
    features = np.stack([values[i].numpy() for i in indices]).astype(np.float32)
    if features.shape != (len(indices), 1024) or not np.isfinite(features).all():
        raise RuntimeError(f"update0 全量特征异常: {features.shape}")
    metadata = {
        "image_index": np.asarray(indices, dtype=np.int64),
        "image_path": np.asarray([str(dataset.train[i]["image_path"]) for i in indices]),
        "pid": np.asarray([int(dataset.train[i]["pid"]) for i in indices], dtype=np.int64),
        "clothes": np.asarray([int(dataset.train[i]["clothes_id"]) for i in indices], dtype=np.int64),
        "camid": np.asarray([int(dataset.train[i]["camid"]) for i in indices], dtype=np.int64),
    }
    npz_path = args.output_dir / f"update0_full_features_seed{args.seed}.npz"
    np.savez_compressed(npz_path, features=features, update=np.asarray(0, dtype=np.int64), **metadata)
    write_json(args.output_dir / "feature_manifest.json", {
        "status": "PASS_PPU_UPDATE0_EVAL_NO_GRAD", "seed": args.seed, "update": 0,
        "feature_definition": "PPU EVA02 feat[0] -> FP32 L2 normalize -> detached CPU export",
        "split": "LTCC train only", "count": len(indices), "dim": 1024,
        "checkpoint": str(source_path), "checkpoint_sha256": sha256_file(source_path),
        "npz": str(npz_path), "npz_sha256": sha256_file(npz_path),
        "optimizer_step_executed": False, "official_query_gallery_used": False,
        "cpu_model_fallback": False, "device": str(device), "world_size": 2,
        "elapsed_seconds": time.time() - started,
        "source_hashes": {"script": sha256_file(Path(__file__)),
                          "activation_helper": sha256_file(ROOT / "scripts/ndp_real_trajectory_activation.py"),
                          "engine": sha256_file(Path("/data/DIFFER/scripts/gpc_v2_train.py"))},
    })
    print(json.dumps({"status": "PASS_PPU_UPDATE0_EVAL_NO_GRAD", "seed": args.seed, "images": len(indices), "npz": str(npz_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
