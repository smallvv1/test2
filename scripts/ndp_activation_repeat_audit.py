#!/usr/local/bin/python3.10
"""同一真实轨迹 checkpoint 的无更新重复重编码审计。

只使用既有 B0+GPC-v1 continuation 的 update400 模型和已冻结诊断关系；
不执行 optimizer.step，不读取官方 query/gallery。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
ORIGINAL_RUN_TIME = "20261015_155822"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE / "scripts"))
import gpc_v2_train as v2  # noqa: E402
import ndp_real_trajectory_activation as act  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=(1234, 4321), required=True)
    ap.add_argument("--trajectory-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("PPU repeat audit requires two PPU devices; CPU fallback forbidden")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if any("PPU-ZW810E" not in name for name in names):
        raise RuntimeError(f"unexpected devices: {names}")

    class Args:
        output = OLD_OUTPUT
        run_time = ORIGINAL_RUN_TIME
        arm = "B0"
        stage = "two_seed"
        epochs = 60
        full_data = False

    install_args = Args()
    install_args.seed = args.seed
    split = v2.load_split(OLD_OUTPUT)
    protocol = v2.install_protocol(install_args, split, args.output, None)
    cfg = v2.p7.configure()
    dataset = v2.base.build_dataset(cfg)
    _, transform_test = v2.base.build_img_transforms(cfg)
    groups_path = args.trajectory_dir / "activation_audit" / "fixed_diagnostic_groups.json"
    groups_payload = json.loads(groups_path.read_text(encoding="utf-8"))
    groups = groups_payload["groups"]
    all_indices = sorted({int(idx) for group in groups for idx in [group["anchor"], *group["positives"], *group["negatives"]]})
    state_path = args.trajectory_dir / "node_update0400" / "model.pth"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model = act.load_model(cfg, state, device)
    model.eval()
    with torch.no_grad():
        first = act.encode(model, dataset, all_indices, transform_test, device)
        second = act.encode(model, dataset, all_indices, transform_test, device)
    rows = []
    max_abs = 0.0
    min_cos = 1.0
    for idx in all_indices:
        a = first[idx].float()
        b = second[idx].float()
        max_abs = max(max_abs, float((a - b).abs().max().item()))
        min_cos = min(min_cos, float(F.cosine_similarity(a[None], b[None]).item()))
        rows.append({"index": idx, "first_sha256": sha256_tensor(a), "second_sha256": sha256_tensor(b), "max_abs": float((a - b).abs().max().item()), "cosine": float(F.cosine_similarity(a[None], b[None]).item())})
    write_json(args.output / "repeat_encoding_summary.json", {
        "status": "PASS" if max_abs == 0.0 else "NUMERIC_DIFFERENCE_OBSERVED",
        "seed": args.seed,
        "update": 400,
        "device_names": names,
        "checkpoint": str(state_path),
        "checkpoint_sha256": sha256_file(state_path),
        "groups_sha256": groups_payload["groups_sha256"],
        "graph_sha256": groups_payload["graph_sha256"],
        "group_count": len(groups),
        "image_count": len(all_indices),
        "feature_dim": 1024,
        "same_model_state": True,
        "same_eval_preprocess": True,
        "optimizer_step_executed": False,
        "max_abs_difference": max_abs,
        "min_cosine": min_cos,
        "rows": rows,
    })
    print(json.dumps({"seed": args.seed, "images": len(all_indices), "max_abs": max_abs, "min_cosine": min_cos}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
