#!/usr/local/bin/python3.10
"""A5 tau hinge 的 train-only PPU 饱和/梯度失活诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))

import gpc_a4_mechanism_probe as probe  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402

base = p7.base

SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
ROOT_TRAIN = ROOT / "outputs/gpc_v2_ltcc_20261015_155822/training/two_seed"
TAU = 0.9021800160
EPOCH = 60
RANK = 0
PROBE_BATCHES = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_protocol(seed: int, phase: str) -> None:
    p7.PHASE = phase
    b6.PHASE = phase
    base.PHASE = phase
    p7.SEED = seed
    b6.SEED = seed
    base.SEED = seed
    p7.WORLD_SIZE = 2
    b6.WORLD_SIZE = 2
    base.WORLD_SIZE = 2


def active_state(dataset, train_pids):
    values = defaultdict(set)
    for item in dataset.train:
        if int(item["pid"]) in train_pids:
            values[int(item["pid"])].add(int(item["clothes_id"]))
    return {pid: sorted(clothes) for pid, clothes in sorted(values.items())}


def norm(value):
    return float(torch.linalg.vector_norm(value.detach().float()).cpu())


def cosine(a, b):
    aa = a.detach().float()
    bb = b.detach().float()
    denom = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    if float(denom.cpu()) <= 1e-12:
        return None
    return float((aa * bb).sum().div(denom).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 不可用")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not names or any(name != "PPU-ZW810E" for name in names):
        raise RuntimeError(f"设备不是 PPU-ZW810E：{names}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    rows = []
    for seed in (1234, 4321):
        run_dir = ROOT_TRAIN / f"A5_seed{seed}"
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        phase = str(manifest["phase"])
        config = run_dir / "config.yml"
        checkpoint = run_dir / "model_final.pth"
        set_protocol(seed, phase)
        model, cfg = probe.load_model(config, checkpoint, device)
        dataset = base.build_dataset(cfg)
        transform_train, _ = probe.build_img_transforms(cfg)
        state = active_state(dataset, train_pids)
        index_map = base.build_index_map(dataset)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        bank = b6.parse_relation_bank(dataset, cfg)
        bank["path_to_index"] = path_to_index
        loader, _meta = base.make_epoch_loader(dataset, transform_train, state, index_map, EPOCH, RANK)
        loss_fn, _center = base.make_loss(cfg, num_classes=base.NUM_CLASSES)
        found = 0
        for iteration, batch in enumerate(loader, start=1):
            entries = [{
                "pid": int(batch["pid"][i]),
                "clothes_id": int(batch["clothes_id"][i]),
                "image_path": str(batch["image_path"][i]),
            } for i in range(len(batch["pid"]))]
            candidates = p7.candidate_lists(entries, path_to_index, bank)
            count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
            selected = p7.stable_selection(candidates["id"], count, EPOCH, RANK, iteration, "id")
            if not selected:
                continue
            images = batch["image"].to(device, non_blocking=False)
            pids = batch["pid"].to(device, non_blocking=False)
            cams = batch["camid"].to(device, non_blocking=False)
            clothes = batch["clothes_id"].to(device, non_blocking=False)
            captions = batch["caption_feature"].to(device, non_blocking=False)
            writer = base.MetricCapture()
            model.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                score, feat = model(images, cam_label=cams)
            base_total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=iteration)
            components = dict(loss_fn.last_components)
            base_loss = components["ce"] + components["triplet"] + components["bio"]
            z = F.normalize(feat[0].float(), dim=-1)
            cos_values = torch.stack([torch.sum(z[i] * z[j]) for i, j, _key, _state in selected])
            relation = F.relu(float(TAU) - cos_values).mean()
            g_base = torch.autograd.grad(base_loss, feat[0], retain_graph=True, allow_unused=True)[0]
            g_gpc = torch.autograd.grad(relation, feat[0], retain_graph=False, allow_unused=True)[0]
            g_base = torch.zeros_like(feat[0]) if g_base is None else g_base
            g_gpc = torch.zeros_like(feat[0]) if g_gpc is None else g_gpc
            rows.append({
                "seed": seed,
                "epoch": EPOCH,
                "iteration": iteration,
                "candidate_count": len(candidates["id"]),
                "matched_count": count,
                "selected_count": len(selected),
                "positive_cosine_mean": float(cos_values.mean().cpu()),
                "positive_cosine_min": float(cos_values.min().cpu()),
                "positive_cosine_max": float(cos_values.max().cpu()),
                "tau": TAU,
                "hinge_loss": float(relation.detach().float().cpu()),
                "hinge_active_pairs": int((cos_values < TAU).sum().cpu()),
                "hinge_zero_pairs": int((cos_values >= TAU).sum().cpu()),
                "hinge_active_fraction": float((cos_values < TAU).float().mean().cpu()),
                "base_loss": float(base_loss.detach().float().cpu()),
                "base_feature_grad_norm": norm(g_base),
                "gpc_feature_grad_norm": norm(g_gpc),
                "gpc_over_base_grad_ratio": norm(g_gpc) * 0.1 / max(norm(g_base), 1e-12),
                "cos_base_gpc": cosine(g_base, g_gpc),
                "finite": bool(torch.isfinite(relation).all().cpu() and torch.isfinite(g_gpc).all().cpu()),
            })
            found += 1
            if found >= PROBE_BATCHES:
                break
        del model, loss_fn, dataset, loader
        torch.cuda.empty_cache()
        if found < PROBE_BATCHES:
            raise RuntimeError(f"seed={seed} 有效 A5 batch 不足：{found}")
    summary = {}
    for seed in (1234, 4321):
        subset = [row for row in rows if row["seed"] == seed]
        valid_cos = [row["cos_base_gpc"] for row in subset if row["cos_base_gpc"] is not None]
        summary[str(seed)] = {
            "probe_batches": len(subset),
            "active_pair_fraction_mean": sum(row["hinge_active_fraction"] for row in subset) / len(subset),
            "zero_gradient_pair_fraction_mean": sum(1.0 - row["hinge_active_fraction"] for row in subset) / len(subset),
            "hinge_loss_mean": sum(row["hinge_loss"] for row in subset) / len(subset),
            "gpc_over_base_grad_ratio_mean": sum(row["gpc_over_base_grad_ratio"] for row in subset) / len(subset),
            "cos_base_gpc_mean": sum(valid_cos) / len(valid_cos) if valid_cos else None,
            "finite": all(row["finite"] for row in subset),
        }
    payload = {
        "status": "PASS",
        "run_time": args.run_time,
        "tau": TAU,
        "epoch_definition": "每个 seed 的 two_seed model_final.pth，RUN_COMPLETE 记录为完整 E60；不是 model_best，也不是另一条轨迹的 epoch1。",
        "train_only": True,
        "official_query_gallery_used": False,
        "development_validation_used_for_optimization": False,
        "device_names": names,
        "cpu_model_fallback": False,
        "split_sha256": sha256_file(SPLIT),
        "rows": rows,
        "summary": summary,
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (ROOT / "scripts/gpc_a5_saturation_probe.py", ENGINE / "scripts/phase7i_relation_specificity.py")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "a5_saturation_probe.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "summary": summary, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
