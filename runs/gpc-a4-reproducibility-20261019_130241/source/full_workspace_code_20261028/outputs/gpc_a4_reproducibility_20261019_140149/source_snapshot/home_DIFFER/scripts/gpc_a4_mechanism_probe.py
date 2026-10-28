#!/usr/local/bin/python3.10
"""GPC-A4 train-only 机制探针。

只在 PPU 上做独立 forward/autograd.grad，不调用 backward、optimizer.step 或
任何参数更新。使用当前 E30 checkpoint 和开发训练 PID；开发验证 PID 不进入
batch、loss 或选择权重。A4 的选择计划按实际新 wrapper 的跨 rank 全局 epoch
规则生成，随后只对 rank0 的前几个有效 batch 做梯度诊断。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
EXPECTED_DEVICE = "PPU-ZW810E"
TRAIN_ROOT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen"
SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
EPOCH = 30
PROBE_BATCHES = 4

sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))
from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_img_transforms  # noqa: E402
from model.build import factory  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402
import gpc_a4_recheck_train as a4  # noqa: E402

base = p7.base


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_cfg(path: Path):
    cfg = GLOBAL_CFG.clone()
    cfg.defrost()
    cfg.DATA.BIO_INDEX = 0
    cfg.MODEL.DEVICE_ID = 0
    cfg.merge_from_file(str(path))
    cfg.defrost()
    cfg.DATA.BIO_INDEX = [0]
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()
    return cfg


def require_ppu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("PPU-only 机制探针拒绝执行：torch.cuda.is_available()=False")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not names or any(name != EXPECTED_DEVICE for name in names):
        raise RuntimeError(f"PPU 设备不符合要求：{names}")
    torch.cuda.set_device(0)
    print(f"中文说明：机制探针使用 PPU={names}；CPU model fallback=False", flush=True)
    return torch.device("cuda:0")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config: Path, checkpoint: Path, device: torch.device):
    cfg = load_cfg(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    head = state.get("head.weight")
    if head is None:
        raise KeyError(f"checkpoint 缺少 head.weight：{checkpoint}")
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=int(head.shape[0]), camera_num=12)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load 失败：missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)
    model.train()
    return model, cfg


def active_state(dataset, train_pids: set[int]) -> dict[int, list[int]]:
    values: dict[int, set[int]] = defaultdict(set)
    for item in dataset.train:
        pid = int(item["pid"])
        if pid in train_pids:
            values[pid].add(int(item["clothes_id"]))
    if set(values) != train_pids:
        raise RuntimeError("探针 active_state 未覆盖冻结开发训练 PID")
    return {pid: sorted(clothes) for pid, clothes in sorted(values.items())}


def batch_entries(loader, batch_indices):
    return [{
        "pid": int(loader.dataset.samples[index][0]["pid"]),
        "clothes_id": int(loader.dataset.samples[index][0]["clothes_id"]),
        "image_path": str(loader.dataset.samples[index][0]["image_path"]),
    } for index in batch_indices]


def selection_plan(dataset, cfg, arm: str, seed: int, transform, state, index_map):
    p7.PHASE = f"gpc_v2_ltcc_20261016_161631_screen_{arm}_seed{seed}"
    p7.SEED = seed
    p7.WORLD_SIZE = 2
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    loaders = {rank: base.make_epoch_loader(dataset, transform, state, index_map, EPOCH, rank)[0] for rank in (0, 1)}
    records = []
    for rank, loader in loaders.items():
        for iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
            entries = batch_entries(loader, batch_indices)
            candidates = p7.candidate_lists(entries, path_to_index, bank)
            count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
            records.append((iteration, rank, candidates, count))
    records.sort(key=lambda row: (row[0], row[1]))
    totals = Counter()
    plan = {}
    for iteration, rank, candidates, count in records:
        if arm == "A4":
            selected = a4.cumulative_pid_balanced_selection(candidates["id"], count, EPOCH, rank, iteration, totals)
            selector = "pid_balanced_global_epoch"
        elif arm == "R2":
            selected = a4.random_budget_selection(candidates["id"], count, EPOCH, rank, iteration)
            selector = "random_budget"
        else:
            selected = p7.stable_selection(candidates["id"], count, EPOCH, rank, iteration, "id")
            selector = "phase7i_i3"
        plan[(EPOCH, rank, iteration)] = {
            "selected": selected,
            "candidate_counts": {key: len(value) for key, value in candidates.items()},
            "matched_count": count,
            "selector": selector,
        }
    return loaders[0], plan, bank, path_to_index


def zero_if_none(value, reference):
    return torch.zeros_like(reference) if value is None else value


def norm(value) -> float:
    return float(torch.linalg.vector_norm(value.detach().float()).item())


def cosine(a, b):
    na, nb = torch.linalg.vector_norm(a.detach().float()), torch.linalg.vector_norm(b.detach().float())
    if float(na.item()) <= 1e-12 or float(nb.item()) <= 1e-12:
        return None
    return float((a.detach().float() * b.detach().float()).sum().div(na * nb).item())


def diagnose_batch(model, loss_fn, batch, device, selected, arm, info, step):
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    cams = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    writer = base.MetricCapture()
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=cams)
    base_total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=step)
    components = dict(loss_fn.last_components)
    feature = feat[0]
    z = F.normalize(feature.float(), dim=-1)
    cos_values = []
    values = []
    for i, j, _image_key, _state_key in selected:
        c = torch.sum(z[i] * z[j])
        cos_values.append(c)
        values.append(1.0 - c)
    relation = torch.stack(values).mean() if values else z.sum() * 0.0
    base_loss = components["ce"] + components["triplet"] + components["bio"]

    def grad_of(loss, retain):
        return zero_if_none(torch.autograd.grad(loss, feature, retain_graph=retain, allow_unused=True)[0], feature)

    g_ce = grad_of(components["ce"], True)
    g_triplet = grad_of(components["triplet"], True)
    g_bio = grad_of(components["bio"], True)
    g_base = grad_of(base_loss, True)
    g_gpc = grad_of(relation, False)
    with torch.no_grad():
        sim = z @ z.T
        neg_mask = pids[:, None] != pids[None, :]
        nearest_neg = sim.masked_fill(~neg_mask, -2.0).max(dim=1).values
        pos = torch.stack(cos_values) if cos_values else torch.empty(0, device=device)
        finite = all(bool(torch.isfinite(x).all().item()) for x in (base_total, relation, g_base, g_gpc))
    gpc_applied = g_gpc * 0.1
    result = {
        "selected_pair_count": len(selected),
        "candidate_counts": info["candidate_counts"],
        "matched_pair_budget": int(info["matched_count"]),
        "selector": info["selector"],
        "positive_cosine_mean": float(pos.mean().item()) if pos.numel() else None,
        "nearest_negative_cosine_mean": float(nearest_neg.mean().item()),
        "local_identity_margin": float((pos.mean() - nearest_neg.mean()).item()) if pos.numel() else None,
        "relation_loss": float(relation.detach().float().item()),
        "relation_applied_loss": float((0.1 * relation).detach().float().item()),
        "ce_loss": float(components["ce"].detach().float().item()),
        "triplet_loss": float(components["triplet"].detach().float().item()),
        "bio_loss": float(components["bio"].detach().float().item()),
        "ce_grad_norm": norm(g_ce),
        "triplet_grad_norm": norm(g_triplet),
        "bio_grad_norm": norm(g_bio),
        "base_grad_norm": norm(g_base),
        "gpc_grad_norm_raw": norm(g_gpc),
        "gpc_grad_norm_applied_lambda": norm(gpc_applied),
        "gpc_over_base_grad_ratio": norm(gpc_applied) / max(norm(g_base), 1e-12),
        "cos_base_gpc": cosine(g_base, g_gpc),
        "cos_triplet_gpc": cosine(g_triplet, g_gpc),
        "finite": finite,
        "gradient_space": "共享 feat[0] 输出层，不是完整参数梯度",
    }
    del score, feat, base_total, relation, g_ce, g_triplet, g_bio, g_base, g_gpc
    model.zero_grad(set_to_none=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    device = require_ppu()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    rows = []
    for seed in (1234, 4321):
        for arm in ("B0", "A4", "R2"):
            directory = TRAIN_ROOT / f"{arm}_seed{seed}"
            checkpoint, config = directory / "model_final.pth", directory / "config.yml"
            set_seed(seed)
            model, cfg = load_model(config, checkpoint, device)
            dataset = base.build_dataset(cfg)
            transform, _ = build_img_transforms(cfg)
            state = active_state(dataset, train_pids)
            index_map = base.build_index_map(dataset)
            loader, plan, _bank, _path_to_index = selection_plan(dataset, cfg, arm, seed, transform, state, index_map)
            loss_fn, _center = __import__("loss.make_loss", fromlist=["make_loss"]).make_loss(cfg, num_classes=len(train_pids))
            found = 0
            for iteration, batch in enumerate(loader, start=1):
                info = plan[(EPOCH, 0, iteration)]
                selected = info["selected"]
                if not selected:
                    continue
                result = diagnose_batch(model, loss_fn, batch, device, selected, arm, info, iteration)
                result.update({"run_time": args.run_time, "arm": arm, "seed": seed, "epoch": EPOCH, "iteration": iteration, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)})
                rows.append(result)
                found += 1
                print(f"中文结果：{arm} seed={seed} batch={iteration} base_grad={result['base_grad_norm']:.6g} gpc_grad={result['gpc_grad_norm_raw']:.6g} ratio={result['gpc_over_base_grad_ratio']:.6g} cos={result['cos_base_gpc']}", flush=True)
                if found >= PROBE_BATCHES:
                    break
            if found < PROBE_BATCHES:
                raise RuntimeError(f"有效探针 batch 不足：{arm} seed={seed} got={found}")
            del loader, loss_fn, dataset, model
            torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "mechanism_probe_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    groups = {}
    for arm in ("B0", "A4", "R2"):
        for seed in (1234, 4321):
            subset = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            groups[f"{arm}_seed{seed}"] = {
                "probe_count": len(subset),
                "base_grad_norm": float(np.mean([r["base_grad_norm"] for r in subset])),
                "gpc_grad_norm_raw": float(np.mean([r["gpc_grad_norm_raw"] for r in subset])),
                "gpc_grad_norm_applied_lambda": float(np.mean([r["gpc_grad_norm_applied_lambda"] for r in subset])),
                "gpc_over_base_grad_ratio": float(np.mean([r["gpc_over_base_grad_ratio"] for r in subset])),
                "cos_base_gpc": float(np.mean([r["cos_base_gpc"] for r in subset if r["cos_base_gpc"] is not None])),
                "cos_triplet_gpc": float(np.mean([r["cos_triplet_gpc"] for r in subset if r["cos_triplet_gpc"] is not None])),
                "positive_cosine_mean": float(np.mean([r["positive_cosine_mean"] for r in subset])),
                "nearest_negative_cosine_mean": float(np.mean([r["nearest_negative_cosine_mean"] for r in subset])),
                "local_identity_margin": float(np.mean([r["local_identity_margin"] for r in subset])),
                "finite": all(r["finite"] for r in subset),
            }
    payload = {"run_time": args.run_time, "train_only": True, "official_query_gallery_used": False, "cpu_model_fallback": False, "device": "PPU-ZW810E", "gradient_space": "共享 feat[0] 输出层，不是完整参数梯度", "checkpoint_root": str(TRAIN_ROOT), "rows": rows, "summary": groups}
    (args.output / "mechanism_probe_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "rows": len(rows), "summary": groups}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
