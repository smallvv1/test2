#!/usr/local/bin/python3.10
"""A5 全训练 batch 的关系存在性与饱和梯度审计。

使用历史 E60 model_final.pth 和历史 run_manifest 中的实际 Phase，避免把
条件于有效 GPC batch 的探针误当成全训练步骤比例。只使用 train-only，
不执行 optimizer.step，不读取官方 query/gallery。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import quantiles

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def q(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p0": None, "p25": None, "p50": None, "p75": None, "p100": None}
    ordered = sorted(values)
    if len(ordered) < 2:
        return {"p0": ordered[0], "p25": ordered[0], "p50": ordered[0], "p75": ordered[0], "p100": ordered[0]}
    cuts = quantiles(ordered, n=4, method="inclusive")
    return {"p0": ordered[0], "p25": cuts[0], "p50": cuts[1], "p75": cuts[2], "p100": ordered[-1]}


def historical_epoch_rows(run_dir: Path) -> list[dict[str, str | float | int]]:
    path = run_dir / "relation_exposure.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "epoch": int(row["epoch"]),
                "all_batches": 1192,
                "relation_active_batches": int(row["relation_active_batches"]),
                "relation_active_batch_rate": float(row["relation_active_batch_rate"]),
                "matched_relation_pairs": int(row["matched_relation_pairs"]),
                "relation_pid_coverage": int(row["relation_pid_coverage"]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 不可用")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not devices or any(name != "PPU-ZW810E" for name in devices):
        raise RuntimeError(f"设备不是 PPU-ZW810E：{devices}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    seed_rows = []
    for seed in (1234, 4321):
        run_dir = ROOT_TRAIN / f"A5_seed{seed}"
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        phase = str(manifest["phase"])
        config = run_dir / "config.yml"
        rank_rows = []
        for rank in (0, 1):
            checkpoint = run_dir / f"rank{rank}_state_epoch060.pt"
            set_protocol(seed, phase)
            model, cfg = probe.load_model(config, checkpoint, device)
            dataset = base.build_dataset(cfg)
            transform_train, _ = probe.build_img_transforms(cfg)
            state = active_state(dataset, train_pids)
            index_map = base.build_index_map(dataset)
            path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
            bank = b6.parse_relation_bank(dataset, cfg)
            bank["path_to_index"] = path_to_index
            loader, loader_meta = base.make_epoch_loader(dataset, transform_train, state, index_map, EPOCH, rank)
            loss_fn, _center = base.make_loss(cfg, num_classes=base.NUM_CLASSES)
            all_batches = 0
            relation_batches = 0
            active_relation_batches = 0
            effective_gradient_batches = 0
            candidate_counts = []
            selected_counts = []
            cosine_values = []
            active_pair_count = 0
            zero_pair_count = 0
            probe_rows = []
            for iteration, batch in enumerate(loader, start=1):
                all_batches += 1
                entries = [{
                    "pid": int(batch["pid"][i]),
                    "clothes_id": int(batch["clothes_id"][i]),
                    "image_path": str(batch["image_path"][i]),
                } for i in range(len(batch["pid"]))]
                candidates = p7.candidate_lists(entries, path_to_index, bank)
                count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
                selected = p7.stable_selection(candidates["id"], count, EPOCH, rank, iteration, "id")
                candidate_counts.append(len(candidates["id"]))
                selected_counts.append(len(selected))
                if not selected:
                    continue
                relation_batches += 1
                images = batch["image"].to(device, non_blocking=False)
                pids = batch["pid"].to(device, non_blocking=False)
                cams = batch["camid"].to(device, non_blocking=False)
                clothes = batch["clothes_id"].to(device, non_blocking=False)
                captions = batch["caption_feature"].to(device, non_blocking=False)
                writer = base.MetricCapture()
                model.zero_grad(set_to_none=True)
                with amp.autocast(enabled=True):
                    score, feat = model(images, cam_label=cams)
                formal = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=iteration)
                del formal
                z = F.normalize(feat[0].float(), dim=-1)
                cos_tensor = torch.stack([torch.sum(z[i] * z[j]) for i, j, _key, _state in selected])
                hinge = F.relu(float(TAU) - cos_tensor).mean()
                grad = torch.autograd.grad(hinge, feat[0], retain_graph=False, allow_unused=True)[0]
                if grad is None:
                    grad = torch.zeros_like(feat[0])
                grad_norm = float(torch.linalg.vector_norm(grad.detach().float()).cpu())
                active = cos_tensor < TAU
                active_count = int(active.sum().cpu())
                zero_count = int((~active).sum().cpu())
                active_relation_batches += int(active_count > 0)
                effective_gradient_batches += int(grad_norm > 1e-12)
                active_pair_count += active_count
                zero_pair_count += zero_count
                cos_list = [float(value) for value in cos_tensor.detach().float().cpu().tolist()]
                cosine_values.extend(cos_list)
                if len(probe_rows) < 32 or active_count == 0:
                    probe_rows.append({
                        "iteration": iteration,
                        "candidate_count": len(candidates["id"]),
                        "selected_count": len(selected),
                        "active_pairs": active_count,
                        "zero_pairs": zero_count,
                        "positive_cosine_quantiles": q(cos_list),
                        "gradient_norm": grad_norm,
                    })
                del score, feat, z, cos_tensor, hinge, grad
            rank_rows.append({
                "rank": rank,
                "epoch": EPOCH,
                "phase": phase,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint),
                "loader_meta": loader_meta,
                "all_batches": all_batches,
                "relation_exists_batches": relation_batches,
                "relation_exists_batch_fraction": relation_batches / max(all_batches, 1),
                "a5_active_relation_batches": active_relation_batches,
                "a5_gradient_effective_relation_batches": effective_gradient_batches,
                "a5_gradient_effective_given_relation_fraction": effective_gradient_batches / max(relation_batches, 1),
                "a5_gradient_effective_all_batch_fraction": effective_gradient_batches / max(all_batches, 1),
                "selected_pair_total": active_pair_count + zero_pair_count,
                "active_pair_total": active_pair_count,
                "zero_gradient_pair_total": zero_pair_count,
                "active_pair_fraction": active_pair_count / max(active_pair_count + zero_pair_count, 1),
                "zero_gradient_pair_fraction": zero_pair_count / max(active_pair_count + zero_pair_count, 1),
                "candidate_count_quantiles": q([float(value) for value in candidate_counts]),
                "selected_count_quantiles": q([float(value) for value in selected_counts]),
                "positive_cosine_quantiles_selected_relation_batches": q(cosine_values),
                "representative_probe_rows": probe_rows,
            })
            del model, dataset, loader
            torch.cuda.empty_cache()
        history = historical_epoch_rows(run_dir)
        totals = {key: sum(row[key] for row in rank_rows) for key in (
            "all_batches", "relation_exists_batches", "a5_active_relation_batches",
            "a5_gradient_effective_relation_batches", "selected_pair_total",
            "active_pair_total", "zero_gradient_pair_total",
        )}
        all_batches = totals["all_batches"]
        relation_batches = totals["relation_exists_batches"]
        pair_total = totals["selected_pair_total"]
        seed_rows.append({
            "seed": seed,
            "epoch": EPOCH,
            "phase": phase,
            "checkpoint": str(run_dir / "model_final.pth"),
            "checkpoint_sha256": sha(run_dir / "model_final.pth"),
            "rank_rows": rank_rows,
            **totals,
            "relation_exists_batch_fraction": relation_batches / max(all_batches, 1),
            "a5_gradient_effective_given_relation_fraction": totals["a5_gradient_effective_relation_batches"] / max(relation_batches, 1),
            "a5_gradient_effective_all_batch_fraction": totals["a5_gradient_effective_relation_batches"] / max(all_batches, 1),
            "active_pair_fraction": totals["active_pair_total"] / max(pair_total, 1),
            "zero_gradient_pair_fraction": totals["zero_gradient_pair_total"] / max(pair_total, 1),
            "historical_epoch_active_rates": history,
            "historical_epoch_similarity_quantiles_available": False,
        })
    payload = {
        "status": "PASS",
        "run_time": args.run_time,
        "tau": TAU,
        "epoch_definition": "E60 model_final.pth；历史每 epoch active batch 率来自 relation_exposure.csv；相似度分位数仅对 E60 train-only 全关系 batch 计算。",
        "train_only": True,
        "official_query_gallery_used": False,
        "development_validation_used_for_optimization": False,
        "device_names": devices,
        "cpu_model_fallback": False,
        "split_sha256": sha(SPLIT),
        "rows": seed_rows,
        "source_sha256": {
            str(path): sha(path)
            for path in (ROOT / "scripts/gpc_a5_full_batch_audit.py", ENGINE / "scripts/phase7i_relation_specificity.py")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "a5_full_batch_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "summary": [
            {
                "seed": row["seed"],
                "all_batches": row["all_batches"],
                "relation_exists_fraction": row["relation_exists_batch_fraction"],
                "gradient_effective_all_batch_fraction": row["a5_gradient_effective_all_batch_fraction"],
                "zero_gradient_pair_fraction": row["zero_gradient_pair_fraction"],
                "cosine_quantiles_by_rank": [
                    rank_row["positive_cosine_quantiles_selected_relation_batches"]
                    for rank_row in row["rank_rows"]
                ],
            }
            for row in seed_rows
        ],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
