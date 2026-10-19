#!/usr/local/bin/python3
"""Leakage-safe fixed LTCC development validation for GPC-v2 screening."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase7i_relation_specificity as p7  # noqa: E402

base = p7.base
b6 = p7.b6


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_split(output: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((output / "development_split_manifest.json").read_text(encoding="utf-8"))
    with (output / "development_split.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if manifest.get("query_gallery_used_for_training"):
        raise RuntimeError("split manifest permits query/gallery in training")
    return manifest, rows


def build_dev_protocol(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_pid_cloth: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        if row["split"] != "dev_validation":
            continue
        item = {"image_path": row["image_path"], "pid": int(row["pid"]), "clothes_id": int(row["clothes_id"]), "camid": 0}
        by_pid_cloth.setdefault((item["pid"], item["clothes_id"]), []).append(item)
    query, gallery = [], []
    for pid in sorted({key[0] for key in by_pid_cloth}):
        clothes = sorted(cloth for (item_pid, cloth) in by_pid_cloth if item_pid == pid)
        if len(clothes) < 2:
            continue
        query.append(sorted(by_pid_cloth[(pid, clothes[0])], key=lambda x: x["image_path"])[0])
        for cloth in clothes[1:]:
            gallery.extend(sorted(by_pid_cloth[(pid, cloth)], key=lambda x: x["image_path"]))
    if not query or not gallery:
        raise RuntimeError("fixed development validation protocol has no query/gallery items")
    return query, gallery


def cfg_for(epochs: int):
    cfg = p7.configure()
    cfg.defrost()
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.MAX_EPOCHS = epochs
    cfg.freeze()
    return cfg


def extract(model, device, items, transform, rank):
    from data.dataset_loader import ImageDataset
    local = items[rank::2]
    loader = DataLoader(ImageDataset(local, transform=transform), batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    records = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=False)
            cams = batch["camid"].to(device, non_blocking=False)
            with amp.autocast(enabled=True):
                output = model(images, cam_label=cams)
            feat = output[1][0] if isinstance(output, tuple) else output
            values = torch.nn.functional.normalize(feat.float(), dim=-1).detach().cpu().numpy()
            for index, path in enumerate(batch["image_path"]):
                records.append((str(path), int(batch["pid"][index]), values[index].tolist()))
    gathered = [None, None]
    dist.all_gather_object(gathered, records)
    return [record for rank_records in gathered for record in (rank_records or [])]


def retrieval_metrics(query_records, gallery_records):
    gallery_paths = [item[0] for item in gallery_records]
    gallery_pid = np.asarray([item[1] for item in gallery_records])
    gallery_features = np.asarray([item[2] for item in gallery_records], dtype=np.float32)
    query_features = np.asarray([item[2] for item in query_records], dtype=np.float32)
    sims = query_features @ gallery_features.T
    rank_hits = {1: [], 5: [], 10: []}
    aps = []
    for index, query in enumerate(query_records):
        order = np.argsort(-sims[index], kind="stable")
        matches = gallery_pid[order] == int(query[1])
        relevant = int(matches.sum())
        if relevant <= 0:
            continue
        for k in rank_hits:
            rank_hits[k].append(float(matches[:k].any()))
        cumulative = np.cumsum(matches)
        positions = np.flatnonzero(matches) + 1
        aps.append(float(np.mean(cumulative[matches] / positions)))
    return {
        "query_count": len(query_records),
        "gallery_count": len(gallery_records),
        "rank1": float(np.mean(rank_hits[1])) if rank_hits[1] else float("nan"),
        "rank5": float(np.mean(rank_hits[5])) if rank_hits[5] else float("nan"),
        "rank10": float(np.mean(rank_hits[10])) if rank_hits[10] else float("nan"),
        "mAP": float(np.mean(aps)) if aps else float("nan"),
        "query_gallery_protocol": "dev PID-disjoint validation: lowest-clothes first image query; remaining clothes gallery",
        "official_query_gallery_used": False,
        "gallery_paths_sha256": __import__("hashlib").sha256("\n".join(sorted(gallery_paths)).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "two_seed", "formal"), default="screen")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    manifest, rows = load_split(args.output)
    query, gallery = build_dev_protocol(rows)
    rank, device = b6.init_dist()
    try:
        cfg = cfg_for(args.epochs)
        _dataset = base.build_dataset(cfg)
        _train_transform, transform_test = base.build_img_transforms(cfg)
        results = []
        for arm in args.arms:
            candidates = sorted((args.output / "training" / args.stage).glob(f"{arm}_seed*/model_final.pth"))
            if not candidates:
                raise FileNotFoundError(f"no final checkpoint for {arm} in {args.stage}")
            for checkpoint in candidates:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=base.NUM_CLASSES, camera_num=base.NUM_CAMERAS)
                model.load_state_dict(state, strict=True)
                model.to(device)
                records_q = extract(model, device, query, transform_test, rank)
                records_g = extract(model, device, gallery, transform_test, rank)
                if rank == 0:
                    qmap = {item[0]: item for item in records_q}
                    gmap = {item[0]: item for item in records_g}
                    metrics = retrieval_metrics([qmap[key] for key in sorted(qmap)], [gmap[key] for key in sorted(gmap)])
                    metrics.update({"arm": arm, "checkpoint": str(checkpoint), "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(), "seed": int(checkpoint.parent.name.split("seed")[-1])})
                    results.append(metrics)
                del model, state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                dist.barrier()
        if rank == 0:
            write_json(args.output / f"dev_evaluation_{args.stage}.json", {"split_manifest": manifest, "query_count": len(query), "gallery_count": len(gallery), "results": results})
            with (args.output / f"dev_evaluation_{args.stage}.csv").open("w", newline="", encoding="utf-8") as stream:
                fields = ["arm", "seed", "checkpoint", "checkpoint_sha256", "query_count", "gallery_count", "rank1", "rank5", "rank10", "mAP", "query_gallery_protocol", "official_query_gallery_used", "gallery_paths_sha256"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results)
            print(json.dumps({"stage": args.stage, "results": results}, indent=2, sort_keys=True))
    finally:
        b6.cleanup_dist()


if __name__ == "__main__":
    main()
