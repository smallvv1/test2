#!/usr/local/bin/python3
"""用冻结开发训练集和 Bio common init 在 PPU 上冻结 A5 阈值。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
import phase7i_relation_specificity as p7  # noqa: E402

base = p7.base
b6 = p7.b6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, device = b6.init_dist()
    try:
        cfg = p7.configure()
        cfg.defrost()
        cfg.DATA.ROOT = "/home/data"
        cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"
        cfg.DATA.NUM_WORKERS = 0
        cfg.DATA.PIN_MEMORY = False
        cfg.freeze()
        state, init_audit = b6.load_common_init()
        model = base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=base.NUM_CLASSES, camera_num=base.NUM_CAMERAS)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        _, transform_test = base.build_img_transforms(cfg)
        rows = list(csv.DictReader((args.output / "development_split.csv").open(encoding="utf-8")))
        grouped = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["split"] == "development_train":
                grouped[int(row["pid"])][int(row["clothes_id"])].append(row)
        pairs = []
        for pid in sorted(grouped):
            clothes = sorted(grouped[pid])
            if len(clothes) < 2:
                continue
            left = sorted(grouped[pid][clothes[0]], key=lambda row: row["image_path"])[0]
            right = sorted(grouped[pid][clothes[1]], key=lambda row: row["image_path"])[0]
            pairs.append((pid, left, right))
        if len(pairs) < 2:
            raise RuntimeError("冻结开发训练集没有足够的跨衣 PID 阈值样本")
        from data.dataset_loader import ImageDataset
        local_pairs = pairs[rank::2]
        local_items = []
        for pid, left, right in local_pairs:
            local_items.extend([
                {"image_path": left["image_path"], "pid": pid, "camid": int(left["camid"])},
                {"image_path": right["image_path"], "pid": pid, "camid": int(right["camid"])},
            ])
        loader = DataLoader(ImageDataset(local_items, transform=transform_test), batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
        features = {}
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=False)
                cams = batch["camid"].to(device, non_blocking=False)
                with amp.autocast(enabled=True):
                    output = model(images, cam_label=cams)
                feature_output = output[1] if isinstance(output, tuple) else output
                if isinstance(feature_output, (list, tuple)):
                    feature_output = feature_output[0]
                values = torch.nn.functional.normalize(feature_output.float(), dim=-1)
                if not bool(torch.isfinite(values).all().item()):
                    raise RuntimeError("A5 初始阈值特征出现 NaN/Inf")
                for index, path in enumerate(batch["image_path"]):
                    features[str(path)] = values[index].detach().cpu().tolist()
        local_values = []
        for pid, left, right in local_pairs:
            a = np.asarray(features[left["image_path"]], dtype=np.float32)
            b = np.asarray(features[right["image_path"]], dtype=np.float32)
            local_values.append({"pid": pid, "left": left["image_path"], "right": right["image_path"], "cosine": float(np.dot(a, b))})
        gathered = [None, None]
        dist.all_gather_object(gathered, local_values)
        if rank == 0:
            values = [item for part in gathered for item in (part or [])]
            cosines = np.asarray([item["cosine"] for item in values], dtype=np.float64)
            if not np.isfinite(cosines).all():
                raise RuntimeError("A5 阈值分布出现 NaN/Inf")
            tau = float(np.quantile(cosines, 0.75))
            payload = {
                "formula": "relu(tau - cosine) with normalized feat[0]",
                "tau": tau,
                "tau_quantile": 0.75,
                "pair_count": len(values),
                "pair_definition": "每个开发训练 PID 的两个最低 clothing label，各取字典序第一张训练图",
                "source": "固定开发训练集；不含开发验证、官方 query/gallery/test",
                "frozen_development_split": str(args.output / "development_split_manifest.json"),
                "common_init": init_audit,
                "cosine_min": float(cosines.min()),
                "cosine_median": float(np.median(cosines)),
                "cosine_max": float(cosines.max()),
                "cosine_p25": float(np.quantile(cosines, 0.25)),
                "cosine_p75": float(np.quantile(cosines, 0.75)),
                "cosine_p90": float(np.quantile(cosines, 0.90)),
                "pairs_sha256": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
                "frozen_before_candidate_training": True,
                "official_query_gallery_used": False,
            }
            (args.output / "saturation_threshold.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            print("A5_THRESHOLD_FREEZE_PASS " + json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str), flush=True)
        dist.barrier()
        b6.cleanup_dist()
        return 0
    except Exception as exc:
        print("中文结论：A5 阈值冻结失败，禁止候选训练。原始错误：" + repr(exc), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
