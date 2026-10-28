#!/usr/bin/env python3
"""LTCC train-only 静态关系诊断。

使用既有 Bio/GPC train feature bank，不读取 query/gallery，不训练模型。
Bio->GPC 是两个独立终点的横向比较，只用于检查候选关系的覆盖与数值行为，
不解释为单次训练更新的因果风险。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ndp_vladr_relation import local_identity_advantage, vladr_relation_kl  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def top_indices(scores: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    if len(candidates) <= count:
        order = np.argsort(-scores[candidates], kind="stable")
        return candidates[order]
    local = np.argpartition(-scores[candidates], count - 1)[:count]
    chosen = candidates[local]
    return chosen[np.argsort(-scores[chosen], kind="stable")]


def lme(values: np.ndarray, temperature: float) -> float:
    x = values.astype(np.float64) / float(temperature)
    m = float(np.max(x))
    return float(m + np.log(np.exp(x - m).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-anchors", type=int, default=1024)
    ap.add_argument("--tau", type=float, default=0.07)
    args = ap.parse_args()
    root = Path("/home/DIFFER/outputs/ndp_local_intrusion_20261020_155558/feature_banks") / f"seed{args.seed}"
    bio_path = root / "bio_feature_bank.npz"
    gpc_path = root / "gpc_feature_bank.npz"
    bio = np.load(bio_path, allow_pickle=True)
    gpc = np.load(gpc_path, allow_pickle=True)
    required = {"features", "pid", "clothes", "camid", "image_path"}
    if not required.issubset(bio.files) or not required.issubset(gpc.files):
        raise RuntimeError("feature bank 缺少必要字段")
    n = int(bio["features"].shape[0])
    if gpc["features"].shape != bio["features"].shape:
        raise RuntimeError("Bio/GPC feature shape 不一致")
    for key in ("pid", "clothes", "camid", "image_path"):
        if not np.array_equal(bio[key], gpc[key]):
            raise RuntimeError(f"Bio/GPC {key} 对齐失败")
    paths = bio["image_path"].astype(str)
    if not np.all(np.char.startswith(paths, "/home/data/LTCC_ReID/train/")):
        raise RuntimeError("检测到非 LTCC train 路径")
    if not np.isfinite(bio["features"]).all() or not np.isfinite(gpc["features"]).all():
        raise RuntimeError("feature bank 含非有限值")
    ref = bio["features"].astype(np.float32)
    cur = gpc["features"].astype(np.float32)
    ref /= np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
    cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
    sample_count = min(int(args.sample_anchors), n)
    rng = np.random.default_rng(20261021 + args.seed)
    anchors = np.sort(rng.choice(n, size=sample_count, replace=False))
    # 只为固定 anchor 计算相似度；不构造9576^2全矩阵。
    ref_scores = ref[anchors] @ ref.T
    cur_scores = cur[anchors] @ cur.T
    pos_counts = []
    neg_counts = []
    ref_adv = []
    cur_adv = []
    reliable = []
    raw_drop = []
    pair_rows = []
    for row, idx in enumerate(anchors):
        same = np.flatnonzero(bio["pid"] == bio["pid"][idx])
        same = same[same != idx]
        diff = np.flatnonzero(bio["pid"] != bio["pid"][idx])
        pos = top_indices(ref_scores[row], same, 4)
        neg = top_indices(ref_scores[row], diff, 20)
        if len(pos) == 0 or len(neg) == 0:
            continue
        r = lme(ref[idx][None, :] @ ref[pos].T, args.tau) - lme(ref[idx][None, :] @ ref[neg].T, args.tau)
        c = lme(cur[idx][None, :] @ cur[pos].T, args.tau) - lme(cur[idx][None, :] @ cur[neg].T, args.tau)
        pos_counts.append(len(pos)); neg_counts.append(len(neg)); ref_adv.append(r); cur_adv.append(c)
        reliable.append(r > 0.05); raw_drop.append(r - c)
        pair_rows.append({
            "anchor": int(idx), "pid": int(bio["pid"][idx]),
            "positive_count": int(len(pos)), "negative_count": int(len(neg)),
            "positive_cross_clothes": bool(np.any(bio["clothes"][pos] != bio["clothes"][idx])),
            "reference_advantage": float(r), "current_advantage": float(c),
            "advantage_drop": float(r - c), "reference_reliable": bool(r > 0.05),
        })
    ref_adv_arr = np.asarray(ref_adv, dtype=np.float64)
    cur_adv_arr = np.asarray(cur_adv, dtype=np.float64)
    reliable_arr = np.asarray(reliable, dtype=bool)
    drop_arr = np.asarray(raw_drop, dtype=np.float64)
    # A1 关系蒸馏只在固定子集上计算；同时给出官方含对角线与NDP排除对角线版本。
    subset = anchors[: min(512, len(anchors))]
    ref_t = torch.from_numpy(ref[subset])
    cur_t = torch.from_numpy(cur[subset]).requires_grad_(True)
    kl_ex, kl_info_ex = vladr_relation_kl(cur_t, ref_t, temperature=0.005, exclude_diagonal=True)
    kl_off, kl_info_off = vladr_relation_kl(cur_t, ref_t, temperature=0.005, exclude_diagonal=False)
    result = {
        "status": "PASS",
        "run_type": "train_only_static_diagnostic",
        "causal_warning": "Bio E60 与 GPC E60 是独立终点，不是同一轨迹相邻状态；本结果不证明单次更新风险。",
        "seed": args.seed,
        "n_train_images": n,
        "n_train_pids": int(np.unique(bio["pid"]).size),
        "sample_anchor_count": sample_count,
        "valid_local_anchor_count": int(len(ref_adv_arr)),
        "valid_local_anchor_rate": float(len(ref_adv_arr) / sample_count),
        "feature_bank": {
            "bio": str(bio_path), "bio_sha256": sha256_file(bio_path),
            "gpc": str(gpc_path), "gpc_sha256": sha256_file(gpc_path),
            "path_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
            "feature_definition": "normalized feat[0], train only",
        },
        "local_candidate": {
            "positive_neighbor_count": {"mean": float(np.mean(pos_counts)), "min": int(min(pos_counts)), "max": int(max(pos_counts))},
            "negative_neighbor_count": {"mean": float(np.mean(neg_counts)), "min": int(min(neg_counts)), "max": int(max(neg_counts))},
            "reference_reliable_threshold": 0.05,
            "reference_reliable_anchor_count": int(reliable_arr.sum()),
            "reference_reliable_rate": float(reliable_arr.mean()),
            "advantage_drop_mean_reliable": float(drop_arr[reliable_arr].mean()) if reliable_arr.any() else None,
            "advantage_drop_median_reliable": float(np.median(drop_arr[reliable_arr])) if reliable_arr.any() else None,
            "violation_counts_by_epsilon": {
                str(e): int(np.sum(reliable_arr & (drop_arr > e))) for e in (0.0, 0.02, 0.1)
            },
            "cross_clothes_positive_anchor_rate": float(np.mean([r["positive_cross_clothes"] for r in pair_rows])) if pair_rows else 0.0,
        },
        "global_relation_kl": {
            "sample_count": int(len(subset)),
            "temperature": 0.005,
            "official_including_diagonal": float(kl_off.detach()),
            "official_active_rows": int(kl_info_off["active_rows"]),
            "ndp_excluding_diagonal": float(kl_ex.detach()),
            "ndp_active_rows": int(kl_info_ex["active_rows"]),
        },
        "relation_rows_preview": pair_rows[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("seed", "valid_local_anchor_count", "global_relation_kl")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
