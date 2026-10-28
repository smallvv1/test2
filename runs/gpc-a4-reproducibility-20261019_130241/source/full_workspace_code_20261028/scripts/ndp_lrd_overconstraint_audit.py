#!/usr/bin/env python3
"""LTCC train-only LRD-A/B/C 过约束审计。

使用既有 Bio/GPC E60 特征库做横向静态诊断；不读取 query/gallery，不更新模型。
Bio 与 GPC 是独立终点，因此任何差异都不解释为单步因果效果。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lme(x: np.ndarray, tau: float) -> float:
    y = x.astype(np.float64) / tau
    m = float(y.max())
    return float(m + np.log(np.exp(y - m).mean()))


def top(scores: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(candidates))
    order = np.argsort(-scores[candidates], kind="stable")[:k]
    return candidates[order]


def row_prob(pos: np.ndarray, neg: np.ndarray, tau: float) -> np.ndarray:
    values = np.concatenate([pos, neg]).astype(np.float64) / tau
    values -= values.max()
    ex = np.exp(values)
    return ex / ex.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-anchors", type=int, default=1024)
    ap.add_argument("--positive-k", type=int, default=4)
    ap.add_argument("--negative-k", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--reliability", type=float, default=0.05)
    ap.add_argument("--mass-epsilon", type=float, default=0.02)
    args = ap.parse_args()
    root = Path("/home/DIFFER/outputs/ndp_local_intrusion_20261020_155558/feature_banks") / f"seed{args.seed}"
    bio_path, gpc_path = root / "bio_feature_bank.npz", root / "gpc_feature_bank.npz"
    bio, gpc = np.load(bio_path, allow_pickle=True), np.load(gpc_path, allow_pickle=True)
    for key in ("pid", "clothes", "camid", "image_path"):
        if not np.array_equal(bio[key], gpc[key]):
            raise RuntimeError(f"Bio/GPC {key} 未对齐")
    paths = bio["image_path"].astype(str)
    if not np.all(np.char.startswith(paths, "/home/data/LTCC_ReID/train/")):
        raise RuntimeError("发现非 train-only 路径")
    ref = bio["features"].astype(np.float32)
    cur = gpc["features"].astype(np.float32)
    ref /= np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
    cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
    n = len(ref)
    rng = np.random.default_rng(20261021 + args.seed)
    anchors = np.sort(rng.choice(n, size=min(args.sample_anchors, n), replace=False))
    ref_score = ref[anchors] @ ref.T
    cur_score = cur[anchors] @ cur.T
    rows = []
    for row, idx in enumerate(anchors):
        pos_pool = np.flatnonzero(bio["pid"] == bio["pid"][idx])
        pos_pool = pos_pool[pos_pool != idx]
        neg_pool = np.flatnonzero(bio["pid"] != bio["pid"][idx])
        pos = top(ref_score[row], pos_pool, args.positive_k)
        neg = top(ref_score[row], neg_pool, args.negative_k)
        if len(pos) == 0 or len(neg) == 0:
            continue
        rpos, rneg = ref[idx] @ ref[pos].T, ref[idx] @ ref[neg].T
        cpos, cneg = cur[idx] @ cur[pos].T, cur[idx] @ cur[neg].T
        r_adv = lme(rpos, args.tau) - lme(rneg, args.tau)
        c_adv = lme(cpos, args.tau) - lme(cneg, args.tau)
        rp, cp = row_prob(rpos, rneg, args.tau), row_prob(cpos, cneg, args.tau)
        r_mass, c_mass = float(rp[: len(pos)].sum()), float(cp[: len(pos)].sum())
        group_r, group_c = np.array([r_mass, 1.0 - r_mass]), np.array([c_mass, 1.0 - c_mass])
        full = kl(rp, cp)
        group = kl(group_r, group_c)
        # 该布尔量揭示聚合 Q+ 很高时，是否仍存在负样本超过某个正样本。
        ref_local_misorder = bool(float(rneg.max()) >= float(rpos.min()))
        cur_local_misorder = bool(float(cneg.max()) >= float(cpos.min()))
        rows.append({
            "anchor": int(idx), "pid": int(bio["pid"][idx]),
            "positive_count": int(len(pos)), "negative_count": int(len(neg)),
            "positive_cross_clothes": bool(np.any(bio["clothes"][pos] != bio["clothes"][idx])),
            "cross_camera_negative_count": int(np.sum(bio["camid"][neg] != bio["camid"][idx])),
            "reference_advantage": float(r_adv), "current_advantage": float(c_adv),
            "advantage_drop": float(r_adv - c_adv),
            "reference_positive_mass": r_mass, "current_positive_mass": c_mass,
            "positive_mass_drop": r_mass - c_mass,
            "lrd_a": max(r_mass - c_mass - args.mass_epsilon, 0.0),
            "lrd_b_full_kl": full, "lrd_c_group_kl": group,
            "lrd_b_internal_shape_kl": max(full - group, 0.0),
            "reference_reliable": bool(r_adv > args.reliability),
            "reference_local_misorder": ref_local_misorder,
            "current_local_misorder": cur_local_misorder,
            "reference_positive_sim_min": float(rpos.min()),
            "reference_negative_sim_max": float(rneg.max()),
            "current_positive_sim_min": float(cpos.min()),
            "current_negative_sim_max": float(cneg.max()),
        })
    reliable = [r for r in rows if r["reference_reliable"]]
    def mean(key, subset):
        return float(np.mean([r[key] for r in subset])) if subset else None
    result = {
        "status": "PASS", "run_type": "train_only_lrd_abc_overconstraint_audit",
        "causal_warning": "Bio E60 与 GPC E60 是独立终点，不是相邻训练状态；结果不是单步因果证据。",
        "seed": args.seed, "n_train_images": n, "n_train_pids": int(np.unique(bio["pid"]).size),
        "sample_anchor_count": len(anchors), "valid_anchor_count": len(rows),
        "parameters": {"positive_k": args.positive_k, "negative_k": args.negative_k,
                       "tau": args.tau, "reference_reliability": args.reliability,
                       "mass_epsilon": args.mass_epsilon},
        "feature_sources": {"reference_bio": str(bio_path), "current_gpc": str(gpc_path),
                            "reference_sha256": sha256_file(bio_path), "current_sha256": sha256_file(gpc_path),
                            "path_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest()},
        "summary": {
            "reference_reliable": len(reliable), "reference_reliable_rate": len(reliable) / max(len(rows), 1),
            "reference_local_misorder_count": sum(r["reference_local_misorder"] for r in reliable),
            "current_local_misorder_count": sum(r["current_local_misorder"] for r in reliable),
            "advantage_drop_mean_reliable": mean("advantage_drop", reliable),
            "positive_mass_drop_mean_reliable": mean("positive_mass_drop", reliable),
            "lrd_a_active_count_reliable": sum(r["lrd_a"] > 0 for r in reliable),
            "lrd_b_nonzero_count_reliable": sum(r["lrd_b_full_kl"] > 1e-8 for r in reliable),
            "lrd_c_nonzero_count_reliable": sum(r["lrd_c_group_kl"] > 1e-8 for r in reliable),
            "full_kl_mean_reliable": mean("lrd_b_full_kl", reliable),
            "group_kl_mean_reliable": mean("lrd_c_group_kl", reliable),
            "internal_shape_kl_mean_reliable": mean("lrd_b_internal_shape_kl", reliable),
            "full_kl_nonzero_mass_not_drop_count": sum(r["lrd_b_full_kl"] > 0.01 and r["positive_mass_drop"] <= 0 for r in reliable),
            "full_kl_internal_fraction_mean_reliable": mean("lrd_b_internal_shape_kl", reliable) / max(mean("lrd_b_full_kl", reliable) or 1e-12, 1e-12),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fields = list(rows[0]) if rows else ["anchor"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "PASS", "seed": args.seed, "valid": len(rows), "reliable": len(reliable), "json": str(args.output), "csv": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
