#!/usr/bin/env python3
"""P2/P5 软侵入定义一致性与漏检归因。

只读取 LTCC train-only 冻结特征。P2 使用 Q25 正样本阈值和逐负 PID
sigmoid 平均；P5 使用正负集合 log-mean-exp 差的 sigmoid。两者不混合，
也不接入训练 processor。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ndp_risk_common import assert_aligned, load_feature_bank, normalize


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def logmeanexp(x: np.ndarray, tau: float) -> float:
    z = np.asarray(x, dtype=np.float64) / float(tau)
    m = float(np.max(z))
    return m + math.log(float(np.exp(z - m).mean()))


def p2_score(pos: np.ndarray, neg: np.ndarray, tau: float) -> tuple[float, float]:
    u = float(np.quantile(pos, 0.25))
    value = float(np.mean(sigmoid((neg - u) / tau)))
    return u, value


def p5_score(pos: np.ndarray, neg: np.ndarray, tau: float) -> float:
    return float(sigmoid(np.asarray([logmeanexp(neg, tau) - logmeanexp(pos, tau)]))[0])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def parse_indices(text: str) -> np.ndarray:
    return np.asarray([int(x) for x in str(text).split(";") if x != ""], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs=2, required=True)
    ap.add_argument("--reference", type=Path, nargs=2, required=True)
    ap.add_argument("--current", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=0.07)
    args = ap.parse_args()
    if args.tau <= 0:
        raise SystemExit("tau 必须大于0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for seed, row_path, ref_path, cur_path in zip((1234, 4321), args.rows, args.reference, args.current):
        ref_bank = load_feature_bank(ref_path)
        cur_bank = load_feature_bank(cur_path)
        assert_aligned(ref_bank, cur_bank)
        ref = normalize(ref_bank["features"])
        cur = normalize(cur_bank["features"])
        seed_rows: list[dict] = []
        with row_path.open(encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                anchor = int(raw["anchor"])
                pos_idx = parse_indices(raw["positive_indices"])
                neg_idx = parse_indices(raw["negative_representatives"])
                neg_pids = parse_indices(raw["negative_pids"])
                r_pos = ref[pos_idx] @ ref[anchor]
                c_pos = cur[pos_idx] @ cur[anchor]
                r_neg = ref[neg_idx] @ ref[anchor]
                c_neg = cur[neg_idx] @ cur[anchor]
                r_u, r_p2 = p2_score(r_pos, r_neg, args.tau)
                c_u, c_p2 = p2_score(c_pos, c_neg, args.tau)
                r_p5 = p5_score(r_pos, r_neg, args.tau)
                c_p5 = p5_score(c_pos, c_neg, args.tau)
                r_edge = float(np.min(r_pos) - np.max(r_neg))
                c_edge = float(np.min(c_pos) - np.max(c_neg))
                edge_flip = bool(r_edge > 0 and c_edge <= 0)
                dangerous = np.flatnonzero(c_neg >= float(np.min(c_pos)))
                row = {
                    "seed": seed,
                    "anchor": anchor,
                    "image_path": str(ref_bank["image_path"][anchor]),
                    "pid": int(ref_bank["pid"][anchor]),
                    "clothes": int(ref_bank["clothes"][anchor]),
                    "camid": int(ref_bank["camid"][anchor]),
                    "positive_indices": raw["positive_indices"],
                    "negative_indices": raw["negative_representatives"],
                    "negative_pids": raw["negative_pids"],
                    "positive_paths": json.dumps([str(x) for x in ref_bank["image_path"][pos_idx]], ensure_ascii=False),
                    "negative_paths": json.dumps([str(x) for x in ref_bank["image_path"][neg_idx]], ensure_ascii=False),
                    "reference_positive_scores": json.dumps(r_pos.tolist()),
                    "current_positive_scores": json.dumps(c_pos.tolist()),
                    "reference_negative_scores": json.dumps(r_neg.tolist()),
                    "current_negative_scores": json.dumps(c_neg.tolist()),
                    "reference_edge": r_edge,
                    "current_edge": c_edge,
                    "edge_flip": edge_flip,
                    "reference_positive_threshold_q25": r_u,
                    "current_positive_threshold_q25": c_u,
                    "positive_threshold_change": c_u - r_u,
                    "p2_reference": r_p2,
                    "p2_current": c_p2,
                    "p2_delta": c_p2 - r_p2,
                    "p5_reference": r_p5,
                    "p5_current": c_p5,
                    "p5_delta": c_p5 - r_p5,
                    "dangerous_negative_count": int(len(dangerous)),
                    "safe_negative_count": int(len(c_neg) - len(dangerous)),
                    "p2_missed_flip": bool(edge_flip and c_p2 <= r_p2),
                    "p5_missed_flip": bool(edge_flip and c_p5 <= r_p5),
                    "reference_wrong_boundary": bool(r_edge <= 0),
                    "p2_mass_up_but_edge_flip": bool(edge_flip and c_p2 > r_p2),
                    "positive_mass_up_but_edge_flip": bool(edge_flip and c_u > r_u),
                }
                seed_rows.append(row)
                all_rows.append(row)
        flips = [r for r in seed_rows if r["edge_flip"]]
        summaries.append({
            "seed": seed,
            "rows": len(seed_rows),
            "edge_flip_count": len(flips),
            "p2_missed_flip_count": sum(r["p2_missed_flip"] for r in flips),
            "p5_missed_flip_count": sum(r["p5_missed_flip"] for r in flips),
            "p2_mass_up_but_edge_flip_count": sum(r["p2_mass_up_but_edge_flip"] for r in flips),
            "edge_flip_positive_threshold_up_count": sum(r["positive_mass_up_but_edge_flip"] for r in flips),
            "mean_p2_delta": float(np.mean([r["p2_delta"] for r in seed_rows])),
            "mean_p5_delta": float(np.mean([r["p5_delta"] for r in seed_rows])),
            "spearman_p2_p5_delta": spearman(np.asarray([r["p2_delta"] for r in seed_rows]), np.asarray([r["p5_delta"] for r in seed_rows])),
            "dangerous_negative_count_total_on_flips": int(sum(r["dangerous_negative_count"] for r in flips)),
            "positive_mass_up_but_edge_flip_examples": [r["anchor"] for r in flips if r["positive_mass_up_but_edge_flip"]][:20],
            "reference_feature_sha256": ref_bank["sha256"],
            "current_feature_sha256": cur_bank["sha256"],
        })
    fields = list(all_rows[0]) if all_rows else []
    with (args.output_dir / "p2_p5_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    (args.output_dir / "p2_p5_summary.json").write_text(json.dumps({
        "status": "PASS",
        "tau": args.tau,
        "p2_definition": "mean_c sigmoid((sim(anchor,negative_c)-Q25(sim(anchor,positive)))/tau), fixed reference negative representatives",
        "p5_definition": "sigmoid(logmeanexp(negative/tau)-logmeanexp(positive/tau))",
        "edge_definition": "reference min positive - max negative > 0 and current <= 0",
        "rows": summaries,
        "official_query_gallery_used": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(all_rows), "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
