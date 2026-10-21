#!/usr/bin/env python3
"""P2：多 PID 软侵入风险审计（仅冻结 train-only 特征）。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ndp_risk_common import (
    assert_aligned,
    batched_scores,
    fixed_anchor_indices,
    load_feature_bank,
    normalize,
    positive_indices,
    top_negative_indices,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def pid_representatives(scores: np.ndarray, anchor: int, pids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """每个负 PID 只取一个代表；代表由 reference 相似度确定。"""
    diff = np.flatnonzero(pids != pids[anchor])
    by_pid: dict[int, int] = {}
    for idx in diff[np.argsort(-scores[diff], kind="stable")]:
        pid = int(pids[idx])
        if pid not in by_pid:
            by_pid[pid] = int(idx)
    selected = sorted(by_pid.items(), key=lambda kv: (-float(scores[kv[1]]), kv[0]))[:k]
    return np.asarray([pid for pid, _ in selected], dtype=np.int64), np.asarray([idx for _, idx in selected], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--sample-anchors", type=int, default=1024)
    ap.add_argument("--negative-pids", type=int, default=20)
    ap.add_argument("--positive-count", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--positive-quantile", type=float, default=0.25)
    args = ap.parse_args()
    if args.tau <= 0 or not 0 <= args.positive_quantile <= 1:
        raise SystemExit("tau/positive-quantile 参数非法")
    ref_bank = load_feature_bank(args.reference)
    cur_bank = load_feature_bank(args.current)
    assert_aligned(ref_bank, cur_bank)
    ref = normalize(ref_bank["features"])
    cur = normalize(cur_bank["features"])
    n = len(ref)
    anchors = fixed_anchor_indices(n, args.seed, args.sample_anchors)
    rows = []
    hubness = np.zeros(n, dtype=np.int64)
    selected_by_anchor: dict[int, set[int]] = {}
    for start, ref_scores in batched_scores(ref[anchors], ref):
        cur_scores = cur[anchors[start : start + len(ref_scores)]] @ cur.T
        for local, anchor in enumerate(anchors[start : start + len(ref_scores)]):
            rrow, crow = ref_scores[local], cur_scores[local]
            pos = positive_indices(rrow, int(anchor), ref_bank["pid"], args.positive_count)
            neg_pid, neg_rep = pid_representatives(rrow, int(anchor), ref_bank["pid"], args.negative_pids)
            if len(pos) == 0 or len(neg_rep) == 0:
                continue
            ref_pos = rrow[pos]; cur_pos = crow[pos]
            ref_neg = rrow[neg_rep]; cur_neg = crow[neg_rep]
            u_ref = float(np.quantile(ref_pos, args.positive_quantile))
            u_cur = float(np.quantile(cur_pos, args.positive_quantile))
            p_ref = sigmoid((ref_neg - u_ref) / args.tau)
            p_cur = sigmoid((cur_neg - u_cur) / args.tau)
            i_ref = float(np.mean(p_ref)); i_cur = float(np.mean(p_cur))
            ref_edge = float(np.min(ref_pos) - np.max(ref_neg))
            cur_edge = float(np.min(cur_pos) - np.max(cur_neg))
            top_neg = top_negative_indices(rrow, int(anchor), ref_bank["pid"], args.negative_pids)
            hubness[top_neg] += 1
            selected_by_anchor[int(anchor)] = set(int(x) for x in neg_rep)
            rows.append({
                "anchor": int(anchor),
                "image_path": str(ref_bank["image_path"][anchor]),
                "pid": int(ref_bank["pid"][anchor]),
                "clothes": int(ref_bank["clothes"][anchor]),
                "camid": int(ref_bank["camid"][anchor]),
                "positive_indices": ";".join(map(str, pos.tolist())),
                "negative_pid_count": int(len(neg_pid)),
                "negative_pids": ";".join(map(str, neg_pid.tolist())),
                "negative_representatives": ";".join(map(str, neg_rep.tolist())),
                "positive_cross_clothes_rate": float(np.mean(ref_bank["clothes"][pos] != ref_bank["clothes"][anchor])),
                "u_ref": u_ref,
                "u_cur": u_cur,
                "i_ref": i_ref,
                "i_cur": i_cur,
                "delta_i": i_cur - i_ref,
                "reference_edge": ref_edge,
                "current_edge": cur_edge,
                "edge_drop": ref_edge - cur_edge,
                "edge_flip": bool(ref_edge > 0 and cur_edge <= 0),
                "reference_positive_correct": bool(ref_edge > 0),
            })

    row_by_anchor = {int(row["anchor"]): row for row in rows}
    for row in rows:
        reverse = []
        for rep in map(int, row["negative_representatives"].split(";")):
            other = row_by_anchor.get(rep)
            if other is not None and int(row["pid"]) in set(map(int, other["negative_pids"].split(";"))):
                reverse.append(float(other["delta_i"]))
        row["reverse_available_count"] = len(reverse)
        row["reverse_delta_i_mean"] = float(np.mean(reverse)) if reverse else None
        row["bidirectional_delta_i"] = float((row["delta_i"] + np.mean(reverse)) / 2) if reverse else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"seed{args.seed}_intrusion_rows.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "PASS",
        "seed": args.seed,
        "reference": str(args.reference),
        "current": str(args.current),
        "n_train_images": n,
        "n_train_pids": int(np.unique(ref_bank["pid"]).size),
        "sample_anchor_count": len(anchors),
        "valid_anchor_count": len(rows),
        "k_negative_pids": args.negative_pids,
        "positive_quantile": args.positive_quantile,
        "tau": args.tau,
        "candidate_rule": "固定正样本与每个负PID的reference代表；current不重新挑选关系",
        "official_query_gallery_used": False,
        "risk": {
            "soft_intrusion_delta_i_mean": float(np.mean([r["delta_i"] for r in rows])) if rows else None,
            "soft_intrusion_delta_i_positive_rate": float(np.mean([r["delta_i"] > 0 for r in rows])) if rows else None,
            "edge_flip_count": int(sum(r["edge_flip"] for r in rows)),
            "edge_reliable_count": int(sum(r["reference_positive_correct"] for r in rows)),
            "edge_drop_mean": float(np.mean([r["edge_drop"] for r in rows])) if rows else None,
            "bidirectional_rows": int(sum(r["reverse_available_count"] > 0 for r in rows)),
        },
        "hubness": {
            "max": int(hubness.max()) if len(hubness) else 0,
            "nonzero_count": int(np.count_nonzero(hubness)),
            "top10_sum": int(np.sort(hubness)[-10:].sum()) if len(hubness) else 0,
            "top10_indices": np.argsort(-hubness, kind="stable")[:10].astype(int).tolist(),
        },
        "rows_csv": str(csv_path),
        "temporal_validity": "INCONCLUSIVE_ENDPOINT_SHIFT_ONLY",
    }
    (args.output_dir / f"seed{args.seed}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "seed": args.seed, "rows": len(rows), "risk": summary["risk"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
