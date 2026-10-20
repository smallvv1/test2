#!/usr/local/bin/python3
"""NDP reference-relation audit on frozen LTCC train-only feature banks.

This is an offline geometry audit.  It does not instantiate a model, call
autograd, read query/gallery data, or update a checkpoint.  The reference
bank is the Bio-only frozen state and the current bank is the fixed Bio+GPC
state for each registered seed.  The thresholds are pre-registered here and
are not selected from any official evaluation result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/DIFFER")
RUN_TIME = "20261020_155558"
OUT = ROOT / "outputs" / f"ndp_local_intrusion_{RUN_TIME}"
RELATION_OUT = OUT / "reference_relations"
K_VALUES = (5, 10, 20)
TAU_REF = 0.05
EPSILON = 0.02


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_bank(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def best_cross_positive(z: np.ndarray, pid: np.ndarray, clothes: np.ndarray) -> np.ndarray:
    """Choose one fixed, reference-ranked same-PID cross-clothes positive."""
    out = np.full(len(pid), -1, dtype=np.int32)
    for value in np.unique(pid):
        indices = np.flatnonzero(pid == value)
        if len(indices) < 2:
            continue
        scores = z[indices] @ z[indices].T
        for local, anchor in enumerate(indices):
            candidates = indices[(clothes[indices] != clothes[anchor]) & (indices != anchor)]
            if len(candidates):
                local_candidates = np.flatnonzero(np.isin(indices, candidates))
                out[anchor] = int(indices[local_candidates[np.argmax(scores[local, local_candidates])]])
    return out


def per_pid_summary(pid: np.ndarray, rows: list[dict[str, object]], field: str) -> list[dict[str, object]]:
    values: dict[int, list[float]] = {}
    for row in rows:
        value = row.get(field)
        if value is not None:
            values.setdefault(int(row["pid"]), []).append(float(value))
    return [{"pid": k, "count": len(v), f"mean_{field}": float(np.mean(v))} for k, v in sorted(values.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", default=RUN_TIME)
    args = parser.parse_args()
    if args.run_time != RUN_TIME:
        raise SystemExit(f"本脚本固定复用 RUN_TIME={RUN_TIME}，收到 {args.run_time}")

    rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "run_time": RUN_TIME,
        "purpose": "冻结 Bio 参考状态到 Bio+GPC 当前状态的 train-only 局部关系审计",
        "reference_role": "Bio-only frozen feature bank; stopped-gradient reference in candidate formula",
        "current_role": "fixed Bio+GPC GPC-v1 feature bank",
        "K": list(K_VALUES),
        "tau_ref": TAU_REF,
        "epsilon": EPSILON,
        "query_gallery_used": False,
        "official_test_used": False,
        "model_training_started": False,
    }

    for seed in (1234, 4321):
        seed_dir = OUT / "feature_banks" / f"seed{seed}"
        ref = load_bank(seed_dir / "bio_feature_bank.npz")
        cur = load_bank(seed_dir / "gpc_feature_bank.npz")
        z_ref = np.asarray(ref["features"], dtype=np.float32)
        z_cur = np.asarray(cur["features"], dtype=np.float32)
        pid = np.asarray(ref["pid"], dtype=np.int64)
        clothes = np.asarray(ref["clothes"], dtype=np.int64)
        if not (np.array_equal(pid, cur["pid"]) and np.array_equal(clothes, cur["clothes"]) and np.array_equal(ref["image_path"], cur["image_path"])):
            raise RuntimeError(f"seed{seed} reference/current alignment failed")

        positive = best_cross_positive(z_ref, pid, clothes)
        graph_path = OUT / "neighbor_graphs" / f"seed{seed}_bio_negative_only_top20.npz"
        graph = np.load(graph_path, allow_pickle=False)["indices"]
        seed_manifest = {"seed": seed, "images": int(len(pid)), "pids": int(np.unique(pid).size), "rows": {}}

        for k in K_VALUES:
            metric_rows: list[dict[str, object]] = []
            for anchor in range(len(pid)):
                p = int(positive[anchor])
                if p < 0:
                    continue
                n_indices = graph[anchor, :k]
                ref_pos = float(z_ref[anchor] @ z_ref[p])
                cur_pos = float(z_cur[anchor] @ z_cur[p])
                for n in n_indices:
                    n = int(n)
                    ref_neg = float(z_ref[anchor] @ z_ref[n])
                    cur_neg = float(z_cur[anchor] @ z_cur[n])
                    g_ref = ref_pos - ref_neg
                    g_cur = cur_pos - cur_neg
                    metric_rows.append({
                        "seed": seed,
                        "K": k,
                        "anchor": anchor,
                        "pid": int(pid[anchor]),
                        "positive": p,
                        "negative": n,
                        "positive_clothes_cross": bool(clothes[anchor] != clothes[p]),
                        "g_ref": g_ref,
                        "g_current": g_cur,
                        "reference_reliable": bool(g_ref > TAU_REF),
                        "reference_correct": bool(g_ref > 0.0),
                        "current_order_correct": bool(g_cur > 0.0),
                        "originally_correct_but_broken": bool(g_ref > 0.0 and g_cur <= 0.0),
                        "reliable_violation": bool(g_ref > TAU_REF and (g_ref - g_cur) > EPSILON),
                        "ndp_hinge_value": float(max(0.0, g_ref - g_cur - EPSILON)),
                        "originally_wrong_but_improved": bool(g_ref <= 0.0 and g_cur > 0.0),
                    })

            if not metric_rows:
                raise RuntimeError(f"seed{seed} K={k} produced no legal train-only relation")
            ref_reliable = np.array([r["reference_reliable"] for r in metric_rows], dtype=bool)
            ref_correct = np.array([r["reference_correct"] for r in metric_rows], dtype=bool)
            broken = np.array([r["originally_correct_but_broken"] for r in metric_rows], dtype=bool)
            violating = np.array([r["reliable_violation"] for r in metric_rows], dtype=bool)
            improved = np.array([r["originally_wrong_but_improved"] for r in metric_rows], dtype=bool)
            hinge = np.array([r["ndp_hinge_value"] for r in metric_rows], dtype=np.float64)
            summary = {
                "seed": seed,
                "K": k,
                "triplet_rows": len(metric_rows),
                "anchor_count": int(len(np.unique([r["anchor"] for r in metric_rows]))),
                "reference_reliable_rows": int(ref_reliable.sum()),
                "reference_reliable_rate": float(ref_reliable.mean()),
                "reference_correct_rows": int(ref_correct.sum()),
                "reference_correct_rate": float(ref_correct.mean()),
                "originally_correct_but_broken_rows": int(broken.sum()),
                "originally_correct_but_broken_rate_all": float(broken.mean()),
                "originally_correct_but_broken_rate_reliable": float(broken[ref_reliable].mean()) if ref_reliable.any() else None,
                "reliable_violation_rows": int(violating.sum()),
                "reliable_violation_rate": float(violating[ref_reliable].mean()) if ref_reliable.any() else None,
                "mean_ndp_hinge_all": float(hinge.mean()),
                "mean_ndp_hinge_reliable": float(hinge[ref_reliable].mean()) if ref_reliable.any() else None,
                "originally_wrong_but_improved_rows": int(improved.sum()),
                "originally_wrong_but_improved_rate": float(improved.mean()),
                "positive_cross_clothes_rate": float(np.mean([r["positive_clothes_cross"] for r in metric_rows])),
            }
            rows.append(summary)
            write_csv(RELATION_OUT / f"seed{seed}_K{k}_relations.csv", metric_rows)
            write_csv(RELATION_OUT / f"seed{seed}_K{k}_per_pid.csv", per_pid_summary(pid, metric_rows, "ndp_hinge_value"))
            seed_manifest["rows"][str(k)] = {"count": len(metric_rows), "sha256": sha256(RELATION_OUT / f"seed{seed}_K{k}_relations.csv")}

        manifest[str(seed)] = seed_manifest

    write_csv(RELATION_OUT / "summary.csv", rows)
    manifest["summary_sha256"] = sha256(RELATION_OUT / "summary.csv")
    write_json(RELATION_OUT / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "summary": str(RELATION_OUT / "summary.csv"), "manifest": str(RELATION_OUT / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
