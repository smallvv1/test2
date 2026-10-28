#!/usr/bin/env python3
"""基于逐节点特征的固定多 PID 局部侵入审计。

候选负 PID 完全来自运行前固定的 groups；脚本不在未来节点重建 KNN。每个
anchor→negative PID 只生成一条关系，多个同 PID 图像先取固定候选中的 max
similarity，不重复计数为多个身份。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_groups(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = data["groups"]
    if data.get("groups_sha256") and data["groups_sha256"] != digest(groups):
        raise RuntimeError(f"groups hash mismatch: {path}")
    return groups


def state(prev: float, cur: float) -> str:
    if prev > 0 and cur > 0:
        return "correct_to_correct"
    if prev > 0 and cur <= 0:
        return "correct_to_wrong"
    if prev <= 0 and cur > 0:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epsilon", type=float, default=0.0)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    bank = np.load(args.features, allow_pickle=False)
    features = bank["features"].astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
    updates = [int(x) for x in bank["updates"]]
    index_to_pos = {int(x): i for i, x in enumerate(bank["image_index"])}
    pid = {int(idx): int(p) for idx, p in zip(bank["image_index"], bank["pid"])}
    clothes = {int(idx): int(c) for idx, c in zip(bank["image_index"], bank["clothes"])}
    camid = {int(idx): int(c) for idx, c in zip(bank["image_index"], bank["camid"])}
    groups = load_groups(args.groups)
    if not set(updates) == {0, 20, 50, 100, 200, 400}:
        raise RuntimeError(f"节点不完整: {updates}")
    rows = []
    for g in groups:
        anchor = int(g["anchor"]); ai = index_to_pos[anchor]
        pos_idx = [int(x) for x in g["positives"] if int(x) in index_to_pos]
        neg_idx = [int(x) for x in g["negatives"] if int(x) in index_to_pos and pid[int(x)] != pid[anchor]]
        by_pid = {}
        for n in neg_idx:
            by_pid.setdefault(pid[n], []).append(n)
        if not pos_idx or not by_pid:
            continue
        positive_cross = any(clothes[p] != clothes[anchor] for p in pos_idx)
        negative_pid_items = sorted(by_pid.items())
        for c, negs in negative_pid_items:
            for prev_pos, target_pos in zip([updates[i] for i in range(len(updates)-1)], [updates[i] for i in range(1, len(updates))]):
                pi = index_to_pos
                pprev = features[updates.index(prev_pos)]
                ptarget = features[updates.index(target_pos)]
                pos_s_prev = pprev[ai] @ pprev[[pi[x] for x in pos_idx]].T
                pos_s_target = ptarget[ai] @ ptarget[[pi[x] for x in pos_idx]].T
                neg_s_prev = pprev[ai] @ pprev[[pi[x] for x in negs]].T
                neg_s_target = ptarget[ai] @ ptarget[[pi[x] for x in negs]].T
                u_prev = float(np.quantile(pos_s_prev, 0.25))
                u_target = float(np.quantile(pos_s_target, 0.25))
                v_prev = float(np.max(neg_s_prev)); v_target = float(np.max(neg_s_target))
                m_prev = u_prev - v_prev; m_target = u_target - v_target
                rows.append({
                    "seed": args.seed, "anchor": anchor, "anchor_pid": pid[anchor], "negative_pid": c,
                    "prev_update": prev_pos, "target_update": target_pos,
                    "positive_indices": ";".join(map(str, pos_idx)), "negative_indices": ";".join(map(str, negs)),
                    "positive_count": len(pos_idx), "negative_image_count": len(negs),
                    "positive_cross_clothes": positive_cross,
                    "negative_cross_clothes_count": sum(clothes[n] != clothes[anchor] for n in negs),
                    "negative_cross_camera_count": sum(camid[n] != camid[anchor] for n in negs),
                    "positive_cross_camera_count": sum(camid[p] != camid[anchor] for p in pos_idx),
                    "reference_positive_q25": u_prev, "target_positive_q25": u_target,
                    "reference_negative_max": v_prev, "target_negative_max": v_target,
                    "m_prev": m_prev, "m_target": m_target, "delta_margin": m_target - m_prev,
                    "state": state(m_prev, m_target),
                    "new_error_pid": bool(m_prev > 0 and m_target <= 0),
                    "recovered_pid": bool(m_prev <= 0 and m_target > 0),
                    "relation_sha256": g.get("relation_sha256", ""),
                })
    fields = list(rows[0]) if rows else []
    with (args.output_dir / f"pid_level_rows_seed{args.seed}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    by_interval = {}
    for r in rows:
        key = f"{r['prev_update']}->{r['target_update']}"
        by_interval.setdefault(key, []).append(r)
    summary = {
        "status": "PASS_FIXED_MULTI_PID_AUDIT" if rows else "NO_VALID_RELATIONS",
        "seed": args.seed, "groups": len(groups), "pid_level_rows": len(rows),
        "anchor_count": len({r["anchor"] for r in rows}), "negative_pid_count": len({r["negative_pid"] for r in rows}),
        "new_error_pid_events": sum(r["new_error_pid"] for r in rows),
        "recovered_pid_events": sum(r["recovered_pid"] for r in rows),
        "state_counts": {k: sum(r["state"] == k for r in rows) for k in ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")},
        "intervals": {k: {"rows": len(v), "new_error_pid": sum(x["new_error_pid"] for x in v), "recovered_pid": sum(x["recovered_pid"] for x in v)} for k, v in sorted(by_interval.items())},
        "candidate_selection": "固定 groups；不在未来节点重选负 PID；同 PID 多图像按 max similarity 聚合",
        "official_query_gallery_used": False, "optimizer_step_executed": False,
        "features": str(args.features), "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "groups": str(args.groups), "groups_sha256": digest(groups),
    }
    (args.output_dir / f"pid_level_summary_seed{args.seed}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
