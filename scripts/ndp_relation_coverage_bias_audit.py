#!/usr/bin/env python3
"""固定关系覆盖与抽样偏差审计；不重新选择未来邻居。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    bank = np.load(args.features, allow_pickle=False)
    idx = [int(x) for x in bank["image_index"]]
    pos = {x: i for i, x in enumerate(idx)}
    pid = {x: int(p) for x, p in zip(idx, bank["pid"])}
    clothes = {x: int(c) for x, c in zip(idx, bank["clothes"])}
    cams = {x: int(c) for x, c in zip(idx, bank["camid"])}
    gd = json.loads(args.groups.read_text(encoding="utf-8")); groups = gd["groups"]
    if gd.get("groups_sha256") and gd["groups_sha256"] != digest(groups):
        raise RuntimeError("groups hash mismatch")
    stats = []
    for g in groups:
        a = int(g["anchor"]); ps = [int(x) for x in g["positives"] if int(x) in pos]
        ns = [int(x) for x in g["negatives"] if int(x) in pos and pid[int(x)] != pid[a]]
        stats.append({"anchor": a, "anchor_pid": pid[a], "positive_count": len(ps), "negative_image_count": len(ns),
                      "negative_pid_count": len({pid[x] for x in ns}),
                      "positive_cross_clothes": sum(clothes[x] != clothes[a] for x in ps),
                      "positive_cross_camera": sum(cams[x] != cams[a] for x in ps),
                      "negative_cross_camera": sum(cams[x] != cams[a] for x in ns)})
    summary = {
        "status": "PASS_FIXED_COVERAGE_BIAS_AUDIT",
        "groups": len(groups), "anchor_count": len({s["anchor"] for s in stats}),
        "pid_count": len({s["anchor_pid"] for s in stats}), "image_count": len(idx),
        "negative_pid_total": sum(s["negative_pid_count"] for s in stats),
        "negative_pid_mean_per_anchor": float(np.mean([s["negative_pid_count"] for s in stats])) if stats else None,
        "negative_pid_max_per_anchor": max((s["negative_pid_count"] for s in stats), default=0),
        "multi_negative_pid_anchor_count": sum(s["negative_pid_count"] > 1 for s in stats),
        "cross_clothes_positive_anchor_count": sum(s["positive_cross_clothes"] > 0 for s in stats),
        "cross_camera_positive_anchor_count": sum(s["positive_cross_camera"] > 0 for s in stats),
        "cross_camera_negative_anchor_count": sum(s["negative_cross_camera"] > 0 for s in stats),
        "representative_stratum_available": False,
        "difficulty_stratum_available": True,
        "bias_warning": "当前 groups 是历史困难/固定关系集合；没有独立按 PID×衣服×相机预注册的代表性集合，不能外推全量 LTCC。",
        "group_count": len(groups), "group_path": str(args.groups), "groups_sha256": digest(groups),
        "features": str(args.features), "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "official_query_gallery_used": False,
    }
    (args.output_dir / "relation_coverage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "relation_coverage_rows.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
