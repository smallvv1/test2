#!/usr/bin/env python3
"""从 update0 train-only 特征构建真正按负 PID 去重的局部关系图。

输出代表性分层集合与 update0 困难集合；两者使用相同 anchor，但负 PID 选择规则
不同，绝不混合统计。相似度计算按 anchor 分块，避免分配完整 N×N 矩阵。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def choose_anchors(pid, clothes, camid, max_anchors):
    by_pid_cloth = defaultdict(list)
    by_pid = defaultdict(list)
    for i, (p, c) in enumerate(zip(pid, clothes)):
        by_pid_cloth[(int(p), int(c))].append(i)
        by_pid[int(p)].append(i)
    anchors = []
    # 第一轮覆盖 PID×衣服；随后 round-robin 补足，避免困难近邻集合只覆盖少数 PID。
    for key in sorted(by_pid_cloth):
        anchors.append(by_pid_cloth[key][0])
    if len(anchors) > max_anchors:
        # 每个 PID 至少保留一个，再按索引截断。
        first = [by_pid[p][0] for p in sorted(by_pid)]
        rest = [x for x in sorted(set(anchors)) if x not in set(first)]
        anchors = first + rest
    if len(anchors) < max_anchors:
        cursors = {p: 1 for p in by_pid}
        while len(anchors) < max_anchors:
            changed = False
            for p in sorted(by_pid):
                values = by_pid[p]
                if cursors[p] < len(values):
                    x = values[cursors[p]]; cursors[p] += 1
                    if x not in anchors:
                        anchors.append(x); changed = True
                        if len(anchors) >= max_anchors:
                            break
            if not changed:
                break
    return sorted(anchors[:max_anchors])


def block_similarity_top(features, anchors, block_size=64, topk=20):
    result = {}
    for start in range(0, len(anchors), block_size):
        ids = anchors[start:start + block_size]
        sims = features[ids] @ features.T
        for local, idx in enumerate(ids):
            row = sims[local].copy()
            row[idx] = -np.inf
            k = min(topk, len(row) - 1)
            top = np.argpartition(row, -k)[-k:]
            top = top[np.argsort(row[top])[::-1]]
            result[int(idx)] = [int(x) for x in top]
    return result


def positive_indices(anchor, pid, clothes, camid, max_pos=4):
    p = int(pid[anchor])
    candidates = [i for i, x in enumerate(pid) if int(x) == p and i != anchor]
    candidates.sort(key=lambda i: (int(clothes[i] == clothes[anchor]), int(camid[i] == camid[anchor]), i))
    return candidates[:max_pos]


def representative_negative_pid_images(anchor, pid, clothes, camid, all_pids, k):
    others = [p for p in all_pids if p != int(pid[anchor])]
    if not others:
        return []
    offset = anchor % len(others)
    selected = [others[(offset + j) % len(others)] for j in range(min(k, len(others)))]
    out = []
    for p in selected:
        images = [i for i, x in enumerate(pid) if int(x) == p]
        images.sort(key=lambda i: (int(clothes[i] == clothes[anchor]), int(camid[i] == camid[anchor]), i))
        out.append((p, images[:1]))
    return out


def difficult_negative_pid_images(anchor, features, pid, k):
    sims = features[anchor] @ features.T
    by_pid = {}
    for i, p in enumerate(pid):
        p = int(p)
        if p == int(pid[anchor]):
            continue
        score = float(sims[i])
        if p not in by_pid or score > by_pid[p][0]:
            by_pid[p] = (score, int(i))
    ranked = sorted(by_pid.items(), key=lambda item: (-item[1][0], item[0]))[:k]
    return [(p, [item[1]]) for p, item in ranked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-anchors", type=int, default=512)
    ap.add_argument("--k-pid", nargs="+", type=int, default=[1, 3, 5, 10])
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    bank = np.load(args.features, allow_pickle=False)
    features = bank["features"].astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    image_index = [int(x) for x in bank["image_index"]]
    pid = bank["pid"].astype(np.int64); clothes = bank["clothes"].astype(np.int64); camid = bank["camid"].astype(np.int64)
    if len(features) != len(pid) or len(features) != len(image_index):
        raise RuntimeError("feature metadata length mismatch")
    anchors = choose_anchors(pid, clothes, camid, args.max_anchors)
    all_pids = sorted({int(x) for x in pid})
    top_anchor = block_similarity_top(features, anchors, topk=20)
    # 旧实现只缓存了 anchor 的查询结果；反向查询的候选负图像往往不在
    # anchors 中，不能用 dict 缺失直接等同于“非互为近邻”。先收集所有
    # 候选负图像，再对它们真实计算 Top-20。
    candidate_reverse = set()
    for anchor in anchors:
        for neg_pid, negs in representative_negative_pid_images(anchor, pid, clothes, camid, all_pids, max(args.k_pid)):
            candidate_reverse.update(negs)
        for neg_pid, negs in difficult_negative_pid_images(anchor, features, pid, max(args.k_pid)):
            candidate_reverse.update(negs)
    top_reverse = block_similarity_top(features, sorted(candidate_reverse), topk=20)
    records = []
    for graph_type in ("representative", "difficult"):
        for k in sorted(set(args.k_pid)):
            for anchor_order, anchor in enumerate(anchors):
                positives = positive_indices(anchor, pid, clothes, camid)
                if not positives:
                    continue
                if graph_type == "representative":
                    neg_groups = representative_negative_pid_images(anchor, pid, clothes, camid, all_pids, k)
                else:
                    neg_groups = difficult_negative_pid_images(anchor, features, pid, k)
                for neg_pid, negs in neg_groups:
                    pos_sims = features[anchor] @ features[positives].T
                    neg_sims = features[anchor] @ features[negs].T
                    u = float(np.quantile(pos_sims, 0.25))
                    v = float(np.max(neg_sims))
                    if anchor not in top_anchor or negs[0] not in top_reverse:
                        raise RuntimeError("反向 KNN 查询缓存缺失，禁止默认判为 False")
                    bidirectional = bool(negs[0] in top_anchor[anchor] and anchor in top_reverse[negs[0]])
                    relation = {
                        "graph_type": graph_type, "k_pid": int(k), "anchor_order": anchor_order,
                        "anchor_image_index": int(image_index[anchor]), "anchor_local_index": int(anchor),
                        "anchor_pid": int(pid[anchor]), "positive_indices": [int(image_index[x]) for x in positives],
                        "positive_local_indices": [int(x) for x in positives], "negative_pid": int(neg_pid),
                        "negative_indices": [int(image_index[x]) for x in negs],
                        "negative_local_indices": [int(x) for x in negs], "reference_positive_q25": u,
                        "reference_negative_max": v, "reference_margin": u - v,
                        "positive_cross_clothes": any(clothes[x] != clothes[anchor] for x in positives),
                        "negative_cross_camera": any(camid[x] != camid[anchor] for x in negs),
                        "bidirectional_image_knn20": bidirectional, "source_update": 0,
                        "source_checkpoint_feature_file": str(args.features),
                    }
                    relation["relation_sha256"] = stable_hash(relation)
                    records.append(relation)
    summary = {
        "status": "PASS_UPDATE0_FIXED_MULTI_PID_GRAPH", "feature_source": str(args.features),
        "feature_sha256": sha256_file(args.features), "source_update": 0, "anchor_count": len(anchors),
        "pid_count": len(all_pids), "image_count": len(features), "k_pid": sorted(set(args.k_pid)),
        "graph_types": ["representative", "difficult"], "record_count": len(records),
        "records_by_type_k": {}, "multi_pid_anchor_by_type_k": {},
        "representative_rule": "round-robin PID×clothes first image, no similarity-based selection",
        "difficult_rule": "update0 max-similarity representative per negative PID",
        "future_nodes_used_for_selection": False, "official_query_gallery_used": False,
        "reverse_knn_query_count": len(candidate_reverse), "reverse_knn_topk_computed": True,
        "optimizer_step_executed": False, "blockwise_similarity": True,
        "elapsed_seconds": time.time() - started,
    }
    for kind in ("representative", "difficult"):
        for k in sorted(set(args.k_pid)):
            subset = [r for r in records if r["graph_type"] == kind and r["k_pid"] == k]
            summary["records_by_type_k"][f"{kind}_k{k}"] = len(subset)
            summary["multi_pid_anchor_by_type_k"][f"{kind}_k{k}"] = len({r["anchor_local_index"] for r in subset if len({x["negative_pid"] for x in subset if x["anchor_local_index"] == r["anchor_local_index"]}) > 1})
    out_json = args.output_dir / "multi_pid_reference_graph.json"
    out_json.write_text(json.dumps({"summary": summary, "relations": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "multi_pid_reference_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
