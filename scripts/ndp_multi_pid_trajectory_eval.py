#!/usr/bin/env python3
"""在固定 update0 多 PID 关系上评估同一 continuation 的后续节点。

关系只从 update0 graph JSON 读取；20/50/100/200/400 的特征只用于评估，
不会重新挑选邻居，也不执行 optimizer.step。输出按关系、节点和相邻节点
转移分开保存，便于后续审计真实 PID 侵入与双侧边界。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
sys.path.insert(0, str(ROOT / "scripts"))
import ndp_real_trajectory_activation as act  # noqa: E402
import gpc_v2_train as v2  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def as_model_state(value):
    if isinstance(value, dict) and "model" in value and isinstance(value["model"], dict):
        return value["model"]
    return value


def load_graph(path: Path, graph_type: str, k_pid: int, max_anchors: int | None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload["relations"] if r["graph_type"] == graph_type and int(r["k_pid"]) == k_pid]
    rows.sort(key=lambda r: (int(r["anchor_order"]), int(r["negative_pid"])))
    if max_anchors is not None:
        chosen = []
        seen = set()
        for row in rows:
            a = int(row["anchor_local_index"])
            if a not in seen and len(seen) >= max_anchors:
                continue
            seen.add(a)
            chosen.append(row)
        rows = chosen
    if not rows:
        raise RuntimeError("固定 graph 没有满足 graph_type/k_pid/max_anchors 的关系")
    # 同一 anchor/PID 只能出现一条固定关系；否则会把图片重复误计为 PID。
    keys = [(int(r["anchor_local_index"]), int(r["negative_pid"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("固定 graph 存在重复 anchor-negative PID 关系")
    return payload, rows


def encode(model, dataset, indices, transform_test, device, batch_size: int = 16):
    from PIL import Image
    values = {}
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        tensors, cams = [], []
        for idx in chunk:
            with Image.open(Path(str(dataset.train[idx]["image_path"]))) as image:
                tensors.append(transform_test(image.convert("RGB")))
            cams.append(int(dataset.train[idx]["camid"]))
        images = torch.stack(tensors, dim=0).to(device, non_blocking=False)
        cam = torch.tensor(cams, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True):
            out = model(images, cam_label=cam)
        feat = out[1][0] if isinstance(out, tuple) else out
        if feat.ndim != 2 or feat.shape[1] != 1024:
            raise RuntimeError(f"特征形状异常: {tuple(feat.shape)}")
        feat = torch.nn.functional.normalize(feat.float(), dim=-1).detach().cpu().numpy()
        for idx, value in zip(chunk, feat):
            values[int(idx)] = value.astype(np.float32)
    return values


def relation_metrics(row, feats):
    a = feats[int(row["anchor_local_index"])]
    pos = np.asarray([feats[int(x)] for x in row["positive_local_indices"]], dtype=np.float32)
    neg = np.asarray([feats[int(x)] for x in row["negative_local_indices"]], dtype=np.float32)
    pos_s = pos @ a
    neg_s = neg @ a
    u = float(np.quantile(pos_s, 0.25))
    v = float(np.max(neg_s))
    return {
        "positive_q25": u,
        "positive_mean": float(np.mean(pos_s)),
        "positive_min": float(np.min(pos_s)),
        "negative_max": v,
        "margin_q25": u - v,
        "edge_margin": float(np.min(pos_s) - np.max(neg_s)),
        "positive_similarities": [float(x) for x in pos_s],
        "negative_similarities": [float(x) for x in neg_s],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--feature-update0", type=Path, required=True)
    ap.add_argument("--trajectory-dir", type=Path, required=True)
    ap.add_argument("--resume-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--graph-type", choices=("representative", "difficult"), default="difficult")
    ap.add_argument("--k-pid", type=int, default=5)
    ap.add_argument("--max-anchors", type=int, default=512)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("需要两张 PPU；禁止 CPU model fallback")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    class Args:
        output = OLD_OUTPUT
        run_time = "20261015_155822"
        arm = "B0"
        stage = "two_seed"
        seed = args.seed
        epochs = 60
        full_data = False

    split = json.loads((OLD_OUTPUT / "development_split_manifest.json").read_text(encoding="utf-8"))
    protocol = v2.install_protocol(Args, split, args.output_dir, None)
    cfg = v2.p7.configure()
    dataset = v2.base.build_dataset(cfg)
    _train_transform, transform_test = v2.base.build_img_transforms(cfg)
    payload, relations = load_graph(args.graph, args.graph_type, args.k_pid, args.max_anchors)
    all_indices = sorted({int(x) for r in relations for x in [r["anchor_local_index"], *r["positive_local_indices"], *r["negative_local_indices"]]})
    bank0 = np.load(args.feature_update0, allow_pickle=False)
    if list(bank0["image_index"].astype(int)) != list(range(len(dataset.train))):
        raise RuntimeError("update0 full feature 的 image_index 不是完整 train 顺序")
    features_by_update = {0: {int(i): np.asarray(f, dtype=np.float32) for i, f in zip(bank0["image_index"], bank0["features"])}}
    source_path = args.resume_dir / "rank0_state_epoch060.pt"
    source = act.restore_numpy_pickle(source_path)
    source_state = as_model_state(source)
    nodes = [(0, source_state, str(source_path))]
    for path in sorted(args.trajectory_dir.glob("node_update*/model.pth")):
        update = int(path.parent.name.replace("node_update", ""))
        nodes.append((update, as_model_state(act.restore_numpy_pickle(path)), str(path)))
    nodes = sorted(nodes, key=lambda x: x[0])
    model = act.load_model(cfg, source_state, torch.device("cuda:0"))
    model.eval()
    for update, state, state_path in nodes:
        if update == 0:
            continue
        model.load_state_dict(state, strict=True)
        model.eval()
        features_by_update[update] = encode(model, dataset, all_indices, transform_test, torch.device("cuda:0"))
        torch.cuda.empty_cache()

    node_rows, transition_rows = [], []
    for update in sorted(features_by_update):
        feats = features_by_update[update]
        for idx, row in enumerate(relations):
            m = relation_metrics(row, feats)
            node_rows.append({
                "seed": args.seed, "graph_type": args.graph_type, "k_pid": args.k_pid,
                "relation_index": idx, "anchor_local_index": int(row["anchor_local_index"]),
                "anchor_image_index": int(row["anchor_image_index"]), "anchor_pid": int(row["anchor_pid"]),
                "negative_pid": int(row["negative_pid"]), "update": update,
                "positive_local_indices": json.dumps([int(x) for x in row["positive_local_indices"]]),
                "negative_local_indices": json.dumps([int(x) for x in row["negative_local_indices"]]),
                "positive_count": len(row["positive_local_indices"]), "negative_image_count": len(row["negative_local_indices"]),
                "reference_update": 0, "relation_sha256": row["relation_sha256"], **m,
            })
    by_relation = {}
    for r in node_rows:
        by_relation.setdefault(int(r["relation_index"]), {})[int(r["update"])] = r
    sorted_updates = sorted(features_by_update)
    for rel_idx, states in by_relation.items():
        for prev, target in zip(sorted_updates, sorted_updates[1:]):
            a, b = states[prev], states[target]
            m0, m1 = float(a["margin_q25"]), float(b["margin_q25"])
            transition_rows.append({
                "seed": args.seed, "graph_type": args.graph_type, "k_pid": args.k_pid,
                "relation_index": rel_idx, "anchor_local_index": a["anchor_local_index"],
                "anchor_image_index": a["anchor_image_index"], "anchor_pid": a["anchor_pid"],
                "negative_pid": a["negative_pid"], "prev_update": prev, "target_update": target,
                "positive_local_indices": a["positive_local_indices"],
                "negative_local_indices": a["negative_local_indices"],
                "prev_margin": m0, "target_margin": m1, "delta_margin": m1 - m0,
                "prev_correct": m0 > 0, "target_correct": m1 > 0,
                "strict_flip": m0 > 0 and m1 <= 0,
                "new_negative_pid": m0 > 0 and m1 <= 0,
                "recovered_pid": m0 <= 0 and m1 > 0,
                "relation_sha256": a["relation_sha256"],
            })
    def write_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_csv(args.output_dir / "multi_pid_node_rows.csv", node_rows)
    write_csv(args.output_dir / "multi_pid_transition_rows.csv", transition_rows)
    summary = {
        "status": "PASS_FIXED_UPDATE0_GRAPH_TRAJECTORY_EVAL",
        "seed": args.seed, "graph": str(args.graph), "graph_sha256": sha256_file(args.graph),
        "feature_update0": str(args.feature_update0), "feature_update0_sha256": sha256_file(args.feature_update0),
        "graph_type": args.graph_type, "k_pid": args.k_pid, "anchor_count": len({r["anchor_local_index"] for r in relations}),
        "relation_count": len(relations), "image_count": len(all_indices), "updates": sorted_updates,
        "transition_count": len(transition_rows), "strict_flip_count": int(sum(bool(r["strict_flip"]) for r in transition_rows)),
        "new_negative_pid_count": int(sum(bool(r["new_negative_pid"]) for r in transition_rows)),
        "recovered_pid_count": int(sum(bool(r["recovered_pid"]) for r in transition_rows)),
        "future_nodes_used_for_selection": False, "official_query_gallery_used": False,
        "optimizer_step_executed": False, "cpu_model_fallback": False,
        "script_sha256": sha256_file(Path(__file__)), "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "multi_pid_trajectory_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
