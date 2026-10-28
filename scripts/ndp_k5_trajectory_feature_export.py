#!/usr/local/bin/python3.10
"""导出 update0/20/50/100/200/400 的 K=5 多 PID train-only 特征。

关系在 update0 图上冻结；后续节点只重新编码同一图像索引，不重建邻域、不执行
optimizer.step。模型计算必须在两张 PPU 环境中运行，输出只包含小型 train-only
候选特征库和元信息，禁止 test/query/gallery。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def load_graph(path: Path, graph_type: str, k_pid: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload["relations"] if r["graph_type"] == graph_type and int(r["k_pid"]) == k_pid]
    rows.sort(key=lambda r: (int(r["anchor_order"]), int(r["negative_pid"])))
    if not rows:
        raise RuntimeError("K=5 graph 为空")
    anchors = {int(r["anchor_local_index"]) for r in rows}
    if len(rows) != len(anchors) * k_pid:
        raise RuntimeError(f"K=5 关系不满足每 anchor 五个负 PID: rows={len(rows)}, anchors={len(anchors)}")
    if any(len({int(x) for x in r["negative_indices"]}) == 0 for r in rows):
        raise RuntimeError("存在空负样本集合")
    return rows


def encode_raw_and_norm(model, dataset, indices, transform_test, device):
    from PIL import Image
    raw_values, norm_values = {}, {}
    for start in range(0, len(indices), 8):
        chunk = indices[start:start + 8]
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
            raise RuntimeError(f"feat[0] shape 异常: {tuple(feat.shape)}")
        raw = feat.float().detach().cpu()
        norm = F.normalize(feat.float(), dim=-1).detach().cpu()
        for idx, a, b in zip(chunk, raw, norm):
            raw_values[int(idx)] = a
            norm_values[int(idx)] = b
    return raw_values, norm_values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--trajectory-dir", type=Path, required=True)
    ap.add_argument("--resume-dir", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--graph-type", default="difficult", choices=("difficult", "representative"))
    ap.add_argument("--k-pid", type=int, default=5)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("K=5 特征导出需要两张 PPU，禁止 CPU fallback")
    started = time.time(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_graph(args.graph, args.graph_type, args.k_pid)
    all_indices = sorted({int(r["anchor_image_index"]) for r in rows} | {int(x) for r in rows for x in r["positive_indices"]} | {int(x) for r in rows for x in r["negative_indices"]})

    class Args:
        output = OLD_OUTPUT; run_time = "20261015_155822"; arm = "B0"; stage = "two_seed"; seed = args.seed; epochs = 60; full_data = False
    split = json.loads((OLD_OUTPUT / "development_split_manifest.json").read_text(encoding="utf-8"))
    v2.install_protocol(Args, split, args.output_dir, None)
    cfg = v2.p7.configure(); dataset = v2.base.build_dataset(cfg); _train, transform_test = v2.base.build_img_transforms(cfg)
    source_path = args.resume_dir / "rank0_state_epoch060.pt"; source = act.restore_numpy_pickle(source_path)
    node_specs = [(0, source["model"], str(source_path))]
    for path in sorted(args.trajectory_dir.glob("node_update*/model.pth")):
        node_specs.append((int(path.parent.name.replace("node_update", "")), act.restore_numpy_pickle(path), str(path)))
    node_specs.sort(key=lambda x: x[0])
    if [x[0] for x in node_specs] != [0, 20, 50, 100, 200, 400]:
        raise RuntimeError(f"continuation 节点不完整: {[x[0] for x in node_specs]}")
    meta = {
        "image_index": np.asarray(all_indices, dtype=np.int64),
        "image_path": np.asarray([str(dataset.train[i]["image_path"]) for i in all_indices]),
        "pid": np.asarray([int(dataset.train[i]["pid"]) for i in all_indices], dtype=np.int64),
        "clothes": np.asarray([int(dataset.train[i]["clothes_id"]) for i in all_indices], dtype=np.int64),
        "camid": np.asarray([int(dataset.train[i]["camid"]) for i in all_indices], dtype=np.int64),
    }
    image_meta_sha = hashlib.sha256(json.dumps({k: v.tolist() for k, v in meta.items()}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    model = act.load_model(cfg, node_specs[0][1], torch.device("cuda:0")); raw_nodes=[]; norm_nodes=[]; node_meta=[]
    for update, state, state_path in node_specs:
        model.load_state_dict(state, strict=True); model.eval()
        raw, norm = encode_raw_and_norm(model, dataset, all_indices, transform_test, torch.device("cuda:0"))
        raw_nodes.append(np.stack([raw[i].numpy() for i in all_indices]).astype(np.float32))
        norm_nodes.append(np.stack([norm[i].numpy() for i in all_indices]).astype(np.float32))
        if not np.isfinite(raw_nodes[-1]).all() or not np.isfinite(norm_nodes[-1]).all(): raise RuntimeError(f"update{update} 非有限特征")
        node_meta.append({"update": update, "state_path": state_path, "state_sha256": sha256_file(Path(state_path))})
        del raw, norm; torch.cuda.empty_cache()
    npz = args.output_dir / f"k5_trajectory_features_seed{args.seed}.npz"
    np.savez_compressed(npz, raw_features=np.stack(raw_nodes), features=np.stack(norm_nodes), updates=np.asarray([x["update"] for x in node_meta], dtype=np.int64), **meta)
    write_json(args.output_dir / "k5_feature_export_manifest.json", {
        "status":"PASS_PPU_EVAL_NO_GRAD_K5", "seed":args.seed, "graph":str(args.graph), "graph_sha256":sha256_file(args.graph), "graph_type":args.graph_type, "k_pid":args.k_pid,
        "relation_count":len(rows), "anchor_count":len({int(r["anchor_local_index"]) for r in rows}), "image_count":len(all_indices), "image_meta_sha256":image_meta_sha,
        "updates":[x["update"] for x in node_meta], "nodes":node_meta, "npz":str(npz), "npz_sha256":sha256_file(npz),
        "feature_definition":"PPU EVA02 feat[0]; raw FP32 and FP32 L2-normalized detached CPU export", "optimizer_step_executed":False, "official_query_gallery_used":False, "cpu_model_fallback":False,
        "elapsed_seconds":time.time()-started, "source_hashes":{"script":sha256_file(Path(__file__)),"activation_helper":sha256_file(ROOT/"scripts/ndp_real_trajectory_activation.py"),"engine":sha256_file(Path("/data/DIFFER/scripts/gpc_v2_train.py"))}
    })
    print(json.dumps({"status":"PASS_PPU_EVAL_NO_GRAD_K5","seed":args.seed,"relations":len(rows),"anchors":len({int(r['anchor_local_index']) for r in rows}),"images":len(all_indices),"npz":str(npz)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
