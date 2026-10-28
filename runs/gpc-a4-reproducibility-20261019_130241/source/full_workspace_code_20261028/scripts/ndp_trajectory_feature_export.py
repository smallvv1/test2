#!/usr/bin/env python3
"""导出同一 Bio+GPC continuation 的逐节点 train-only 特征。

本脚本只在 PPU 上执行 eval/no_grad forward，不执行 optimizer.step。关系集合在
脚本外预先固定，所有节点使用同一 image index 顺序和同一 test transform。输出
单个压缩 NPZ，足以离线按 PID 分解多身份关系，而不是只保存聚合 advantage。
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
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def compare_reference_advantages(
    rows_path: Path,
    groups: list[dict],
    updates: list[int],
    feature_nodes: list[np.ndarray],
    output_dir: Path,
) -> dict:
    """用导出的逐样本特征复算旧的 LogMeanExp advantage。

    这里不把单个正负 margin 当作旧指标。旧阶段使用
    logmeanexp(pos/tau)-logmeanexp(neg/tau)，因此复算必须调用同一公式。
    """
    group_by_anchor = {int(g["anchor"]): g for g in groups}
    update_to_pos = {int(u): i for i, u in enumerate(updates)}
    feature_maps = []
    for matrix in feature_nodes:
        feature_maps.append({int(idx): torch.from_numpy(row) for idx, row in zip(
            sorted({int(x) for g in groups for x in [g["anchor"], *g["positives"], *g["negatives"]]}), matrix
        )})
    checked = []
    with rows_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            update = int(row["update"])
            anchor = int(row["anchor"])
            if update not in update_to_pos or anchor not in group_by_anchor:
                checked.append({"update": update, "anchor": anchor, "status": "UNMATCHED"})
                continue
            recomputed = act.advantage(group_by_anchor[anchor], feature_maps[update_to_pos[update]])
            old_current = float(row["current_advantage"])
            old_reference = float(row["reference_advantage"])
            checked.append({
                "update": update,
                "anchor": anchor,
                "old_current_advantage": old_current,
                "recomputed_current_advantage": recomputed,
                "abs_error": abs(recomputed - old_current),
                "old_reference_advantage": old_reference,
                "status": "MATCHED",
            })
    out_csv = output_dir / "advantage_reproduction_rows.csv"
    if checked:
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for row in checked for k in row}))
            writer.writeheader()
            writer.writerows(checked)
    matched = [x for x in checked if x.get("status") == "MATCHED"]
    errors = np.asarray([x["abs_error"] for x in matched], dtype=np.float64)
    return {
        "reference_rows": str(rows_path),
        "rows_total": len(checked),
        "rows_matched": len(matched),
        "rows_unmatched": len(checked) - len(matched),
        "max_abs_error": float(errors.max()) if len(errors) else None,
        "mean_abs_error": float(errors.mean()) if len(errors) else None,
        "p95_abs_error": float(np.percentile(errors, 95)) if len(errors) else None,
        "comparison_csv": str(out_csv),
        "formula": "logmeanexp(pos_similarity/0.07)-logmeanexp(neg_similarity/0.07)",
    }


def load_groups(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = data["groups"]
    if data.get("groups_sha256") and data["groups_sha256"] != digest(groups):
        raise RuntimeError("固定关系 groups_sha256 校验失败")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--trajectory-dir", type=Path, required=True)
    ap.add_argument("--resume-dir", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--reference-rows", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("需要两张 PPU；禁止 CPU fallback")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda:0")

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
    _transform_train, transform_test = v2.base.build_img_transforms(cfg)
    groups = load_groups(args.groups)
    if not groups:
        raise RuntimeError("固定关系为空")
    all_indices = sorted({int(x) for g in groups for x in [g["anchor"], *g["positives"], *g["negatives"]]})
    source_path = args.resume_dir / "rank0_state_epoch060.pt"
    source = act.restore_numpy_pickle(source_path)
    source_state = source["model"]
    node_specs = [(0, source_state, str(source_path))]
    for path in sorted(args.trajectory_dir.glob("node_update*/model.pth")):
        update = int(path.parent.name.replace("node_update", ""))
        node_specs.append((update, act.restore_numpy_pickle(path), str(path)))
    if [x[0] for x in node_specs] != [0, 20, 50, 100, 200, 400]:
        raise RuntimeError(f"continuation 节点不完整: {[x[0] for x in node_specs]}")

    metadata = {
        "image_index": np.asarray(all_indices, dtype=np.int64),
        "image_path": np.asarray([str(dataset.train[i]["image_path"]) for i in all_indices]),
        "pid": np.asarray([int(dataset.train[i]["pid"]) for i in all_indices], dtype=np.int64),
        "clothes": np.asarray([int(dataset.train[i]["clothes_id"]) for i in all_indices], dtype=np.int64),
        "camid": np.asarray([int(dataset.train[i]["camid"]) for i in all_indices], dtype=np.int64),
    }
    image_digest = hashlib.sha256(
        json.dumps({k: v.tolist() for k, v in metadata.items()}, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    model = act.load_model(cfg, source_state, device)
    feature_nodes = []
    node_meta = []
    repeat_errors = {}
    for update, state, state_path in node_specs:
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            first = act.encode(model, dataset, all_indices, transform_test, device)
            if update == 0:
                second = act.encode(model, dataset, all_indices, transform_test, device)
            else:
                second = None
        matrix = np.stack([first[i].numpy() for i in all_indices]).astype(np.float32)
        if matrix.shape != (len(all_indices), 1024) or not np.isfinite(matrix).all():
            raise RuntimeError(f"update{update} 特征形状或有限性错误: {matrix.shape}")
        feature_nodes.append(matrix)
        if second is not None:
            delta = np.stack([(first[i] - second[i]).abs().numpy() for i in all_indices])
            repeat_errors = {
                "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
                "p95_abs": float(np.percentile(delta, 95)),
            }
        node_meta.append({"update": update, "state_path": state_path, "state_sha256": sha256_file(Path(state_path))})
        del second
        torch.cuda.empty_cache()

    features = np.stack(feature_nodes).astype(np.float32)
    npz_path = args.output_dir / f"trajectory_features_seed{args.seed}.npz"
    np.savez_compressed(
        npz_path,
        features=features,
        updates=np.asarray([x["update"] for x in node_meta], dtype=np.int64),
        **metadata,
    )
    reproduction = compare_reference_advantages(
        args.reference_rows, groups, [x["update"] for x in node_meta], feature_nodes, args.output_dir
    )
    write_json(args.output_dir / "advantage_reproduction_summary.json", reproduction)
    write_json(args.output_dir / "feature_export_manifest.json", {
        "status": "PASS_PPU_EVAL_NO_GRAD",
        "seed": args.seed,
        "npz": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "groups": len(groups),
        "groups_sha256": digest(groups),
        "image_count": len(all_indices),
        "image_index_sha256": image_digest,
        "updates": [x["update"] for x in node_meta],
        "nodes": node_meta,
        "repeat_encode_error_update0": repeat_errors,
        "feature_definition": "PPU EVA02 feat[0] -> FP32 L2 normalize -> detached CPU export",
        "preprocess": "same build_img_transforms test transform at every node",
        "checkpoint_source": str(args.resume_dir),
        "reference_rows": str(args.reference_rows),
        "advantage_reproduction": reproduction,
        "optimizer_step_executed": False,
        "official_query_gallery_used": False,
        "cpu_model_fallback": False,
        "elapsed_seconds": time.time() - started,
        "source_hashes": {
            "feature_export_script": sha256_file(Path(__file__)),
            "activation_helper": sha256_file(ROOT / "scripts/ndp_real_trajectory_activation.py"),
            "engine": sha256_file(Path("/data/DIFFER/scripts/gpc_v2_train.py")),
        },
    })
    print(json.dumps({"status": "PASS_PPU_EVAL_NO_GRAD", "seed": args.seed, "groups": len(groups), "images": len(all_indices), "npz": str(npz_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
