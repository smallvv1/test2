#!/usr/local/bin/python3.10
"""扩大同一 continuation 的固定关系覆盖；仅 PPU eval/no_grad，不更新参数。

关系在运行前由固定 epoch sampler 与固定 train-only 图构建；所有节点复用同一关系集合。
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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--trajectory-dir", type=Path, required=True)
    ap.add_argument("--resume-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-anchors", type=int, default=256)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("需要两张 PPU；禁止 CPU fallback")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    start = time.time()

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
    graph_path = ROOT / "outputs/ndp_local_intrusion_20261020_155558/neighbor_graphs" / f"seed{args.seed}_bio_negative_only_top20.npz"
    original_max = act.MAX_ANCHORS
    act.MAX_ANCHORS = args.max_anchors
    groups, _, _ = act.choose_groups(dataset, split, args.seed, graph_path, args.output_dir, protocol)
    act.MAX_ANCHORS = original_max
    _transform_train, transform_test = v2.base.build_img_transforms(cfg)

    source_path = args.resume_dir / "rank0_state_epoch060.pt"
    source = act.restore_numpy_pickle(source_path)
    source_state = source["model"]
    node_specs = [(0, source_state, str(source_path))]
    for path in sorted(args.trajectory_dir.glob("node_update*/model.pth")):
        update = int(path.parent.name.replace("node_update", ""))
        node_specs.append((update, act.restore_numpy_pickle(path), str(path)))

    all_indices = sorted({x for g in groups for x in [g["anchor"], *g["positives"], *g["negatives"]]})
    model = act.load_model(cfg, source_state, device)
    with torch.no_grad():
        ref_features = act.encode(model, dataset, all_indices, transform_test, device)
    ref_adv = {g["anchor"]: act.advantage(g, ref_features) for g in groups}
    rows = []
    summaries = []
    for update, state, state_path in node_specs:
        model.load_state_dict(state, strict=True); model.eval()
        with torch.no_grad():
            current = act.encode(model, dataset, all_indices, transform_test, device)
        active = 0
        for g in groups:
            cur = act.advantage(g, current)
            drop = ref_adv[g["anchor"]] - cur
            active += int(drop > act.EPSILON)
            rows.append({
                "seed": args.seed, "update": update, "anchor": g["anchor"], "pid": g["pid"],
                "positive_indices": ";".join(map(str, g["positives"])),
                "negative_indices": ";".join(map(str, g["negatives"])),
                "reference_advantage": ref_adv[g["anchor"]], "current_advantage": cur,
                "advantage_drop": drop, "epsilon": act.EPSILON,
                "active": drop > act.EPSILON, "positive_count": len(g["positives"]),
                "negative_count": len(g["negatives"]), "relation_sha256": g["relation_sha256"],
                "state_path": state_path,
            })
        summaries.append({"seed": args.seed, "update": update, "anchor_count": len(groups),
                          "active_anchor_count": active, "activation_rate": active / len(groups),
                          "state_path": state_path})
        del current
        torch.cuda.empty_cache()

    with (args.output_dir / "expanded_activation_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(rows[0]); writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write_json(args.output_dir / "expanded_activation_summary.json", {
        "status": "PASS_PPU_EVAL_NO_GRAD", "seed": args.seed, "groups": len(groups),
        "unique_indices": len(all_indices), "groups_sha256": digest(groups),
        "graph": str(graph_path), "graph_sha256": sha256_file(graph_path),
        "reference_checkpoint": str(source_path), "reference_checkpoint_sha256": sha256_file(source_path),
        "nodes": summaries, "optimizer_step_executed": False, "official_query_gallery_used": False,
        "cpu_model_fallback": False, "elapsed_seconds": time.time() - start,
    })


if __name__ == "__main__":
    main()
