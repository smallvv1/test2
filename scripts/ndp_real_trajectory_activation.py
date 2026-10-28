#!/usr/local/bin/python3.10
"""对 E60 后同一 continuation 轨迹节点做 train-only NDP 激活审计。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
ORIGINAL_RUN_TIME = "20261015_155822"
WORLD_SIZE = 2
TAU = 0.2
EPSILON = 0.02
MAX_ANCHORS = 64

sys.path.insert(0, str(ENGINE / "scripts"))
import gpc_v2_train as v2  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def install_failure_hook(path: Path) -> None:
    """确保子进程在模型前向前后均留下完整异常证据。"""
    def hook(exc_type, exc_value, exc_tb):
        try:
            write_json(path, {
                "status": "FAILED",
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
                "argv": sys.argv,
            })
        finally:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = hook


def load_model(cfg, state, device):
    model = v2.base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=v2.base.NUM_CLASSES, camera_num=v2.base.NUM_CAMERAS)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def restore_numpy_pickle(path: Path):
    import numpy.core
    sys.modules.setdefault("numpy._core", numpy.core)
    for name in ("multiarray", "umath", "_multiarray_umath", "_dtype", "numerictypes", "numeric", "shape_base", "fromnumeric", "_asarray", "records", "memmap"):
        try:
            sys.modules.setdefault("numpy._core." + name, __import__("numpy.core." + name, fromlist=["*"]))
        except Exception:
            pass
    return torch.load(path, map_location="cpu", weights_only=False)


def choose_groups(dataset, split, plan_seed: int, graph_path: Path, run_dir: Path, protocol: dict):
    graph = np.load(graph_path, allow_pickle=False)["indices"]
    train_pids = set(int(x) for x in split["train_pids"])
    active_state = {}
    for item in dataset.train:
        pid = int(item["pid"])
        if pid in train_pids:
            active_state.setdefault(pid, set()).add(int(item["clothes_id"]))
    if set(active_state) != train_pids:
        raise RuntimeError(f"冻结训练 PID active state 不完整: {len(active_state)} != {len(train_pids)}")
    index_map = v2.base.build_index_map(dataset)
    path_to_index = {str(item["image_path"]): i for i, item in enumerate(dataset.train)}
    # 恢复入口只读取一次原始 loader 工厂；不要在每个 batch 再包装 protocol。
    original_loader = protocol["restore"][2]
    groups = []
    seen = set()
    for epoch in (60, 61):
        for rank in range(WORLD_SIZE):
            loader, _meta = original_loader(dataset, None, active_state, index_map, epoch, rank)
            for batch_indices in loader.batch_sampler:
                entries = [loader.dataset.samples[pos][0] for pos in batch_indices]
                batch_ids = sorted(path_to_index[str(item["image_path"])] for item in entries)
                bset = set(batch_ids)
                for anchor in batch_ids:
                    if anchor in seen:
                        continue
                    pid = int(dataset.train[anchor]["pid"])
                    if pid not in train_pids:
                        continue
                    positives = [x for x in batch_ids if x != anchor and int(dataset.train[x]["pid"]) == pid and int(dataset.train[x]["clothes_id"]) != int(dataset.train[anchor]["clothes_id"])]
                    negatives = [int(x) for x in graph[anchor] if int(x) in bset and int(dataset.train[int(x)]["pid"]) != pid and int(dataset.train[int(x)]["pid"]) in train_pids]
                    negatives = list(dict.fromkeys(negatives))
                    if not positives or not negatives:
                        continue
                    positives = sorted(positives)[:2]
                    negatives = negatives[:5]
                    groups.append({"anchor": anchor, "pid": pid, "positives": positives, "negatives": negatives, "rank": rank, "epoch_plan": epoch, "batch_indices": batch_ids})
                    seen.add(anchor)
                    if len(groups) >= MAX_ANCHORS:
                        break
                if len(groups) >= MAX_ANCHORS:
                    break
            if len(groups) >= MAX_ANCHORS:
                break
        if len(groups) >= MAX_ANCHORS:
            break
    if len(groups) < 8:
        raise RuntimeError(f"固定训练采样计划中只有 {len(groups)} 个可用诊断 anchor")
    for group in groups:
        group["relation_sha256"] = digest({"anchor": group["anchor"], "positives": group["positives"], "negatives": group["negatives"]})
    write_json(run_dir / "fixed_diagnostic_groups.json", {"seed": plan_seed, "graph": str(graph_path), "graph_sha256": sha256_file(graph_path), "count": len(groups), "groups_sha256": digest(groups), "groups": groups})
    return groups, protocol, path_to_index


def encode(model, dataset, indices, transform_test, device):
    values = {}
    for start in range(0, len(indices), 8):
        chunk = indices[start:start + 8]
        tensors = []
        cams = []
        for idx in chunk:
            from PIL import Image
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
        feat = F.normalize(feat.float(), dim=-1).detach().cpu()
        for idx, value in zip(chunk, feat):
            values[int(idx)] = value
    return values


def advantage(group, feats):
    anchor = feats[group["anchor"]]
    pos = torch.stack([feats[x] for x in group["positives"]])
    neg = torch.stack([feats[x] for x in group["negatives"]])
    pos_sim = torch.mv(pos, anchor) / TAU
    neg_sim = torch.mv(neg, anchor) / TAU
    return float((torch.logsumexp(pos_sim, 0) - np.log(len(group["positives"])) - torch.logsumexp(neg_sim, 0) + np.log(len(group["negatives"]))).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(1234, 4321))
    ap.add_argument("--trajectory-dir", type=Path, required=True)
    ap.add_argument("--resume-dir", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("真实轨迹激活审计需要两张 PPU，禁止 CPU fallback")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    run_dir = args.trajectory_dir / "activation_audit"
    run_dir.mkdir(parents=True, exist_ok=True)
    install_failure_hook(run_dir / "activation_error.json")

    class Args:
        output = OLD_OUTPUT
        run_time = ORIGINAL_RUN_TIME
        arm = "B0"
        stage = "two_seed"
        seed = args.seed
        epochs = 60
        full_data = False
    split = json.loads((OLD_OUTPUT / "development_split_manifest.json").read_text())
    runtime_protocol = v2.install_protocol(Args, split, run_dir, None)
    cfg = v2.p7.configure()
    dataset = v2.base.build_dataset(cfg)
    graph_path = ROOT / "outputs/ndp_local_intrusion_20261020_155558/neighbor_graphs" / f"seed{args.seed}_bio_negative_only_top20.npz"
    groups, _protocol, path_to_index = choose_groups(dataset, split, args.seed, graph_path, run_dir, runtime_protocol)
    _transform_train, transform_test = v2.base.build_img_transforms(cfg)
    source = restore_numpy_pickle(args.resume_dir / "rank0_state_epoch060.pt")
    source_state = source["model"]
    nodes = [(0, source_state, str(args.resume_dir / "rank0_state_epoch060.pt"))]
    for path in sorted(args.trajectory_dir.glob("node_update*/model.pth")):
        number = int(path.parent.name.replace("node_update", ""))
        nodes.append((number, restore_numpy_pickle(path), str(path)))
    all_indices = sorted({x for g in groups for x in [g["anchor"], *g["positives"], *g["negatives"]]})
    model = load_model(cfg, source_state, device)
    ref_features = encode(model, dataset, all_indices, transform_test, device)
    ref_adv = {g["anchor"]: advantage(g, ref_features) for g in groups}
    rows = []
    for update, state, state_path in nodes:
        model.load_state_dict(state, strict=True)
        model.eval()
        current = encode(model, dataset, all_indices, transform_test, device)
        active = 0
        for group in groups:
            cur_adv = advantage(group, current)
            delta = ref_adv[group["anchor"]] - cur_adv
            is_active = delta > EPSILON
            active += int(is_active)
            rows.append({"seed": args.seed, "update": update, "anchor": group["anchor"], "pid": group["pid"], "reference_advantage": ref_adv[group["anchor"]], "current_advantage": cur_adv, "advantage_drop": delta, "epsilon": EPSILON, "active": is_active, "positive_count": len(group["positives"]), "negative_count": len(group["negatives"]), "relation_sha256": group["relation_sha256"], "state_path": state_path})
        write_json(run_dir / f"node_update{update:04d}_summary.json", {"seed": args.seed, "update": update, "anchor_count": len(groups), "active_anchor_count": active, "activation_rate": active / len(groups), "epsilon": EPSILON, "tau": TAU, "state_path": state_path})
        del current
        torch.cuda.empty_cache()
    with (run_dir / "activation_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summaries = [json.loads(path.read_text()) for path in sorted(run_dir.glob("node_update*_summary.json"))]
    write_json(run_dir / "activation_summary.json", {"seed": args.seed, "reference_checkpoint": str(args.resume_dir / "rank0_state_epoch060.pt"), "reference_is_same_trajectory_start": True, "groups": len(groups), "groups_sha256": digest(groups), "tau": TAU, "epsilon": EPSILON, "nodes": summaries, "real_trajectory_activation": any(x["active_anchor_count"] > 0 for x in summaries), "optimizer_step_executed": False, "official_query_gallery_used": False})


if __name__ == "__main__":
    main()
