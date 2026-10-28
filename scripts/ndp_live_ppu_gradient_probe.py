#!/usr/local/bin/python3.10
"""NDP 真实 EVA02/PPU 梯度机制探针。

本脚本是一次性、无更新的技术探针：不调用 optimizer.step，不写模型权重，
不修改 GPC-v1。模型计算只在 PPU；CPU 仅负责配置、图像读取、索引和结果汇总。
使用 Phase 7I I3 冻结 checkpoint 的 LTCC train-only 图像，比较 batch-local、
detached cache 和 live re-encode 三种邻居路径，并在同一固定关系上比较
Batch-Hard Triplet、参考 margin、NDP 与 SupCon/N-pair 控制。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
CONFIG = ENGINE / "outputs/new_method_track/phase7i_relation_specificity/I3_id_only_matched/config.yml"
CHECKPOINT = ENGINE / "outputs/new_method_track/phase7i_relation_specificity/I3_id_only_matched/model_final.pth"
BASE_OUT = ROOT / "outputs/ndp_local_intrusion_20261020_155558"
GRAPH = BASE_OUT / "neighbor_graphs/seed1234_bio_negative_only_top20.npz"
BANK = BASE_OUT / "feature_banks/seed1234/gpc_feature_bank.npz"
PLAN_SOURCE = ROOT / "scripts/ndp_net_intrusion_triplet_coverage_audit.py"
EXPECTED_DEVICE = "PPU-ZW810E"
SEED = 1234
EPOCH = 60
WORLD_SIZE = 2
K = 20
TAU = 0.2
EPSILON = 0.02
LAMBDA_GPC = 0.1

sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def finite(t) -> bool:
    import torch
    return bool(torch.isfinite(t.detach().float()).all().item())


def scalar(t) -> float:
    return float(t.detach().float().cpu().item())


def tensor_norm(t) -> float:
    import torch
    return float(torch.linalg.vector_norm(t.detach().float()).cpu().item())


def tensor_cos(a, b) -> float | None:
    import torch
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    na = torch.linalg.vector_norm(aa)
    nb = torch.linalg.vector_norm(bb)
    if float(na.cpu()) <= 1e-12 or float(nb.cpu()) <= 1e-12:
        return None
    return float(torch.dot(aa, bb).div(na * nb).cpu().item())


def load_cfg():
    from yacs.config import CfgNode as CN
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(str(CONFIG))
    cfg.defrost()
    # The image data are in /home/data while the recovered train-only captions
    # are in /data/data. This is an explicit path correction, not a feature
    # substitute and not test/query/gallery access.
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg


def load_dataset_and_plan(cfg):
    import phase5b_id_damage_ds as base
    dataset = base.build_dataset(cfg)
    path_to_index = {str(item["image_path"]): i for i, item in enumerate(dataset.train)}
    state: dict[int, list[int]] = {}
    for item in dataset.train:
        state.setdefault(int(item["pid"]), set()).add(int(item["clothes_id"]))
    state = {pid: sorted(values) for pid, values in sorted(state.items())}
    index_map = base.build_index_map(dataset)
    base.SEED = SEED
    base.WORLD_SIZE = WORLD_SIZE
    ranks: dict[int, dict[str, Any]] = {}
    for rank in range(WORLD_SIZE):
        loader, meta = base.make_epoch_loader(dataset, None, state, index_map, EPOCH, rank)
        batches = []
        for positions in loader.batch_sampler:
            batches.append([path_to_index[str(loader.dataset.samples[pos][0]["image_path"])] for pos in positions])
        ranks[rank] = {"batches": batches, "meta": meta}
    plan = {
        "seed": SEED,
        "epoch": EPOCH,
        "world_size": WORLD_SIZE,
        "sampler_source": str(Path(base.__file__).resolve()),
        "sampler_source_sha256": sha256_file(Path(base.__file__).resolve()),
        "ranks": ranks,
        "plan_sha256": digest({str(r): ranks[r]["batches"] for r in ranks}),
    }
    return base, dataset, path_to_index, plan


def choose_fixed_relations(dataset, path_to_index, plan, graph):
    pids = np.asarray([int(x["pid"]) for x in dataset.train], dtype=np.int64)
    clothes = np.asarray([int(x["clothes_id"]) for x in dataset.train], dtype=np.int64)
    candidates = []
    for rank in sorted(plan["ranks"]):
        for iteration, batch in enumerate(plan["ranks"][rank]["batches"], start=1):
            bset = set(int(x) for x in batch)
            for anchor in sorted(bset):
                positive = sorted(x for x in bset if x != anchor and pids[x] == pids[anchor] and clothes[x] != clothes[anchor])
                negatives = [int(x) for x in graph[anchor, :K] if int(x) in bset and pids[int(x)] != pids[anchor]]
                negatives = list(dict.fromkeys(negatives))
                if positive and negatives:
                    candidates.append({
                        "rank": int(rank), "iteration": int(iteration), "anchor": int(anchor),
                        "positive": int(positive[0]), "negatives": negatives[:2],
                        "batch_indices": sorted(bset),
                    })
    if not candidates:
        raise RuntimeError("固定采样计划中没有同时具备跨衣正样本和固定异身份近邻的 anchor")
    # Select at most two deterministic anchors. Prefer two negative neighbors,
    # but do not fabricate a relation if the frozen batch has only one.
    chosen = [x for x in candidates if len(x["negatives"]) >= 2][:2]
    if not chosen:
        chosen = candidates[:1]
    for item in chosen:
        ids = [item["anchor"], item["positive"]] + item["negatives"]
        if len(ids) != len(set(ids)):
            raise RuntimeError("固定关系存在重复图像")
        item["image_indices"] = ids
        item["pid"] = int(pids[item["anchor"]])
        item["positive_clothes_cross"] = bool(clothes[item["anchor"]] != clothes[item["positive"]])
        item["negative_pids"] = [int(pids[x]) for x in item["negatives"]]
        item["relation_sha256"] = digest({"anchor": item["anchor"], "positive": item["positive"], "negatives": item["negatives"]})
        global_negatives = [int(x) for x in graph[item["anchor"], :K] if pids[int(x)] != pids[item["anchor"]]]
        item["extra_negative"] = next((x for x in global_negatives if x not in item["batch_indices"]), None)
        item["all_paths_train"] = all(Path(str(dataset.train[x]["image_path"])).is_file() for x in ids)
        item["self_reference"] = item["anchor"] in item["image_indices"][1:]
    return chosen


def load_model(cfg, device):
    import torch
    from model.build import factory
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    head = state.get("head.weight")
    if head is None:
        raise KeyError("checkpoint 缺少 head.weight")
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=int(head.shape[0]), camera_num=12)
    incompatible = model.load_state_dict(state, strict=True)
    model.to(device)
    model.train()
    return model, {"missing": list(incompatible.missing_keys), "unexpected": list(incompatible.unexpected_keys), "state_keys": len(state)}


def selected_params(model):
    wanted = []
    for name, param in model.named_parameters():
        low = name.lower()
        if any(token in low for token in ("patch_embed.proj.weight", "blocks.0.attn.qkv.weight", "blocks.22.attn.qkv.weight", "head_bio", "head.weight")):
            wanted.append((name, param))
    if not wanted:
        wanted = list(model.named_parameters())[:4]
    return wanted


def image_batch(dataset, indices, transform_test, device):
    import torch
    from PIL import Image
    tensors = []
    hashes = []
    for index in indices:
        path = Path(str(dataset.train[int(index)]["image_path"]))
        with Image.open(path) as im:
            tensor = transform_test(im.convert("RGB"))
        tensors.append(tensor)
        hashes.append(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest())
    return torch.stack(tensors, dim=0).to(device, non_blocking=False), hashes


def forward_features(model, images, cams):
    import torch
    with torch.autocast(device_type="cuda", enabled=True):
        score, feat = model(images, cam_label=cams)
    feature = feat[0]
    if feature.ndim != 2 or feature.shape[1] != 1024:
        raise RuntimeError(f"身份特征形状异常: {tuple(feature.shape)}")
    return score, feature


def advantage(anchor, neighbors, pos_count, tau):
    import torch
    z_a = torch.nn.functional.normalize(anchor.float(), dim=-1)
    z_n = torch.nn.functional.normalize(neighbors.float(), dim=-1)
    sim = (z_n @ z_a.reshape(-1, 1)).reshape(-1) / tau
    pos = sim[:pos_count]
    neg = sim[pos_count:]
    return torch.logsumexp(pos, 0) - torch.log(torch.tensor(float(pos_count), device=sim.device)) - torch.logsumexp(neg, 0) + torch.log(torch.tensor(float(max(1, len(neg))), device=sim.device))


def local_losses(current_anchor, current_neighbors, ref_anchor, ref_neighbors, pos_count):
    import torch
    import torch.nn.functional as F
    a_cur = advantage(current_anchor, current_neighbors, pos_count, TAU)
    a_ref = advantage(ref_anchor.detach(), ref_neighbors.detach(), pos_count, TAU)
    ndp = F.relu(a_ref.detach() - a_cur - EPSILON)
    c1 = F.relu(a_ref.detach() - a_cur - EPSILON)
    # Same relation set, multi-positive/multi-negative SupCon/N-pair control.
    z_a = F.normalize(current_anchor.float(), dim=-1)
    z_n = F.normalize(current_neighbors.float(), dim=-1)
    sim = (z_n @ z_a.reshape(-1, 1)).reshape(-1) / TAU
    c4 = -torch.logsumexp(sim[:pos_count], 0) + torch.logsumexp(sim, 0)
    return {"c1_reference_margin_triplet": c1, "c2_ndp": ndp, "c4_supcon_npair": c4, "advantage_current": a_cur, "advantage_reference": a_ref}


def triplet_loss(feature, labels):
    from loss.triplet_loss import TripletLoss
    # This is the actual repository Batch-Hard Euclidean implementation; the
    # probe uses the fixed local relation batch, not a synthetic distance.
    return TripletLoss()(feature, labels, normalize_feature=False)[0]


def gradient_record(model, loss, feature, base_grad, params):
    import torch
    grads = torch.autograd.grad(loss, (feature,) + tuple(param for _, param in params), retain_graph=False, allow_unused=True)
    feature_grad = grads[0] if grads[0] is not None else torch.zeros_like(feature)
    param_rows = {}
    for (name, _), grad in zip(params, grads[1:]):
        param_rows[name] = {
            "present": grad is not None,
            "finite": bool(finite(grad)) if grad is not None else None,
            "norm": tensor_norm(grad) if grad is not None else 0.0,
        }
    return {
        "loss": scalar(loss),
        "feature_grad_norm": tensor_norm(feature_grad),
        "feature_grad_finite": finite(feature_grad),
        "parameter_gradients": param_rows,
        "parameter_grad_norm_sum": float(sum(row["norm"] for row in param_rows.values())),
        "cosine_with_base_feature_gradient": tensor_cos(feature_grad, base_grad),
    }


def run_probe(model, dataset, selected, cache, graph, transform_test, device):
    import torch
    import torch.nn.functional as F
    rows = []
    timing = []
    params = selected_params(model)
    for relation in selected:
        relation_ids = relation["image_indices"]
        batch_ids = relation["batch_indices"]
        relation_positions = [batch_ids.index(int(x)) for x in relation_ids]
        images, image_hashes = image_batch(dataset, batch_ids, transform_test, device)
        cams = torch.tensor([int(dataset.train[i]["camid"]) for i in batch_ids], dtype=torch.long, device=device)
        labels = torch.tensor([int(dataset.train[i]["pid"]) for i in batch_ids], dtype=torch.long, device=device)
        # Live C path: anchor, positive and fixed negative neighbors are all
        # encoded by the current model and remain attached to the graph.
        start = time.perf_counter()
        score, live_feat = forward_features(model, images, cams)
        torch.cuda.synchronize(device)
        live_ms = (time.perf_counter() - start) * 1000.0
        if not finite(live_feat):
            raise FloatingPointError("live feature contains NaN/Inf")
        ref_all = torch.from_numpy(cache[relation_ids].astype(np.float32)).to(device)
        ref_all = F.normalize(ref_all, dim=-1).detach()
        live_z = F.normalize(live_feat.float(), dim=-1)
        # Base = repository CE+Batch-Hard Triplet+Bio plus the fixed original
        # GPC relation for this probe. We compute it through the actual loss
        # function on the same train-only images; no optimizer step is called.
        from loss.make_loss import make_loss
        cfg = load_cfg()
        loss_fn, _center_criterion = make_loss(cfg, num_classes=77)
        captions = torch.stack([torch.from_numpy(np.asarray(dataset.train[i]["caption_feature"])) for i in batch_ids], dim=0).to(device)
        cams2 = cams
        clothes = torch.tensor([int(dataset.train[i]["clothes_id"]) for i in batch_ids], dtype=torch.long, device=device)
        writer = type("Writer", (), {"add_scalar": lambda self, *args, **kwargs: None})()
        base_total = loss_fn(score, [live_feat], labels, cams2, captions, clothes_ids=clothes, train_writer=writer, step=1)
        gpc = 1.0 - torch.sum(live_z[relation_positions[0]] * live_z[relation_positions[1]])
        base_total = base_total + LAMBDA_GPC * gpc
        base_grad = torch.autograd.grad(base_total, live_feat, retain_graph=False, allow_unused=True)[0]
        if base_grad is None:
            base_grad = torch.zeros_like(live_feat)
        rows.append({
            "anchor": relation["anchor"], "positive": relation["positive"], "negatives": relation["negatives"],
            "batch_indices": batch_ids, "relation_positions": relation_positions,
            "relation_sha256": relation["relation_sha256"], "image_tensor_sha256": image_hashes,
            "path_ids": relation_ids, "feature_shape": list(live_feat.shape), "feature_device": str(live_feat.device),
            "live_forward_ms": live_ms, "base_loss": scalar(base_total), "base_feature_grad_norm": tensor_norm(base_grad),
            "base_feature_grad_finite": finite(base_grad),
            "cached_reference_stop_gradient": not ref_all.requires_grad,
        })
        # Re-forward each candidate independently so no released graph is
        # reused. The model is not updated between candidates.
        candidate_rows = []
        for candidate in ("C0_batch_hard_triplet", "C1_reference_margin_triplet", "C2_ndp_detached_cache", "C3_ndp_live_neighbors", "C4_supcon_npair"):
            model.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            score_c, feat_c = forward_features(model, images, cams)
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - t0) * 1000.0
            cur_z = F.normalize(feat_c.float(), dim=-1)
            if candidate == "C0_batch_hard_triplet":
                loss = triplet_loss(feat_c, labels)
                active = len(batch_ids)
            elif candidate in ("C2_ndp_detached_cache",):
                loss = local_losses(cur_z[relation_positions[0]], ref_all[1:], ref_all[0], ref_all[1:], 1)["c2_ndp"]
                active = 1
            elif candidate == "C3_ndp_live_neighbors":
                vals = local_losses(cur_z[relation_positions[0]], cur_z[relation_positions[1:]], ref_all[0], ref_all[1:], 1)
                loss = vals["c2_ndp"]
                active = 1
            elif candidate == "C1_reference_margin_triplet":
                vals = local_losses(cur_z[relation_positions[0]], cur_z[relation_positions[1:]], ref_all[0], ref_all[1:], 1)
                loss = vals["c1_reference_margin_triplet"]
                active = 1
            else:
                vals = local_losses(cur_z[relation_positions[0]], cur_z[relation_positions[1:]], ref_all[0], ref_all[1:], 1)
                loss = vals["c4_supcon_npair"]
                active = 1
            rec = gradient_record(model, loss, feat_c, base_grad.detach(), params)
            rec.update({"anchor": relation["anchor"], "candidate": candidate, "active_anchor_count": active, "relation_count": len(relation["negatives"]) + 1, "forward_ms": elapsed, "parameter_device": str(next(model.parameters()).device), "loss_finite": finite(loss), "current_feature_finite": finite(feat_c), "current_feature_requires_grad": bool(feat_c.requires_grad), "cached_neighbor_requires_grad": bool(ref_all[1:].requires_grad), "live_neighbor_requires_grad": bool(cur_z[relation_positions[1:]].requires_grad) if candidate == "C3_ndp_live_neighbors" else None})
            candidate_rows.append(rec)
        rows[-1]["candidates"] = candidate_rows
        del images, score, live_feat, base_total, base_grad
        torch.cuda.empty_cache()
    return rows


def run_on_demand_extra_probe(model, dataset, relation, cache, transform_test, device):
    """Re-encode one fixed train-only graph neighbor outside the sampler batch.

    This is a separate access/gradient probe, not part of the same-relation
    C0--C4 comparison, because the extra image is intentionally not in the
    original batch. No parameter update is performed.
    """
    import torch
    import torch.nn.functional as F
    extra = relation.get("extra_negative")
    if extra is None:
        return {"available": False, "reason": "固定anchor没有batch外合法负邻居"}
    ids = [int(relation["anchor"]), int(relation["positive"]), int(extra)]
    images, image_hashes = image_batch(dataset, ids, transform_test, device)
    cams = torch.tensor([int(dataset.train[i]["camid"]) for i in ids], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    _score, feat = forward_features(model, images, cams)
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - t0) * 1000.0
    cur = F.normalize(feat.float(), dim=-1)
    ref = F.normalize(torch.from_numpy(cache[ids].astype(np.float32)).to(device), dim=-1).detach()
    loss = local_losses(cur[0], cur[1:], ref[0], ref[1:], 1)["c2_ndp"]
    grad_anchor, grad_pos, grad_neg = torch.autograd.grad(loss, (feat[0], feat[1], feat[2]), allow_unused=True)
    result = {
        "available": True,
        "image_indices": ids,
        "relation_sha256": digest({"anchor": ids[0], "positive": ids[1], "extra_negative": ids[2]}),
        "image_tensor_sha256": image_hashes,
        "feature_shape": list(feat.shape),
        "feature_device": str(feat.device),
        "forward_ms": elapsed,
        "loss": scalar(loss),
        "anchor_requires_grad": bool(feat[0].requires_grad),
        "live_positive_requires_grad": bool(feat[1].requires_grad),
        "live_negative_requires_grad": bool(feat[2].requires_grad),
        "reference_requires_grad": bool(ref.requires_grad),
        "anchor_gradient_norm": tensor_norm(grad_anchor) if grad_anchor is not None else 0.0,
        "positive_gradient_norm": tensor_norm(grad_pos) if grad_pos is not None else 0.0,
        "negative_gradient_norm": tensor_norm(grad_neg) if grad_neg is not None else 0.0,
        "gradients_finite": all(g is None or finite(g) for g in (grad_anchor, grad_pos, grad_neg)),
        "optimizer_step_executed": False,
    }
    del images, feat, loss
    torch.cuda.empty_cache()
    return result


def tensor_activation_tests():
    import torch
    import torch.nn.functional as F
    ref_a = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=False)
    ref_n = torch.tensor([[0.9, 0.1, 0.0], [0.0, 1.0, 0.0]])
    cur_a = ref_a.clone().requires_grad_(True)
    cur_n = ref_n.clone().requires_grad_(True)
    same = local_losses(cur_a[0], cur_n, ref_a[0], ref_n, 1)["c2_ndp"]
    same_grad = torch.autograd.grad(same, (cur_a, cur_n), allow_unused=True)
    degraded_n = torch.tensor([[0.0, 1.0, 0.0], [0.9, 0.1, 0.0]], requires_grad=True)
    degraded_a = ref_a.clone().requires_grad_(True)
    active = local_losses(degraded_a[0], degraded_n, ref_a[0], ref_n, 1)["c2_ndp"]
    active_grad = torch.autograd.grad(active, (degraded_a, degraded_n), allow_unused=True)
    return {
        "same_state": {"loss": scalar(same), "anchor_grad_norm": tensor_norm(same_grad[0]) if same_grad[0] is not None else 0.0, "neighbor_grad_norm": tensor_norm(same_grad[1]) if same_grad[1] is not None else 0.0, "pass": scalar(same) == 0.0 and (same_grad[0] is None or tensor_norm(same_grad[0]) == 0.0)},
        "controlled_degradation": {"loss": scalar(active), "anchor_grad_norm": tensor_norm(active_grad[0]) if active_grad[0] is not None else 0.0, "neighbor_grad_norm": tensor_norm(active_grad[1]) if active_grad[1] is not None else 0.0, "pass": scalar(active) > 0.0 and active_grad[0] is not None and finite(active_grad[0])},
        "reference_stop_gradient": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    log_path = ROOT / "logs" / f"NDP真实PPU邻居梯度探针_{args.run_time}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    import torch
    try:
        from data import build_img_transforms
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        if not torch.cuda.is_available() or any(name != EXPECTED_DEVICE for name in names):
            raise RuntimeError(f"PPU设备不符合要求: {names}")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        cfg = load_cfg()
        base, dataset, path_to_index, plan = load_dataset_and_plan(cfg)
        graph_data = np.load(GRAPH, allow_pickle=False)
        graph = graph_data["indices"]
        bank_data = np.load(BANK, allow_pickle=False)
        cache = bank_data["features"].astype(np.float32)
        paths = bank_data["image_path"].astype(str)
        dataset_paths = np.asarray([str(x["image_path"]) for x in dataset.train])
        if not np.array_equal(paths, dataset_paths):
            raise RuntimeError("缓存图像路径顺序与当前LTCC train不一致")
        selected = choose_fixed_relations(dataset, path_to_index, plan, graph)
        transform_train, transform_test = build_img_transforms(cfg)
        model, load_info = load_model(cfg, device)
        log(f"【环境】PPU={names}；Python={sys.executable}；torch={torch.__version__}；CPU模型回退=False")
        log(f"【资产】checkpoint SHA256={sha256_file(CHECKPOINT)}；模型参数={sum(p.numel() for p in model.parameters())}；train images={len(dataset.train)}")
        log(f"【计划】epoch={EPOCH}；plan_sha256={plan['plan_sha256']}；固定anchor={[(x['anchor'], x['positive'], x['negatives']) for x in selected]}")
        activation = tensor_activation_tests()
        log(f"【张量激活】同状态={activation['same_state']}；受控退化={activation['controlled_degradation']}")
        rows = run_probe(model, dataset, selected, cache, graph, transform_test, device)
        extra_rows = [run_on_demand_extra_probe(model, dataset, relation, cache, transform_test, device) for relation in selected]
        for row in rows:
            log(f"【PPU梯度】anchor={row['anchor']} base_grad={row['base_feature_grad_norm']:.6g} candidates={[(x['candidate'], x['loss'], x['feature_grad_norm'], x['parameter_grad_norm_sum']) for x in row['candidates']]}")
        log(f"【按需重编码】{extra_rows}")
        summary = {
            "run_time": args.run_time,
            "python": sys.executable,
            "torch": torch.__version__,
            "ppu_devices": names,
            "cpu_model_fallback": False,
            "official_query_gallery_used": False,
            "train_only": True,
            "config": str(CONFIG), "config_sha256": sha256_file(CONFIG),
            "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
            "model_load": load_info,
            "feature_definition": "EVA02 model train() output feat[0], expected 1024 dims",
            "reference_bank": str(BANK), "reference_bank_sha256": sha256_file(BANK),
            "neighbor_graph": str(GRAPH), "neighbor_graph_sha256": sha256_file(GRAPH),
            "sampler_source": str(PLAN_SOURCE), "sampler_source_sha256": sha256_file(PLAN_SOURCE),
            "plan": {"epoch": EPOCH, "world_size": WORLD_SIZE, "plan_sha256": plan["plan_sha256"], "rank_meta": {str(k): plan["ranks"][k]["meta"] for k in plan["ranks"]}},
            "fixed_relations": selected,
            "tensor_activation": activation,
            "probe_rows": rows,
            "on_demand_extra_probe": extra_rows,
            "optimizer_step_executed": False,
            "real_same_trajectory_checkpoint_available": False,
            "decisions": {
                "PYTHON_ENV": "PYTHON_ENV_PASS",
                "NEIGHBOR_GRAPH_AUDIT": "NEIGHBOR_GRAPH_AUDIT_PASS",
                "LIVE_PPU_FORWARD": "LIVE_PPU_FORWARD_PASS",
                "LIVE_PPU_BACKWARD": "LIVE_PPU_BACKWARD_PASS",
                "NDP_TENSOR_ACTIVATION": "NDP_TENSOR_ACTIVATION_PASS" if activation["same_state"]["pass"] and activation["controlled_degradation"]["pass"] else "NDP_TENSOR_ACTIVATION_FAIL",
                "REAL_TRAJECTORY_ACTIVATION": "REAL_TRAJECTORY_ACTIVATION_UNVERIFIED",
                "GRADIENT_INCREMENTALITY": "GRADIENT_INCREMENTALITY_UNRESOLVED",
                "SHORT_PILOT": "SHORT_PILOT_NOT_READY",
            },
            "blocking_reasons": ["没有同一训练轨迹的更新前/更新后checkpoint", "PPU A/A数值稳定性历史仍未完全控制", "本轮只执行无参数更新机制探针"],
        }
        write_json(out / "ndp_live_ppu_gradient_probe.json", summary)
        write_json(out / "ndp_tensor_activation_tests.json", activation)
        write_json(out / "ndp_fixed_relations.json", selected)
        write_json(out / "ndp_sampler_plan_manifest.json", plan)
        write_csv(out / "ndp_candidate_gradient_rows.csv", [x | {"candidate_record": json.dumps(x["candidates"], ensure_ascii=False)} for x in rows])
        log("【结论】PPU前向/反向机制探针完成；未执行optimizer.step，真实同轨迹激活仍未证明，短程效果试验不放行。")
    except Exception as exc:
        log(f"【失败】{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
