#!/usr/local/bin/python3.10
"""S0/S1 严格配对短程训练诊断。

S0 与 S1 在同一进程中复用同一批已物化的 train-only 输入；每个分支从相同
checkpoint、optimizer、GradScaler 和 seed 重新开始。脚本不修改正式训练器，
不读取 query/gallery，不保存模型 checkpoint。

任何 batch 输入、初始状态、AMP skip 序列或有效 optimizer 更新序列不一致，
都显式标记 INVALID_PAIR；不删除该步骤、不用另一条轨迹补齐。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
CONFIG = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen/B0_seed1234/config.yml"
CHECKPOINT = ENGINE / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth"
SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
EXPECTED = "PPU-ZW810E"
EPOCH = 60
LAMBDA = 0.1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_json(value: Any) -> str:
    raw = json.dumps(canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256_bytes(raw)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach().float()).all().item())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_hash() -> dict[str, str]:
    result = {
        "python": sha256_bytes(pickle.dumps(random.getstate(), protocol=4)),
        "numpy": sha256_bytes(pickle.dumps(np.random.get_state(), protocol=4)),
        "torch": sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
    }
    try:
        result["torch_cuda"] = sha256_bytes(b"".join(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()))
    except Exception as exc:
        result["torch_cuda"] = f"unavailable:{type(exc).__name__}"
    return result


def tensor_hash(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor.detach().contiguous().cpu().numpy().tobytes())


def hash_batch(batch: dict[str, Any]) -> dict[str, Any]:
    tensor_hashes = {key: tensor_hash(value) for key, value in sorted(batch.items()) if torch.is_tensor(value)}
    metadata = {key: [str(item) for item in value] if isinstance(value, list) else str(value)
                for key, value in sorted(batch.items()) if not torch.is_tensor(value)}
    return {
        "tensor_hashes": tensor_hashes,
        "metadata_hash": sha256_json(metadata),
        "batch_hash": sha256_json({"tensor_hashes": tensor_hashes, "metadata": metadata}),
        "image_output_hash": tensor_hashes.get("image"),
        "caption_hash": tensor_hashes.get("caption_feature"),
        "image_paths": metadata.get("image_path", []),
    }


def configure_modules(seed: int):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ENGINE))
    sys.path.insert(0, str(ENGINE / "scripts"))
    import gpc_a4_mechanism_probe as probe
    import gpc_strict_budget_plan_audit as audit
    import gpc_science_rng as science
    import phase6b_soft_relation_screen as b6
    import phase7i_relation_specificity as p7
    from model.build import factory

    phase = f"{science.SCIENCE_RNG_NAMESPACE}|{science.SCIENCE_PROTOCOL_ID}"
    for module in (p7, b6, audit.base):
        module.PHASE = phase
        module.SEED = seed
        module.WORLD_SIZE = 2
    return probe, audit, b6, p7, factory


def relation_key(item: tuple) -> str:
    return json.dumps(canonical(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def relation_hash(plan: dict[int, list[tuple]]) -> str:
    rows = [{"iteration": int(iteration), "relations": sorted(relation_key(item) for item in plan[iteration])}
            for iteration in sorted(plan)]
    return sha256_json(rows)


def build_plan_and_batches(seed: int, limit: int):
    probe, audit, b6, p7, factory = configure_modules(seed)
    cfg = probe.load_cfg(CONFIG)
    cfg.defrost(); cfg.DATA.NUM_WORKERS = 0; cfg.DATA.PIN_MEMORY = False; cfg.SOLVER.SEED = seed; cfg.freeze()
    dataset = b6.base.build_dataset(cfg)
    transform, _ = probe.build_img_transforms(cfg)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    state = probe.active_state(dataset, train_pids)
    index_map = b6.base.build_index_map(dataset)
    loader, loader_meta = b6.base.make_epoch_loader(dataset, transform, state, index_map, EPOCH, 0)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index

    candidates: dict[int, dict[str, list[tuple]]] = {}
    s0: dict[int, list[tuple]] = {}
    batch_indices: dict[int, list[int]] = {}
    batch_path_hashes: dict[int, str] = {}
    for iteration, indices in enumerate(loader.batch_sampler, start=1):
        indices = [int(index) for index in indices]
        entries = [loader.dataset.samples[index][0] for index in indices]
        current = p7.candidate_lists(entries, path_to_index, bank)
        candidates[iteration] = {kind: list(current[kind]) for kind in ("cross", "same", "id")}
        budget = min(len(current["cross"]), len(current["same"]), len(current["id"]))
        selected = p7.stable_selection(current["id"], budget, EPOCH, 0, iteration, "id")
        if selected:
            s0[iteration] = selected
        batch_indices[iteration] = indices
        batch_path_hashes[iteration] = sha256_json([str(entry["image_path"]) for entry in entries])
        if iteration >= limit + 1:
            break

    iterator = iter(loader)
    cached_batches: list[dict[str, Any]] = []
    cached_hashes: list[dict[str, Any]] = []
    for iteration in range(1, limit + 2):
        batch = next(iterator)
        frozen = {}
        for key, value in batch.items():
            frozen[key] = value.clone() if torch.is_tensor(value) else list(value) if isinstance(value, list) else value
        frozen["_batch_indices"] = batch_indices[iteration]
        frozen["_batch_path_hash"] = batch_path_hashes[iteration]
        cached_batches.append(frozen)
        cached_hashes.append(hash_batch(frozen))
        actual_path_hash = sha256_json([str(path) for path in batch["image_path"]])
        if actual_path_hash != batch_path_hashes[iteration]:
            raise RuntimeError(f"batch path plan mismatch at iteration {iteration}")

    s1, gate = audit.choose_s1(s0, candidates, seed, 0)
    all_batch_plan_hash = sha256_json([{"iteration": i, "indices": batch_indices[i]} for i in sorted(batch_indices)])
    candidate_hash = sha256_json({str(i): {kind: sorted(relation_key(item) for item in candidates[i][kind])
                                           for kind in ("cross", "same", "id")} for i in sorted(candidates)})
    return {
        "probe": probe, "audit": audit, "b6": b6, "p7": p7, "factory": factory, "cfg": cfg,
        "dataset": dataset, "loader_meta": loader_meta, "cached_batches": cached_batches,
        "cached_hashes": cached_hashes, "s0": s0, "s1": s1, "candidates": candidates,
        "s1_gate": gate, "batch_plan_sha256": all_batch_plan_hash, "candidate_relation_sha256": candidate_hash,
        "s0_relation_sha256": relation_hash(s0), "s1_relation_sha256": relation_hash(s1),
        "split_sha256": sha256_file(SPLIT), "config_sha256": sha256_file(CONFIG),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(factory, cfg):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {str(key).replace("module.", "", 1): value for key, value in state.items()}
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=77, camera_num=12)
    model.load_state_dict(state, strict=True)
    return model, state


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_model_optimizer(plan, seed: int, device: torch.device):
    set_seed(seed)
    model, state = load_state(plan["factory"], plan["cfg"])
    initial_rng = rng_hash()
    model.to(device); model.train()
    loss_fn, center = plan["b6"].base.make_loss(plan["cfg"], num_classes=plan["b6"].base.NUM_CLASSES)
    optimizer, _ = plan["b6"].base.make_optimizer(plan["cfg"], model, center)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    optimizer_hash = sha256_json(optimizer.state_dict())
    scaler_hash = sha256_json(scaler.state_dict())
    return model, loss_fn, optimizer, scaler, {
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "model_state_sha256": state_hash(state),
        "optimizer_init_sha256": optimizer_hash,
        "scaler_init_sha256": scaler_hash,
        "scaler_init_scale": float(scaler.get_scale()),
        "rng_after_seed_sha256": initial_rng,
    }


def feature_gradient(value, feature):
    gradient = torch.autograd.grad(value, feature, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(feature) if gradient is None else gradient


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    na = torch.linalg.vector_norm(aa)
    nb = torch.linalg.vector_norm(bb)
    if float(na.cpu()) <= 1e-12 or float(nb.cpu()) <= 1e-12:
        return None
    return float((aa * bb).sum().div(na * nb).cpu())


def relation_tensor(feat, selected):
    z = F.normalize(feat[0].float(), dim=-1)
    values = [1.0 - torch.sum(z[i] * z[j]) for i, j, _image_key, _state_key in selected]
    return torch.stack(values).mean() if values else z.sum() * 0.0


def tracked_parameter_names(model: torch.nn.Module) -> list[str]:
    names = dict(model.named_parameters())
    preferred = [
        "blocks.22.attn.proj.weight", "head_bio.attn.proj.weight", "head.weight", "head_clip_bio.weight",
    ]
    return [name for name in preferred if name in names]


def tracked_snapshot(model, names):
    lookup = dict(model.named_parameters())
    return {name: lookup[name].detach().clone() for name in names}


def tracked_delta(model, before):
    lookup = dict(model.named_parameters())
    total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    for name, old in before.items():
        total = total + (lookup[name].detach().float() - old.float()).pow(2).sum()
    return float(torch.sqrt(total).cpu())


def tracked_state_hash(model, names):
    lookup = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update(lookup[name].detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def nonfinite_gradient_names(model):
    return [name for name, parameter in model.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad.detach().float()).all().cpu())]


def run_arm(plan, seed: int, arm: str, steps: int, device: torch.device):
    model, loss_fn, optimizer, scaler, init = make_model_optimizer(plan, seed, device)
    names = tracked_parameter_names(model)
    rows = []
    relation_total = 0
    for step, batch in enumerate(plan["cached_batches"][:-1], start=1):
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device, non_blocking=False)
        pids = batch["pid"].to(device, non_blocking=False)
        cams = batch["camid"].to(device, non_blocking=False)
        clothes = batch["clothes_id"].to(device, non_blocking=False)
        captions = batch["caption_feature"].to(device, non_blocking=False)
        writer = plan["b6"].base.MetricCapture()
        with torch.autocast(device_type="cuda", enabled=True):
            score, feat = model(images, cam_label=cams)
        base_total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=step)
        components = getattr(loss_fn, "last_components", {})
        selected = (plan["s0"] if arm == "S0" else plan["s1"]).get(step, [])
        relation = relation_tensor(feat, selected)
        total = base_total + LAMBDA * relation
        g_total = feature_gradient(total, feat[0])
        g_relation = feature_gradient(relation, feat[0])
        g_base = g_total - LAMBDA * g_relation
        before = tracked_snapshot(model, names)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nonfinite_names = nonfinite_gradient_names(model)
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        state = scaler._per_optimizer_states.get(id(optimizer), {})
        found_inf = sum(float(value.item()) for value in state.get("found_inf_per_device", {}).values())
        updated = found_inf == 0.0
        scaler.update()
        torch.cuda.synchronize(device)
        relation_total += len(selected)
        rows.append({
            "step": step,
            "batch_hash": plan["cached_hashes"][step - 1]["batch_hash"],
            "batch_image_output_hash": plan["cached_hashes"][step - 1]["image_output_hash"],
            "selected_relation_count": len(selected),
            "selected_relation_keys_sha256": sha256_json([relation_key(item) for item in selected]),
            "base_loss": float(base_total.detach().float().cpu()),
            "gpc_loss_raw": float(relation.detach().float().cpu()),
            "gpc_loss_applied": float((LAMBDA * relation).detach().float().cpu()),
            "gpc_feature_grad_norm_raw": float(torch.linalg.vector_norm(g_relation.detach().float()).cpu()),
            "gpc_feature_grad_norm_applied": float(torch.linalg.vector_norm((LAMBDA * g_relation).detach().float()).cpu()),
            "base_feature_grad_norm": float(torch.linalg.vector_norm(g_base.detach().float()).cpu()),
            "gpc_base_feature_grad_cosine": cosine(g_base, g_relation),
            "total_feature_grad_norm": float(torch.linalg.vector_norm(g_total.detach().float()).cpu()),
            "ce_loss": float(components.get("ce", torch.tensor(float("nan"))).detach().float().cpu()),
            "triplet_loss": float(components.get("triplet", torch.tensor(float("nan"))).detach().float().cpu()),
            "bio_loss": float(components.get("bio", torch.tensor(float("nan"))).detach().float().cpu()),
            "amp_scale_before": scale_before,
            "amp_scale_after": float(scaler.get_scale()),
            "found_inf": found_inf,
            "nonfinite_gradient_names": nonfinite_names,
            "skipped": not updated,
            "optimizer_step_executed": updated,
            "tracked_parameter_update_norm": tracked_delta(model, before) if updated else 0.0,
            "tracked_parameter_state_sha256": tracked_state_hash(model, names),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "tracked_parameter_names": names,
            "rng_after_step": rng_hash(),
        })
    diagnostic = plan["cached_batches"][-1]
    with torch.no_grad():
        images = diagnostic["image"].to(device, non_blocking=False)
        pids = diagnostic["pid"].to(device, non_blocking=False)
        cams = diagnostic["camid"].to(device, non_blocking=False)
        clothes = diagnostic["clothes_id"].to(device, non_blocking=False)
        captions = diagnostic["caption_feature"].to(device, non_blocking=False)
        with torch.autocast(device_type="cuda", enabled=True):
            score, feat = model(images, cam_label=cams)
        writer = plan["b6"].base.MetricCapture()
        diagnostic_loss = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=9999)
        id_acc = float((score["cls_score"].argmax(dim=-1) == pids).float().mean().cpu())
        z = F.normalize(feat[0].float(), dim=-1)
        consistency_values = [1.0 - torch.sum(z[i] * z[j]) for i in range(len(pids)) for j in range(i + 1, len(pids)) if int(pids[i]) == int(pids[j])]
        consistency = float(torch.stack(consistency_values).mean().cpu()) if consistency_values else None
    torch.cuda.synchronize(device)
    return {
        "arm": arm,
        "seed": seed,
        "steps": len(rows),
        "optimizer_updates": sum(int(row["optimizer_step_executed"]) for row in rows),
        "skipped_steps": sum(int(row["skipped"]) for row in rows),
        "skip_sequence": [int(row["step"]) for row in rows if row["skipped"]],
        "effective_update_sequence": [int(row["optimizer_step_executed"]) for row in rows],
        "initial_state": init,
        "rows": rows,
        "selected_relation_total": relation_total,
        "selected_relation_sha256": relation_hash({
            iteration: values for iteration, values in (plan["s0"] if arm == "S0" else plan["s1"]).items()
            if iteration <= len(rows)
        }),
        "batch_plan_sha256": plan["batch_plan_sha256"],
        "candidate_relation_sha256": plan["candidate_relation_sha256"],
        "input_batch_hashes": [row["batch_hash"] for row in rows],
        "diagnostic_batch_hash": plan["cached_hashes"][-1]["batch_hash"],
        "diagnostic_base_loss": float(diagnostic_loss.detach().float().cpu()),
        "diagnostic_identity_accuracy": id_acc,
        "diagnostic_same_pid_consistency_loss": consistency,
        "cpu_model_fallback": False,
    }


def relation_profile(plan: dict[str, Any], arm: str, max_iteration: int | None = None) -> dict[str, Any]:
    selected_plan = plan["s0"] if arm == "S0" else plan["s1"]
    pid_counts: dict[str, int] = {}
    same_clothes = 0
    cross_clothes = 0
    unique_keys: set[str] = set()
    total = 0
    for iteration in sorted(selected_plan):
        if max_iteration is not None and iteration > max_iteration:
            continue
        for item in selected_plan[iteration]:
            total += 1
            _i, _j, _image_key, state_key = item
            pid = state_key[0]
            pid_key = str(pid)
            pid_counts[pid_key] = pid_counts.get(pid_key, 0) + 1
            if int(state_key[1]) == int(state_key[2]):
                same_clothes += 1
            else:
                cross_clothes += 1
            unique_keys.add(relation_key(item))
    return {
        "total_relations": total,
        "unique_relations": len(unique_keys),
        "duplicate_relations": total - len(unique_keys),
        "pid_counts": dict(sorted(pid_counts.items(), key=lambda pair: int(pair[0]))),
        "same_clothes_relations": same_clothes,
        "cross_clothes_relations": cross_clothes,
        "same_clothes_fraction": (same_clothes / total) if total else None,
        "cross_clothes_fraction": (cross_clothes / total) if total else None,
    }


def compare_pair(s0: dict, s1: dict) -> dict[str, Any]:
    same_initial = s0["initial_state"] == s1["initial_state"]
    same_batch = s0["input_batch_hashes"] == s1["input_batch_hashes"]
    same_diag = s0["diagnostic_batch_hash"] == s1["diagnostic_batch_hash"]
    same_skip = s0["skip_sequence"] == s1["skip_sequence"]
    same_effective = s0["effective_update_sequence"] == s1["effective_update_sequence"]
    return {
        "pair_valid": bool(same_initial and same_batch and same_diag and same_skip and same_effective),
        "same_initial_state": same_initial,
        "same_input_batches": same_batch,
        "same_diagnostic_batch": same_diag,
        "same_amp_skip_sequence": same_skip,
        "same_effective_update_sequence": same_effective,
        "s0_skip_sequence": s0["skip_sequence"],
        "s1_skip_sequence": s1["skip_sequence"],
        "s0_optimizer_updates": s0["optimizer_updates"],
        "s1_optimizer_updates": s1["optimizer_updates"],
        "diagnostic_delta_base_loss_s1_minus_s0": s1["diagnostic_base_loss"] - s0["diagnostic_base_loss"],
        "diagnostic_delta_consistency_s1_minus_s0": (None if s0["diagnostic_same_pid_consistency_loss"] is None or s1["diagnostic_same_pid_consistency_loss"] is None else s1["diagnostic_same_pid_consistency_loss"] - s0["diagnostic_same_pid_consistency_loss"]),
    }


def run_pair(args):
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 不可用，拒绝 CPU model fallback")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or any(name != EXPECTED for name in names):
        raise RuntimeError(f"PPU 设备不符合要求: {names}")
    device = torch.device("cuda:0")
    plan = build_plan_and_batches(args.seed, args.steps)
    print(f"中文进度：seed={args.seed} steps={args.steps} 开始 S0", flush=True)
    s0 = run_arm(plan, args.seed, "S0", args.steps, device)
    gc.collect(); torch.cuda.empty_cache()
    print(f"中文进度：seed={args.seed} steps={args.steps} 开始 S1", flush=True)
    s1 = run_arm(plan, args.seed, "S1", args.steps, device)
    comparison = compare_pair(s0, s1)
    payload = {
        "status": "PASS_CONTROLLED_PAIR" if comparison["pair_valid"] else "INVALID_PAIR",
        "run_time": args.run_time, "seed": args.seed, "steps": args.steps, "epoch": EPOCH,
        "device_names": names, "cpu_model_fallback": False, "lambda": LAMBDA,
        "checkpoint_sha256": plan["checkpoint_sha256"], "config_sha256": plan["config_sha256"],
        "split_sha256": plan["split_sha256"], "batch_plan_sha256": plan["batch_plan_sha256"],
        "candidate_relation_sha256": plan["candidate_relation_sha256"],
        "s0_selected_relation_sha256": s0["selected_relation_sha256"],
        "s1_selected_relation_sha256": s1["selected_relation_sha256"],
        "s0_selected_relation_sha256_full_plan": plan["s0_relation_sha256"],
        "s1_selected_relation_sha256_full_plan": plan["s1_relation_sha256"],
        "s1_offline_gate": plan["s1_gate"],
        "batch_fingerprints": plan["cached_hashes"],
        "s0_relation_profile": relation_profile(plan, "S0", args.steps),
        "s1_relation_profile": relation_profile(plan, "S1", args.steps),
        "s0": s0, "s1": s1, "comparison": comparison,
        "protocol_note": "S0/S1 只允许改变当前 batch 内合法同 PID/不同图片关系；未执行官方 query/gallery。",
    }
    write_json(args.output / f"s0_s1_seed{args.seed}_steps{args.steps}.json", payload)
    print(json.dumps({"status": payload["status"], "seed": args.seed, "steps": args.steps, "comparison": comparison}, ensure_ascii=False, indent=2), flush=True)
    return 0 if comparison["pair_valid"] else 2


def rank_preflight_child(args):
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 不可用")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or any(name != EXPECTED for name in names):
        raise RuntimeError(f"PPU 设备不符合要求: {names}")
    probe, _audit, b6, _p7, factory = configure_modules(args.seed)
    cfg = probe.load_cfg(CONFIG)
    cfg.defrost(); cfg.DATA.NUM_WORKERS = 0; cfg.DATA.PIN_MEMORY = False; cfg.SOLVER.SEED = args.seed; cfg.freeze()
    dataset = b6.base.build_dataset(cfg)
    transform, _ = probe.build_img_transforms(cfg)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    state = probe.active_state(dataset, set(map(int, split["train_pids"])))
    index_map = b6.base.build_index_map(dataset)
    loader, meta = b6.base.make_epoch_loader(dataset, transform, state, index_map, EPOCH, args.rank)
    batch = next(iter(loader))
    device = torch.device(f"cuda:{args.rank}"); torch.cuda.set_device(device)
    model, state_dict = load_state(factory, cfg); model.load_state_dict(state_dict, strict=True); model.to(device); model.train()
    loss_fn, center = b6.base.make_loss(cfg, num_classes=b6.base.NUM_CLASSES)
    optimizer, _ = b6.base.make_optimizer(cfg, model, center)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    images = batch["image"].to(device); pids = batch["pid"].to(device); cams = batch["camid"].to(device)
    clothes = batch["clothes_id"].to(device); captions = batch["caption_feature"].to(device)
    writer = b6.base.MetricCapture()
    with torch.autocast(device_type="cuda", enabled=True):
        score, feat = model(images, cam_label=cams)
    total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=1)
    scaler.scale(total).backward(); scaler.unscale_(optimizer); scaler.step(optimizer)
    state_info = scaler._per_optimizer_states.get(id(optimizer), {})
    found_inf = sum(float(value.item()) for value in state_info.get("found_inf_per_device", {}).values())
    scaler.update(); torch.cuda.synchronize(device)
    payload = {"status": "PASS" if found_inf == 0 and finite(total) else "FAIL", "rank": args.rank,
               "device": names[args.rank], "batch_plan_sha256": meta.get("plan_sha256"),
               "loss": float(total.detach().float().cpu()), "found_inf": found_inf,
               "scale_after": float(scaler.get_scale()), "optimizer_step": found_inf == 0,
               "cpu_model_fallback": False}
    write_json(args.output, payload); print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["status"] == "PASS" else 1


def rank_preflight_parent(args):
    args.output.mkdir(parents=True, exist_ok=True)
    processes = []
    for rank in (0, 1):
        child_output = args.output / f"rank{rank}.json"
        child_log = args.output / f"rank{rank}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", "rank_child", "--rank", str(rank),
                   "--seed", str(args.seed), "--run-time", args.run_time, "--output", str(child_output)]
        stream = child_log.open("w", encoding="utf-8")
        processes.append((rank, subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True), stream))
    rows = []
    for rank, process, stream in processes:
        code = process.wait(); stream.close()
        rows.append({"rank": rank, "returncode": code, "json": str(args.output / f"rank{rank}.json"), "log": str(args.output / f"rank{rank}.log")})
    status = "PASS_TWO_PPU_RANK_PREFLIGHT" if all(row["returncode"] == 0 for row in rows) else "FAIL_TWO_PPU_RANK_PREFLIGHT"
    write_json(args.output / "summary.json", {"status": status, "run_time": args.run_time, "seed": args.seed, "rows": rows, "cpu_model_fallback": False})
    print(json.dumps({"status": status, "rows": rows}, ensure_ascii=False, indent=2))
    return 0 if status.startswith("PASS") else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pair", "rank_preflight", "rank_child"), default="pair")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "pair":
        raise SystemExit(run_pair(args))
    if args.mode == "rank_preflight":
        raise SystemExit(rank_preflight_parent(args))
    raise SystemExit(rank_preflight_child(args))


if __name__ == "__main__":
    main()
