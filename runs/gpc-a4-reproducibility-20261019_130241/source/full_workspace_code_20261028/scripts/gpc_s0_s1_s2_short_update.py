#!/usr/local/bin/python3.10
"""S0/S1/S2 固定预算关系计划的短程 PPU 配对更新诊断。

只使用 LTCC 冻结开发训练 PID；先缓存同一批 train-only 输入，再为每个分支
从同一初始 model/optimizer/GradScaler 状态执行最多 100 个有效 batch。
这不是正式训练，也不读取 query/gallery。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
CONFIG = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen/B0_seed1234/config.yml"
CHECKPOINT = ENGINE / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth"
SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
EXPECTED = "PPU-ZW810E"
EPOCH = 60
SEED = 1234
LAMBDA = 0.1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach().float()).all().item())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def configure_modules():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ENGINE))
    sys.path.insert(0, str(ENGINE / "scripts"))
    import gpc_a4_mechanism_probe as probe
    import gpc_strict_budget_plan_audit as audit
    import phase6b_soft_relation_screen as b6
    import phase7i_relation_specificity as p7
    from model.build import factory
    import gpc_science_rng as science
    phase = f"{science.SCIENCE_RNG_NAMESPACE}|{science.SCIENCE_PROTOCOL_ID}"
    for module in (p7, b6, audit.base):
        module.PHASE = phase
        module.SEED = SEED
        module.WORLD_SIZE = 2
    return probe, audit, b6, p7, factory


def build_plan_and_batches(limit: int):
    probe, audit, b6, p7, factory = configure_modules()
    cfg = probe.load_cfg(CONFIG)
    cfg.defrost(); cfg.DATA.NUM_WORKERS = 0; cfg.DATA.PIN_MEMORY = False; cfg.SOLVER.SEED = SEED; cfg.freeze()
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
    candidates = {}
    s0 = {}
    batch_indices = {}
    cached = []
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
        if iteration <= limit + 1:
            cached.append(next(iter([loader.dataset[i] for i in indices])) if False else None)
        if iteration >= limit + 1:
            break
    # Rebuild the selected batch tensors through the ordinary loader iterator so
    # all transform randomness is materialized exactly once and reused by arms.
    iterator = iter(loader)
    cached_batches = []
    for _ in range(limit + 1):
        batch = next(iterator)
        frozen = {}
        for key, value in batch.items():
            frozen[key] = value.clone() if torch.is_tensor(value) else list(value) if isinstance(value, list) else value
        cached_batches.append(frozen)
    s1, gate1 = audit.choose_s1(s0, candidates, SEED, 0)
    s2, gate2 = audit.choose_s2(candidates, s0, SEED, 0)
    return probe, b6, factory, cfg, cached_batches, s0, s1, s2, gate1, gate2, loader_meta, candidates


def make_model_optimizer(factory, cfg, device):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {str(key).replace("module.", "", 1): value for key, value in state.items()}
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=77, camera_num=12)
    model.load_state_dict(state, strict=True)
    model.to(device); model.train()
    # Exact formal optimizer construction, including center criterion.
    import phase7i_relation_specificity as p7
    loss_fn, center = p7.base.make_loss(cfg, num_classes=p7.base.NUM_CLASSES)
    optimizer, _ = p7.base.make_optimizer(cfg, model, center)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    return model, loss_fn, optimizer, scaler


def relation_tensor(feat, selected):
    z = F.normalize(feat[0].float(), dim=-1)
    values = [1.0 - torch.sum(z[i] * z[j]) for i, j, _image_key, _state_key in selected]
    return torch.stack(values).mean() if values else z.sum() * 0.0


def same_pid_consistency(feat, pids):
    z = F.normalize(feat[0].float(), dim=-1)
    values = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            if int(pids[i]) == int(pids[j]):
                values.append(1.0 - torch.sum(z[i] * z[j]))
    return float(torch.stack(values).mean().detach().float().cpu()) if values else None


def run_arm(name, batches, selected_plan, probe, b6, factory, cfg, device):
    model, loss_fn, optimizer, scaler = make_model_optimizer(factory, cfg, device)
    rows = []
    skipped = 0
    updates = 0
    for step, batch in enumerate(batches[:-1], start=1):
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device, non_blocking=False)
        pids = batch["pid"].to(device, non_blocking=False)
        cams = batch["camid"].to(device, non_blocking=False)
        clothes = batch["clothes_id"].to(device, non_blocking=False)
        captions = batch["caption_feature"].to(device, non_blocking=False)
        writer = b6.base.MetricCapture()
        with torch.autocast(device_type="cuda", enabled=True):
            score, feat = model(images, cam_label=cams)
        base_total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=step)
        components = getattr(loss_fn, "last_components", {})
        selected = selected_plan.get(step, [])
        relation = relation_tensor(feat, selected)
        applied = LAMBDA * relation
        total = base_total + applied
        relation_grad = torch.autograd.grad(relation, feat[0], retain_graph=True, allow_unused=True)[0]
        relation_grad_norm = float(torch.linalg.vector_norm(relation_grad.detach().float()).cpu()) if relation_grad is not None else 0.0
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        grad_finite = all(finite(grad) for grad in grads)
        scale_before = float(scaler.get_scale())
        if not grad_finite:
            skipped += 1
            scaler.update()
            rows.append({"step": step, "selected_pairs": len(selected), "relation_loss": float(relation.detach().float().cpu()), "base_loss": float(base_total.detach().float().cpu()), "relation_grad_feature_norm": relation_grad_norm, "scale_before": scale_before, "skipped": True, "updated": False, "finite": False})
            continue
        scaler.step(optimizer)
        state = scaler._per_optimizer_states.get(id(optimizer), {})
        found_inf = sum(float(value.item()) for value in state.get("found_inf_per_device", {}).values())
        scaler.update()
        skipped += int(found_inf > 0)
        updates += int(found_inf == 0)
        rows.append({
            "step": step, "selected_pairs": len(selected),
            "relation_loss": float(relation.detach().float().cpu()),
            "relation_applied_loss": float(applied.detach().float().cpu()),
            "base_loss": float(base_total.detach().float().cpu()),
            "ce_loss": float(components.get("ce", torch.tensor(float("nan"))).detach().float().cpu()),
            "triplet_loss": float(components.get("triplet", torch.tensor(float("nan"))).detach().float().cpu()),
            "bio_loss": float(components.get("bio", torch.tensor(float("nan"))).detach().float().cpu()),
            "relation_grad_feature_norm": relation_grad_norm,
            "scale_before": scale_before, "scale_after": float(scaler.get_scale()),
            "found_inf": found_inf, "skipped": bool(found_inf > 0), "updated": bool(found_inf == 0), "finite": True,
        })
    diagnostic = batches[-1]
    with torch.no_grad():
        images = diagnostic["image"].to(device, non_blocking=False)
        pids = diagnostic["pid"].to(device, non_blocking=False)
        cams = diagnostic["camid"].to(device, non_blocking=False)
        clothes = diagnostic["clothes_id"].to(device, non_blocking=False)
        captions = diagnostic["caption_feature"].to(device, non_blocking=False)
        with torch.autocast(device_type="cuda", enabled=True):
            score, feat = model(images, cam_label=cams)
        writer = b6.base.MetricCapture()
        diagnostic_loss = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=9999)
        id_acc = float((score["cls_score"].argmax(dim=-1) == pids).float().mean().cpu())
        consistency = same_pid_consistency(feat, pids)
    torch.cuda.synchronize(device)
    return {
        "arm": name, "steps": len(rows), "optimizer_updates": updates, "skipped_steps": skipped,
        "invalid": updates != len(rows), "rows": rows,
        "mean_base_loss": float(sum(row["base_loss"] for row in rows) / max(len(rows), 1)),
        "mean_relation_loss_active": float(sum(row["relation_loss"] for row in rows if row["selected_pairs"] > 0) / max(sum(row["selected_pairs"] > 0 for row in rows), 1)),
        "mean_relation_grad_feature_norm": float(sum(row["relation_grad_feature_norm"] for row in rows) / max(len(rows), 1)),
        "diagnostic_base_loss": float(diagnostic_loss.detach().float().cpu()),
        "diagnostic_identity_accuracy": id_acc,
        "diagnostic_same_pid_consistency_loss": consistency,
        "selected_pair_total": sum(row["selected_pairs"] for row in rows),
        "input_batch_hash": sha256_json([{key: str(value) for key, value in sorted(batch.items()) if key != "image"} for batch in batches[:-1]]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 不可用，拒绝 CPU 模型回退")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or any(name != EXPECTED for name in names):
        raise RuntimeError(f"PPU 设备不符合要求: {names}")
    device = torch.device("cuda:0")
    probe, b6, factory, cfg, batches, s0, s1, s2, gate1, gate2, loader_meta, candidates = build_plan_and_batches(args.steps)
    results = []
    for name, plan in (("S0", s0), ("S1", s1), ("S2", s2)):
        print(f"中文进度：开始 {name}，步骤={args.steps}", flush=True)
        results.append(run_arm(name, batches, plan, probe, b6, factory, cfg, device))
    valid = all(not item["invalid"] for item in results)
    payload = {
        "status": "PASS_VALID_PAIRED_UPDATES" if valid else "INVALID_PAIR",
        "run_time": args.run_time, "seed": SEED, "epoch": EPOCH, "steps": args.steps,
        "device_names": names, "cpu_model_fallback": False, "lambda": LAMBDA,
        "plan_gates": {"S1": gate1, "S2": gate2},
        "batch_plan_meta": loader_meta,
        "selected_relation_totals": {item["arm"]: item["selected_pair_total"] for item in results},
        "results": results,
        "interpretation": "单一 seed、单一训练状态的短程配对诊断；即使 PASS，也不能等同 Rank-1 或正式训练收益。",
    }
    write_json(args.output / "short_update_utility.json", payload)
    print(json.dumps({"status": payload["status"], "selected_relation_totals": payload["selected_relation_totals"], "diagnostic_base_loss": {item["arm"]: item["diagnostic_base_loss"] for item in results}}, ensure_ascii=False, indent=2), flush=True)
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
