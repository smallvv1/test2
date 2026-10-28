#!/usr/local/bin/python3.10
"""固定 Bio+GPC-v1 的跨进程 A/A 可重复性诊断。

该脚本只使用历史 B0 E60 完整状态和 LTCC train-only 采样计划，不加入 NDP，
不读取 query/gallery，不保存新的模型 checkpoint。prepare 阶段一次性物化输入，
两个独立 child 进程从同一个模型/optimizer/scheduler/scaler/RNG 状态运行，
用于区分输入、前向、梯度、AMP 和 optimizer 更新的第一处分叉。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
ORIGINAL_RUN_TIME = "20261015_155822"
WORLD_SIZE_EXPECTED = 2
EPOCH = 61
RANK = 0
LAMBDA_GPC = 0.1
DEFAULT_STEPS = 20

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_digest(value)}
    return value


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode())


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu()
    return sha256_bytes(raw.view(torch.uint8).numpy().tobytes())


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    cpu = value.detach().float().cpu()
    return {
        "shape": list(cpu.shape),
        "dtype": str(value.dtype),
        "sha256": tensor_digest(value),
        "finite": bool(torch.isfinite(cpu).all().item()),
        "mean": float(cpu.mean().item()) if cpu.numel() else 0.0,
        "std": float(cpu.std(unbiased=False).item()) if cpu.numel() else 0.0,
        "min": float(cpu.min().item()) if cpu.numel() else 0.0,
        "max": float(cpu.max().item()) if cpu.numel() else 0.0,
    }


def nested_digest(value: Any) -> str:
    h = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            h.update(b"tensor")
            h.update(str(tuple(item.shape)).encode())
            h.update(str(item.dtype).encode())
            h.update(tensor_digest(item).encode())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=lambda x: str(x)):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(b"seq")
            for child in item:
                visit(child)
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def rng_digest() -> dict[str, str]:
    result = {
        "python": sha256_bytes(pickle.dumps(random.getstate(), protocol=4)),
        "numpy": sha256_bytes(pickle.dumps(np.random.get_state(), protocol=4)),
        "torch_cpu": sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
    }
    try:
        result["torch_cuda"] = sha256_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    except Exception as exc:
        result["torch_cuda"] = f"unavailable:{type(exc).__name__}"
    return result


def relation_key(item: tuple) -> str:
    return json.dumps(canonical(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def relation_digest(values: list[tuple]) -> str:
    return sha256_json([relation_key(item) for item in values])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def install(seed: int, run_dir: Path):
    import gpc_v2_train as v2

    class Args:
        output = OLD_OUTPUT
        run_time = ORIGINAL_RUN_TIME
        arm = "B0"
        stage = "two_seed"
        epochs = 60
        full_data = False
    args = Args()
    args.seed = seed

    split = v2.load_split(OLD_OUTPUT)
    protocol = v2.install_protocol(args, split, run_dir, None)
    cfg = v2.p7.configure()
    if v2.p7.PHASE != f"gpc_v2_ltcc_{ORIGINAL_RUN_TIME}_two_seed_B0_seed{seed}":
        raise RuntimeError(f"GPC-v1 scientific namespace mismatch: {v2.p7.PHASE}")
    return v2, split, protocol, cfg


def freeze_batch(batch: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def batch_fingerprint(batch: dict[str, Any]) -> dict[str, Any]:
    tensors = {k: tensor_digest(v) for k, v in sorted(batch.items()) if torch.is_tensor(v)}
    metadata = {k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in sorted(batch.items()) if not torch.is_tensor(v)}
    return {"tensor_hashes": tensors, "metadata_hash": sha256_json(metadata), "batch_hash": sha256_json({"tensor_hashes": tensors, "metadata": metadata})}


def prepare(seed: int, output: Path, steps: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / f"prepare_seed{seed}"
    v2, split, protocol, cfg = install(seed, run_dir)
    dataset = v2.base.build_dataset(cfg)
    transform_train, _ = v2.base.build_img_transforms(cfg)
    active_state = v2.b6.full_active_state(dataset)
    index_map = v2.base.build_index_map(dataset)
    # install_protocol 的包装器会把 covered_pids 再映射一次；历史 continuation
    # 已审计为使用其保存的原始 loader，避免非连续 LTCC PID 触发二次 remap。
    original_loader = protocol["restore"][2]
    loader, loader_meta = original_loader(dataset, transform_train, active_state, index_map, EPOCH, RANK)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = v2.b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index

    sampled = []
    for iteration, indices in enumerate(loader.batch_sampler, start=1):
        sampled.append([int(x) for x in indices])
        if iteration >= steps + 1:
            break
    iterator = iter(loader)
    batches = []
    selected_plan: dict[int, list[tuple]] = {}
    fingerprints = []
    for iteration in range(1, steps + 2):
        batch = freeze_batch(next(iterator))
        entries = [{"pid": int(batch["pid"][i]), "clothes_id": int(batch["clothes_id"][i]), "image_path": str(batch["image_path"][i])} for i in range(len(batch["pid"]))]
        chosen, matched, info = v2.selected_v2(entries, path_to_index, bank, "B0", EPOCH, RANK, iteration)
        selected = list(chosen["id"])
        if selected:
            selected_plan[iteration] = selected
        batch["_batch_indices"] = sampled[iteration - 1]
        batch["_matched_count"] = int(matched)
        batch["_relation_info"] = {k: v for k, v in info.items() if k != "relation_loss"}
        batches.append(batch)
        fingerprints.append(batch_fingerprint(batch))

    source = torch.load(ROOT / "outputs/gpc_v2_ltcc_20261015_155822/training/two_seed" / f"B0_seed{seed}" / "rank0_state_epoch060.pt", map_location="cpu", weights_only=False)
    state_path = ROOT / "outputs/gpc_v2_ltcc_20261015_155822/training/two_seed" / f"B0_seed{seed}" / "rank0_state_epoch060.pt"
    payload = {
        "seed": seed,
        "epoch": EPOCH,
        "rank": RANK,
        "steps": steps,
        "phase": source["phase"],
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "config_sha256": sha256_file(OLD_OUTPUT / "training/two_seed" / f"B0_seed{seed}" / "config.yml"),
        "split_sha256": sha256_file(OLD_OUTPUT / "development_split_manifest.json"),
        "train_only": True,
        "official_query_gallery_used": False,
        "gpc_v1_preserved": True,
        "lambda_gpc": LAMBDA_GPC,
        "loader_meta": loader_meta,
        "batch_plan_sha256": sha256_json(sampled),
        "selected_relation_sha256": relation_digest([item for iteration in sorted(selected_plan) for item in selected_plan[iteration]]),
        "selected_relation_plan_sha256": sha256_json({str(k): [relation_key(x) for x in selected_plan[k]] for k in sorted(selected_plan)}),
        "batch_fingerprints": fingerprints,
        "batches": batches,
        "selected_plan": selected_plan,
    }
    torch.save(payload, output / f"materialized_seed{seed}.pt")
    summary = {k: payload[k] for k in ("seed", "epoch", "steps", "phase", "state_path", "state_sha256", "config_sha256", "split_sha256", "batch_plan_sha256", "selected_relation_sha256", "selected_relation_plan_sha256", "lambda_gpc", "loader_meta")}
    write_json(output / f"materialized_seed{seed}.json", summary)
    return summary


def ensure_ppu() -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError("PPU unavailable; CPU model fallback is forbidden")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != WORLD_SIZE_EXPECTED or any("PPU-ZW810E" not in name for name in names):
        raise RuntimeError(f"PPU device audit failed: {names}")
    torch.cuda.set_device(0)
    return names


def load_runtime(seed: int, materialized: dict[str, Any], output: Path, device: torch.device):
    import gpc_v2_train as v2

    run_dir = output / f"runtime_seed{seed}"
    v2, _split, _protocol, cfg = install(seed, run_dir)
    state_path = Path(materialized["state_path"])
    source = torch.load(state_path, map_location="cpu", weights_only=False)
    model = v2.base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=v2.base.NUM_CLASSES, camera_num=v2.base.NUM_CAMERAS)
    model.load_state_dict(source["model"], strict=True)
    model.to(device)
    model.train()
    loss_fn, center = v2.base.make_loss(cfg, num_classes=v2.base.NUM_CLASSES)
    optimizer, _center_optimizer = v2.base.make_optimizer(cfg, model, center)
    scheduler = v2.base.create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=v2.base.INITIAL_SCALE)
    optimizer.load_state_dict(source["optimizer"])
    scheduler.load_state_dict(source["scheduler"])
    scaler.load_state_dict(source["scaler"])
    v2.base.restore_rng(source["rng"], device)
    return v2, model, loss_fn, optimizer, scheduler, scaler, source


def selected_parameter_names(model: torch.nn.Module) -> list[str]:
    preferred = [
        "patch_embed.proj.weight",
        "blocks.0.attn.q_proj.weight",
        "blocks.0.attn.proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.22.attn.q_proj.weight",
        "blocks.22.attn.proj.weight",
        "head.weight",
    ]
    names = dict(model.named_parameters())
    return [name for name in preferred if name in names]


def parameter_layer_summary(model: torch.nn.Module, names: list[str], gradients: bool = False) -> dict[str, Any]:
    lookup = dict(model.named_parameters())
    result = {}
    for name in names:
        value = lookup[name].grad if gradients else lookup[name].detach()
        if value is None:
            result[name] = {"present": False}
        else:
            result[name] = tensor_summary(value)
    return result


def optimizer_summary(model: torch.nn.Module, optimizer: torch.optim.Optimizer, names: list[str]) -> dict[str, Any]:
    lookup = dict(model.named_parameters())
    result = {}
    for name in names:
        state = optimizer.state.get(lookup[name], {})
        result[name] = {key: tensor_summary(value) if torch.is_tensor(value) else value for key, value in state.items()}
    return {"sha256": nested_digest(result), "layers": result}


def hook_output(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_summary(value)
    if isinstance(value, (list, tuple)):
        return [hook_output(x) for x in value]
    if isinstance(value, dict):
        return {str(k): hook_output(v) for k, v in value.items()}
    return repr(value)


def install_hooks(model: torch.nn.Module):
    names = ("patch_embed", "blocks.0", "blocks.1", "blocks.22", "head")
    modules = dict(model.named_modules())
    handles = []
    captured = {}
    for name in names:
        module = modules.get(name)
        if module is None:
            continue
        def make_hook(label):
            def hook(_module, _inputs, output):
                captured[label] = hook_output(output)
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles, captured


def found_inf_value(scaler, optimizer) -> float:
    state = scaler._per_optimizer_states.get(id(optimizer), {})
    return sum(float(value.item()) for value in state.get("found_inf_per_device", {}).values())


def feature_gradient_norm(total: torch.Tensor, feature: torch.Tensor) -> float:
    grad = torch.autograd.grad(total, feature, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(grad.detach().float()).cpu())


def run_child(seed: int, label: str, materialized_path: Path, output: Path, steps: int) -> dict[str, Any]:
    device_names = ensure_ppu()
    materialized = torch.load(materialized_path, map_location="cpu", weights_only=False)
    if materialized["seed"] != seed or materialized["steps"] != steps:
        raise RuntimeError("materialized plan seed/steps mismatch")
    state_path = Path(materialized["state_path"])
    source = torch.load(state_path, map_location="cpu", weights_only=False)
    if source["phase"] != materialized["phase"]:
        raise RuntimeError("checkpoint phase mismatch")
    v2, model, loss_fn, optimizer, scheduler, scaler, _source = load_runtime(seed, materialized, output, torch.device("cuda:0"))
    names = selected_parameter_names(model)
    handles, hook_data = install_hooks(model)
    rows = []
    result_path = output / f"{label}.json"
    rows_path = output / f"{label}.jsonl"
    try:
        batches = materialized["batches"]
        selected_plan = materialized["selected_plan"]
        with rows_path.open("w", encoding="utf-8") as rows_stream:
            for step in range(1, steps + 1):
                batch = batches[step - 1]
                selected = list(selected_plan.get(step, []))
                optimizer.zero_grad(set_to_none=True)
                images = batch["image"].to(torch.device("cuda:0"), non_blocking=False)
                pids = batch["pid"].to(torch.device("cuda:0"), non_blocking=False)
                cams = batch["camid"].to(torch.device("cuda:0"), non_blocking=False)
                clothes = batch["clothes_id"].to(torch.device("cuda:0"), non_blocking=False)
                captions = batch["caption_feature"].to(torch.device("cuda:0"), non_blocking=False)
                hook_data.clear()
                with amp.autocast(enabled=True):
                    score, feat = model(images, cam_label=cams)
                forward_summary = {"score": hook_output(score), "feat": hook_output(feat), "sha256": nested_digest({"score": score, "feat": feat})}
                writer = v2.base.MetricCapture()
                base_total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=step)
                z = F.normalize(feat[0].float(), dim=-1)
                relation_values = [1.0 - torch.sum(z[i] * z[j]) for i, j, _key, _state in selected]
                relation = torch.stack(relation_values).mean() if relation_values else z.sum() * 0.0
                total = base_total + LAMBDA_GPC * relation
                scale_before = float(scaler.get_scale())
                grad_norm = feature_gradient_norm(total, feat[0])
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                found_inf = found_inf_value(scaler, optimizer)
                gradient_summary = parameter_layer_summary(model, names, gradients=True)
                gradient_norm_sq = torch.zeros((), device=torch.device("cuda:0"), dtype=torch.float32)
                nonfinite = []
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    if not bool(torch.isfinite(parameter.grad.detach().float()).all().cpu()):
                        nonfinite.append(name)
                    gradient_norm_sq = gradient_norm_sq + parameter.grad.detach().float().pow(2).sum()
                before_model = {name: model.state_dict()[name].detach().clone() for name in names}
                scaler.step(optimizer)
                updated = found_inf == 0.0
                scaler.update()
                torch.cuda.synchronize(torch.device("cuda:0"))
                parameter_update_sq = torch.zeros((), device=torch.device("cuda:0"), dtype=torch.float32)
                for name, old in before_model.items():
                    parameter_update_sq = parameter_update_sq + (model.state_dict()[name].detach().float() - old.float()).pow(2).sum()
                components = getattr(loss_fn, "last_components", {})
                row = {
                    "step": step,
                    "batch_hash": materialized["batch_fingerprints"][step - 1]["batch_hash"],
                    "input_tensor_hashes": materialized["batch_fingerprints"][step - 1]["tensor_hashes"],
                    "selected_relation_sha256": relation_digest(selected),
                    "selected_relation_count": len(selected),
                    "forward_output": forward_summary,
                    "hook_outputs": dict(hook_data),
                    "ce_loss": float(components.get("ce", torch.tensor(float("nan"))).detach().float().cpu()),
                    "triplet_loss": float(components.get("triplet", torch.tensor(float("nan"))).detach().float().cpu()),
                    "bio_loss": float(components.get("bio", torch.tensor(float("nan"))).detach().float().cpu()),
                    "base_loss": float(base_total.detach().float().cpu()),
                    "gpc_loss_raw": float(relation.detach().float().cpu()),
                    "gpc_loss_applied": float((LAMBDA_GPC * relation).detach().float().cpu()),
                    "feature_grad_norm": grad_norm,
                    "parameter_grad_norm": float(torch.sqrt(gradient_norm_sq).cpu()),
                    "gradient_layers": gradient_summary,
                    "nonfinite_gradient_names": nonfinite,
                    "amp_scale_before": scale_before,
                    "found_inf": found_inf,
                    "optimizer_step_executed": updated,
                    "amp_scale_after": float(scaler.get_scale()),
                    "parameter_update_norm_selected": float(torch.sqrt(parameter_update_sq).cpu()) if updated else 0.0,
                    "parameter_layers_after": parameter_layer_summary(model, names, gradients=False),
                    "optimizer_state": optimizer_summary(model, optimizer, names),
                    "rng_after_step": rng_digest(),
                }
                rows.append(row)
                rows_stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                rows_stream.flush()
                print(f"中文进度：{label} seed={seed} step={step} update={int(updated)} found_inf={found_inf} scale={scaler.get_scale()}", flush=True)
        result = {
            "status": "COMPLETED",
            "seed": seed,
            "label": label,
            "steps": steps,
            "device_names": device_names,
            "cpu_model_fallback": False,
            "checkpoint_sha256": materialized["state_sha256"],
            "phase": materialized["phase"],
            "batch_plan_sha256": materialized["batch_plan_sha256"],
            "selected_relation_plan_sha256": materialized["selected_relation_plan_sha256"],
            "selected_relation_sha256": materialized["selected_relation_sha256"],
            "optimizer_updates": sum(int(row["optimizer_step_executed"]) for row in rows),
            "skipped_steps": sum(int(not row["optimizer_step_executed"]) for row in rows),
            "skip_sequence": [row["step"] for row in rows if not row["optimizer_step_executed"]],
            "rows": rows,
        }
        write_json(result_path, result)
        return result
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "seed": seed,
            "label": label,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "rows_completed": len(rows),
            "device_names": device_names,
            "cpu_model_fallback": False,
            "checkpoint_sha256": materialized.get("state_sha256"),
            "partial_rows": rows,
        }
        write_json(result_path, failure)
        raise
    finally:
        for handle in handles:
            handle.remove()


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = {"same_batch_hash": True, "same_input_tensor_hashes": True, "same_relation_hash": True, "step_rows": [], "first_difference": None}
    fields = ("base_loss", "ce_loss", "triplet_loss", "bio_loss", "gpc_loss_raw", "feature_grad_norm", "parameter_grad_norm")
    max_abs = {field: 0.0 for field in fields}
    for a, b in zip(left.get("rows", []), right.get("rows", [])):
        step = a["step"]
        if a.get("batch_hash") != b.get("batch_hash") or a.get("input_tensor_hashes") != b.get("input_tensor_hashes"):
            result["same_batch_hash"] = False
            result["same_input_tensor_hashes"] = False
        if a.get("selected_relation_sha256") != b.get("selected_relation_sha256"):
            result["same_relation_hash"] = False
        diff = {field: abs(float(a[field]) - float(b[field])) for field in fields if isinstance(a.get(field), (int, float)) and isinstance(b.get(field), (int, float))}
        for field, value in diff.items():
            max_abs[field] = max(max_abs[field], value)
        categories = []
        if a.get("forward_output", {}).get("sha256") != b.get("forward_output", {}).get("sha256"):
            categories.append("forward_output")
        if a.get("hook_outputs") != b.get("hook_outputs"):
            categories.append("forward_hook")
        if a.get("gradient_layers") != b.get("gradient_layers"):
            categories.append("gradient")
        if a.get("found_inf") != b.get("found_inf") or a.get("amp_scale_after") != b.get("amp_scale_after"):
            categories.append("amp")
        if a.get("optimizer_step_executed") != b.get("optimizer_step_executed") or a.get("optimizer_state", {}).get("sha256") != b.get("optimizer_state", {}).get("sha256"):
            categories.append("optimizer")
        if a.get("parameter_layers_after") != b.get("parameter_layers_after"):
            categories.append("parameter_state")
        if result["first_difference"] is None and categories:
            result["first_difference"] = {"step": step, "categories": categories}
        result["step_rows"].append({"step": step, "categories": categories, "numeric_abs_delta": diff})
    result["max_abs_delta"] = max_abs
    result["same_skip_sequence"] = left.get("skip_sequence") == right.get("skip_sequence")
    result["same_effective_update_count"] = left.get("optimizer_updates") == right.get("optimizer_updates")
    result["left_updates"] = left.get("optimizer_updates")
    result["right_updates"] = right.get("optimizer_updates")
    return result


def parent(args) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = [prepare(seed, args.output, args.steps) for seed in (1234, 4321)]
    records = []
    overall = 0
    for seed in (1234, 4321):
        materialized = args.output / f"materialized_seed{seed}.pt"
        for label in ("aa0", "aa1"):
            child_dir = args.output / f"seed{seed}_{label}"
            child_dir.mkdir(parents=True, exist_ok=True)
            log_path = child_dir / "child.log"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", "child", "--seed", str(seed), "--label", label, "--materialized", str(materialized), "--output", str(child_dir), "--steps", str(args.steps)]
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("w", encoding="utf-8") as stream:
                proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
            result_path = child_dir / f"{label}.json"
            records.append({"seed": seed, "label": label, "returncode": proc.returncode, "command": cmd, "log": str(log_path), "result": str(result_path), "result_sha256": sha256_file(result_path) if result_path.exists() else None})
            overall = overall or proc.returncode
            if proc.returncode != 0:
                print(f"中文错误：seed={seed} {label} 返回码={proc.returncode}，停止继续启动该 seed 的第二分支。", flush=True)
                break
        pair = {}
        left_path = args.output / f"seed{seed}_aa0/aa0.json"
        right_path = args.output / f"seed{seed}_aa1/aa1.json"
        if left_path.is_file() and right_path.is_file():
            left = json.loads(left_path.read_text(encoding="utf-8")); right = json.loads(right_path.read_text(encoding="utf-8"))
            pair = compare(left, right) if left.get("status") == "COMPLETED" and right.get("status") == "COMPLETED" else {"pair_status": "INVALID_CHILD_RESULT"}
        write_json(args.output / f"aa_pair_seed{seed}.json", {"seed": seed, "pair": pair})
    summary = {"status": "COMPLETED" if overall == 0 else "INCOMPLETE", "run_time": args.run_time, "steps": args.steps, "train_only": True, "official_query_gallery_used": False, "gpc_v1_preserved": True, "materialized": summaries, "children": records, "source_script": str(Path(__file__).resolve()), "source_sha256": sha256_file(Path(__file__).resolve())}
    write_json(args.output / "aa_reproducibility_summary.json", summary)
    return overall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("parent", "child"), required=True)
    ap.add_argument("--run-time", default="")
    ap.add_argument("--seed", type=int, choices=(1234, 4321))
    ap.add_argument("--label", choices=("aa0", "aa1"))
    ap.add_argument("--materialized", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = ap.parse_args()
    if args.mode == "parent":
        raise SystemExit(parent(args))
    result = run_child(args.seed, args.label, args.materialized, args.output, args.steps)
    print(json.dumps({"status": result["status"], "seed": args.seed, "label": args.label, "updates": result["optimizer_updates"], "skips": result["skip_sequence"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
