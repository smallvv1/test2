#!/usr/local/bin/python3.10
"""GPC AMP/FP32 同 batch 精度对照。

这是独立的 train-only 诊断 harness，不修改正式训练处理器。每个条件均
从相同初始化、相同 batch、相同关系选择和相同随机状态开始运行两次：

  original_amp: 正式 AMP 与默认 GradScaler
  fixed_amp:    AMP 与固定 init_scale=1 的 GradScaler
  fp32:         关闭 autocast/scaler，仅用于诊断

比较发生在同一精度条件的两次运行之间，避免把 AMP 与 FP32 的正常精度差
异误判为非确定性。每个损失分量都捕获身份特征层和指定共享骨干参数层梯度。
PPU 负责模型计算，CPU 只做副本、哈希和汇总。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))

import gpc_b0_first_divergence as div  # noqa: E402
import gpc_a4_mechanism_probe as probe  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402

base = p7.base
SEED = 1234
RANK = 0
EPOCH = 1
IDENTITY_COMPONENTS = ("ce", "triplet", "bio", "gpc")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_science() -> None:
    div.configure_science()
    p7.PHASE = div.SCIENCE_PHASE
    b6.PHASE = div.SCIENCE_PHASE
    base.PHASE = div.SCIENCE_PHASE
    p7.SEED = SEED
    b6.SEED = SEED
    base.SEED = SEED
    p7.WORLD_SIZE = 2
    b6.WORLD_SIZE = 2
    base.WORLD_SIZE = 2


def set_seed() -> None:
    div.set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def batch_entries(batch: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "pid": int(batch["pid"][index]),
            "clothes_id": int(batch["clothes_id"][index]),
            "image_path": str(batch["image_path"][index]),
        }
        for index in range(len(batch["pid"]))
    ]


def prepare_active_batch() -> tuple[dict[str, Any], int, dict[str, Any], dict[str, Any]]:
    """Build one fixed active train-only batch once and reuse its CPU tensors."""
    configure_science()
    set_seed()
    cfg = probe.load_cfg(div.CONFIG)
    cfg.defrost()
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = SEED
    cfg.freeze()
    dataset = base.build_dataset(cfg)
    transform_train, _ = probe.build_img_transforms(cfg)
    split = json.loads(div.SPLIT.read_text(encoding="utf-8"))
    state = probe.active_state(dataset, set(map(int, split["train_pids"])))
    index_map = base.build_index_map(dataset)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    loader, loader_meta = base.make_epoch_loader(dataset, transform_train, state, index_map, EPOCH, RANK)
    for iteration, batch in enumerate(loader, start=1):
        entries = batch_entries(batch)
        candidates = p7.candidate_lists(entries, path_to_index, bank)
        count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
        if count > 0:
            relation_meta = {
                "iteration": iteration,
                "candidate_counts": {key: len(value) for key, value in candidates.items()},
                "matched_count": count,
                "selected_hash": div.science.relation_sha256(
                    div.science.science_selection(
                        candidates["id"], count, seed=SEED, epoch=EPOCH, rank=RANK,
                        iteration=iteration, kind="id",
                    )
                ),
                "batch_plan_hash": div.batch_record(batch)["plan_hash"],
                "image_input_digest": div.batch_record(batch)["image_input_digest"],
            }
            return batch, iteration, relation_meta, {"loader_meta": loader_meta, "split_sha256": sha256_file(div.SPLIT)}
    raise RuntimeError("冻结开发训练采样计划中未找到有效 matched-budget GPC batch")


def selected_backbone_names(model: torch.nn.Module) -> list[str]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    preferred = []
    for token in (
        "patch_embed.proj.weight", "blocks.0.norm1.weight", "blocks.0.attn.q_proj.weight",
        "blocks.11.norm1.weight", "blocks.11.attn.q_proj.weight", "head.weight",
    ):
        preferred.extend(name for name in names if name == token)
    if not preferred:
        preferred = names[:4]
    return list(dict.fromkeys(preferred))[:6]


def cpu_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().float().cpu().clone()


def stat(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"present": False}
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"present": True, "finite": True, "norm": 0.0, "max_abs": 0.0, "numel": 0}
    return {
        "present": True,
        "finite": bool(torch.isfinite(flat).all().item()),
        "norm": float(torch.linalg.vector_norm(flat).cpu().item()),
        "max_abs": float(flat.abs().max().cpu().item()),
        "numel": int(flat.numel()),
    }


def scalar_value(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def tensor_delta(left: torch.Tensor | None, right: torch.Tensor | None) -> dict[str, Any]:
    if left is None or right is None:
        return {"present_match": left is None and right is None, "max_abs": None, "rel_max": None}
    diff = (left.float() - right.float()).abs()
    scale = right.float().abs().maximum(torch.tensor(1e-12))
    return {
        "present_match": True,
        "exact": bool(torch.equal(left, right)),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "rel_max": float((diff / scale).max().item()) if diff.numel() else 0.0,
        "left": stat(left),
        "right": stat(right),
    }


def summary(value: Any) -> Any:
    return div.tensor_summary(value)


def run_condition(mode: str, batch: dict[str, Any], iteration: int, relation_meta: dict[str, Any], device: torch.device) -> dict[str, Any]:
    configure_science()
    set_seed()
    model, loss_fn, optimizer, _scheduler, scaler, _cfg = div.load_runtime(device, mode)
    model.train()
    parameter_names = selected_backbone_names(model)
    lookup = dict(model.named_parameters())
    optimizer.zero_grad(set_to_none=True)
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    cams = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    entries = batch_entries(batch)
    # Use the exact fixed batch's candidates; rebuilding only maps the frozen
    # train image identity and cannot change the selected relation list.
    cfg = probe.load_cfg(div.CONFIG)
    cfg.defrost(); cfg.DATA.NUM_WORKERS = 0; cfg.DATA.PIN_MEMORY = False; cfg.freeze()
    dataset = base.build_dataset(cfg)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    candidates = p7.candidate_lists(entries, path_to_index, bank)
    selected = div.science.science_selection(candidates["id"], relation_meta["matched_count"], seed=SEED, epoch=EPOCH, rank=RANK, iteration=iteration, kind="id")
    autocast_enabled = mode != "fp32"

    def forward_losses() -> tuple[Any, Any, dict[str, Any], torch.Tensor, torch.Tensor]:
        writer = base.MetricCapture()
        with amp.autocast(enabled=autocast_enabled):
            local_score, local_feat = model(images, cam_label=cams)
        local_base = loss_fn(local_score, local_feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=iteration)
        local_components = dict(getattr(loss_fn, "last_components", {}))
        local_z = F.normalize(local_feat[0].float(), dim=-1)
        local_values = [1.0 - torch.sum(local_z[i] * local_z[j]) for i, j, _key, _state in selected]
        local_relation = torch.stack(local_values).mean() if local_values else local_z.sum() * 0.0
        return local_score, local_feat, local_components, local_relation, local_base + 0.1 * local_relation

    # Repeated autograd.grad(retain_graph=True) on one PPU graph caused a
    # harness-only SIGABRT.  Each component below therefore gets a fresh graph
    # which is released immediately; this does not alter the formal trainer.
    component_reports: dict[str, Any] = {}
    component_raw: dict[str, dict[str, torch.Tensor | None]] = {}
    for component in IDENTITY_COMPONENTS:
        local_score, local_feat, local_components, local_relation, _local_total = forward_losses()
        loss_value = local_relation if component == "gpc" else local_components.get(component)
        if loss_value is None or not torch.is_tensor(loss_value) or not loss_value.requires_grad:
            component_reports[component] = {"identity_feature": {"present": False}, "parameters": {}}
            component_raw[component] = {"identity_feature": None, **{key: None for key in parameter_names}}
            del local_score, local_feat, local_components, local_relation, loss_value
            torch.cuda.synchronize(device)
            continue
        targets = [local_feat[0], *[lookup[name] for name in parameter_names]]
        grads = torch.autograd.grad(loss_value, targets, retain_graph=False, allow_unused=True)
        feature_grad = cpu_tensor(grads[0])
        parameter_grads = {key: cpu_tensor(value) for key, value in zip(parameter_names, grads[1:])}
        component_raw[component] = {"identity_feature": feature_grad, **parameter_grads}
        component_reports[component] = {
            "loss": scalar_value(loss_value),
            "identity_feature": stat(feature_grad),
            "parameters": {key: stat(value) for key, value in parameter_grads.items()},
        }
        del local_score, local_feat, local_components, local_relation, loss_value, grads
        torch.cuda.synchronize(device)

    # Capture the combined unscaled gradient on a fresh graph, then build a
    # third fresh graph for the actual optimizer update.
    total_score, total_feat, total_components, total_relation, total_value = forward_losses()
    total_grads = torch.autograd.grad(total_value, [total_feat[0], *[lookup[name] for name in parameter_names]], retain_graph=False, allow_unused=True)
    total_raw = {"identity_feature": cpu_tensor(total_grads[0]), **{name: cpu_tensor(value) for name, value in zip(parameter_names, total_grads[1:])}}
    forward_meta = {"score": summary(total_score), "feat": summary(total_feat)}
    loss_meta = {"total": scalar_value(total_value), "ce": scalar_value(total_components.get("ce")),
                 "triplet": scalar_value(total_components.get("triplet")), "bio": scalar_value(total_components.get("bio")),
                 "gpc": scalar_value(total_relation)}
    del total_score, total_feat, total_components, total_relation, total_value, total_grads
    torch.cuda.synchronize(device)

    update_score, update_feat, update_components, update_relation, update_value = forward_losses()
    before_params = div.parameter_fingerprint(model, parameter_names)
    if mode == "fp32":
        update_value.backward()
        found_inf = 0.0
        scale_before = None
        scale_after = None
        amp_grad_scale = 1.0
    else:
        scale_before = float(scaler.get_scale())
        scaler.scale(update_value).backward()
        scaler.unscale_(optimizer)
        found_inf = float(base.found_inf(scaler, optimizer))
        amp_grad_scale = scale_before
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
    if mode == "fp32":
        optimizer.step()
    after_params = div.parameter_fingerprint(model, parameter_names)
    optimizer_after = div.optimizer_fingerprint(optimizer)
    torch.cuda.synchronize(device)
    result = {
        "mode": mode,
        "iteration": iteration,
        "batch_plan_hash": relation_meta["batch_plan_hash"],
        "image_input_digest": relation_meta["image_input_digest"],
        "selected_relation_hash": relation_meta["selected_hash"],
        "selected_count": len(selected),
        "parameter_names": parameter_names,
        "forward": forward_meta,
        "loss": loss_meta,
        "component_gradients": component_reports,
        "total_gradients": {key: stat(value) for key, value in total_raw.items()},
        "amp": {"mode": mode, "scale_before": scale_before, "amp_grad_scale": amp_grad_scale,
                "found_inf": found_inf, "step_called": True, "skipped": bool(found_inf > 0),
                "scale_after": scale_after},
        "parameter_before": before_params,
        "parameter_after": after_params,
        "optimizer_after": optimizer_after,
        "_component_raw": component_raw,
        "_total_raw": total_raw,
    }
    del update_score, update_feat, update_components, update_relation, update_value, model, loss_fn, optimizer
    torch.cuda.empty_cache()
    return result


def compare_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "batch_equal": left["batch_plan_hash"] == right["batch_plan_hash"] and left["image_input_digest"] == right["image_input_digest"],
        "selected_relation_equal": left["selected_relation_hash"] == right["selected_relation_hash"],
        "loss_delta": {key: abs((left["loss"].get(key) or 0.0) - (right["loss"].get(key) or 0.0)) for key in left["loss"]},
        "forward_digest_equal": left["forward"]["score"]["digest"] == right["forward"]["score"]["digest"] and left["forward"]["feat"]["digest"] == right["forward"]["feat"]["digest"],
        "amp_equal": left["amp"] == right["amp"],
        "component_deltas": {},
        "total_gradient_deltas": {},
        "first_gradient_difference": None,
    }
    max_gradient = 0.0
    for component in IDENTITY_COMPONENTS:
        result["component_deltas"][component] = {}
        for target in ["identity_feature", *left["parameter_names"]]:
            delta = tensor_delta(left["_component_raw"][component].get(target), right["_component_raw"][component].get(target))
            result["component_deltas"][component][target] = delta
            current = delta.get("max_abs") or 0.0
            max_gradient = max(max_gradient, current)
            if result["first_gradient_difference"] is None and current > 1e-5:
                result["first_gradient_difference"] = {"loss_component": component, "target": target, **delta}
    for target in ["identity_feature", *left["parameter_names"]]:
        delta = tensor_delta(left["_total_raw"].get(target), right["_total_raw"].get(target))
        result["total_gradient_deltas"][target] = delta
        max_gradient = max(max_gradient, delta.get("max_abs") or 0.0)
    result["max_component_or_total_gradient_abs_delta"] = max_gradient
    result["parameter_update_delta"] = {
        name: abs(float(left["parameter_after"][name]["norm"]) - float(right["parameter_after"][name]["norm"]))
        for name in left["parameter_names"]
    }
    result["optimizer_probe_delta"] = div.max_numeric_delta(left["optimizer_after"], right["optimizer_after"])
    result["pair_status"] = "PASS" if result["batch_equal"] and result["selected_relation_equal"] and result["forward_digest_equal"] and max_gradient <= 1e-5 and result["amp_equal"] else "DIFFERENT"
    return result


def strip_internal(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_internal(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [strip_internal(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 预检失败：torch.cuda.is_available()=False")
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not devices or any(name != "PPU-ZW810E" for name in devices):
        raise RuntimeError(f"PPU 设备不符合要求：{devices}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    batch, iteration, relation_meta, preparation = prepare_active_batch()
    conditions = {}
    for mode in ("amp", "fixed_amp", "fp32"):
        print(f"中文进度：开始 {mode} 同 batch 双重复比较，iteration={iteration}", flush=True)
        left = run_condition(mode, batch, iteration, relation_meta, device)
        right = run_condition(mode, batch, iteration, relation_meta, device)
        conditions[mode] = {"pair": compare_pair(left, right), "runs": [strip_internal(left), strip_internal(right)]}
        print(f"中文结果：{mode} pair_status={conditions[mode]['pair']['pair_status']}", flush=True)
    amp_different = any(conditions[mode]["pair"]["pair_status"] == "DIFFERENT" for mode in ("amp", "fixed_amp"))
    fp32_pass = conditions["fp32"]["pair"]["pair_status"] == "PASS"
    payload = {
        "status": "PASS",
        "run_time": args.run_time,
        "device_names": devices,
        "cpu_model_fallback": False,
        "comparison_rule": "只比较同一条件的两次独立运行；AMP 与 FP32 的正常精度差异不作为非确定性证据。",
        "batch": relation_meta,
        "preparation": preparation,
        "conditions": conditions,
        "interpretation": "AMP_SENSITIVITY_OBSERVED" if amp_different and fp32_pass else "INCONCLUSIVE",
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (ROOT / "scripts/gpc_precision_control_comparison.py", ROOT / "scripts/gpc_b0_first_divergence.py", ROOT / "scripts/gpc_science_rng.py", ENGINE / "scripts/phase5b_id_damage_ds.py", ENGINE / "scripts/phase7i_relation_specificity.py")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "precision_control_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "interpretation": payload["interpretation"], "batch": relation_meta, "pairs": {key: value["pair"] for key, value in conditions.items()}, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
