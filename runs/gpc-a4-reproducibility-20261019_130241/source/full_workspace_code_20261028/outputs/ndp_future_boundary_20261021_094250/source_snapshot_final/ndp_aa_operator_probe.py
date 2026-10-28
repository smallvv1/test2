#!/usr/local/bin/python3.10
"""固定 Bio+GPC-v1 的单步 backward 算子级诊断。

本脚本只读取 A/A 已物化的 train-only batch 与完整 E60 状态，分别在一个独立
PPU 进程中运行 AMP 或隔离 FP32 的一次 forward/backward/optimizer.step。它不接入
NDP、不改变 GPC selector、不保存 checkpoint，目标是区分：输入/forward、缩放前后
梯度、AMP found_inf、optimizer 更新，以及模块输出梯度的首个差异。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.cuda import amp

ROOT = Path("/home/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
import ndp_aa_reproducibility as aa  # noqa: E402


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if torch.is_tensor(value):
        return tensor_summary(value)
    return value


def tensor_summary(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"present": False}
    x = value.detach().float().cpu()
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    finite = torch.isfinite(x)
    return {
        "present": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(finite.all().item()),
        "max_abs": float(x.abs().max().item()) if x.numel() else 0.0,
        "mean": float(x.mean().item()) if x.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(x).item()) if x.numel() else 0.0,
    }


def nested_summary(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_summary(value)
    if isinstance(value, (list, tuple)):
        return [nested_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(k): nested_summary(v) for k, v in value.items()}
    return repr(value)


def install_hooks(model: torch.nn.Module):
    # A/A 使用的是 factory[cfg.MODEL.NAME] 的直接 EVA02 模型，不是
    # build_transformer wrapper；因此实际模块名没有 ``base.`` 前缀。
    # 记录每一个 Transformer block，便于在反向传播链上找第一个明显分叉；
    # 这仍然只保存模块输出梯度，不保存整网激活。
    targets = ("patch_embed",) + tuple(f"blocks.{i}" for i in range(23)) + (
        "blocks.22.norm1", "blocks.22.attn", "blocks.22.attn.q_proj",
        "blocks.22.attn.k_proj", "blocks.22.attn.v_proj", "blocks.22.attn.norm",
        "blocks.22.attn.proj", "blocks.22.norm2", "blocks.22.mlp",
        "blocks.22.mlp.fc1_g", "blocks.22.mlp.fc1_x", "blocks.22.mlp.norm",
        "blocks.22.mlp.fc2", "norm", "bottleneck", "head",
    )
    modules = dict(model.named_modules())
    captures: dict[str, dict[str, Any]] = {}
    handles = []

    def make_hook(label: str):
        def hook(_module, _inputs, output):
            entry = {"forward": nested_summary(output), "tensors": []}

            def retain(item: Any):
                if torch.is_tensor(item):
                    if item.requires_grad:
                        item.retain_grad()
                    entry["tensors"].append(item)
                elif isinstance(item, (list, tuple)):
                    for child in item:
                        retain(child)
                elif isinstance(item, dict):
                    for child in item.values():
                        retain(child)

            retain(output)
            captures[label] = entry
        return hook

    for name in targets:
        module = modules.get(name)
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(name)))
    return captures, handles


def parameter_summary(model: torch.nn.Module, scaled: bool = False) -> dict[str, Any]:
    preferred_prefixes = (
        "patch_embed.proj", "blocks.0", "blocks.1", "blocks.11", "blocks.22",
        "norm", "bottleneck", "head",
    )
    result: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if not name.startswith(preferred_prefixes):
            continue
        result[name] = tensor_summary(parameter.grad)
    return result


def parameter_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """保存少量可复核层的实际梯度张量，避免只凭范数判断分叉。"""
    prefixes = (
        "patch_embed.proj.weight", "blocks.0.attn.q_proj.weight",
        "blocks.0.attn.proj.weight", "blocks.11.attn.q_proj.weight",
        "blocks.22.attn.q_proj.weight", "blocks.22.attn.proj.weight", "head.weight",
    )
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in prefixes and parameter.grad is not None
    }


def optimizer_delta(before: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, Any]:
    result = {}
    lookup = dict(model.named_parameters())
    for name, old in before.items():
        now = lookup[name].detach()
        result[name] = tensor_summary(now - old)
    return result


def run(args: argparse.Namespace) -> None:
    device_names = aa.ensure_ppu()
    materialized = torch.load(args.materialized, map_location="cpu", weights_only=False)
    if int(materialized["seed"]) != args.seed:
        raise RuntimeError("materialized seed mismatch")
    device = torch.device("cuda:0")
    v2, model, loss_fn, optimizer, _scheduler, scaler, source = aa.load_runtime(
        args.seed, materialized, args.output, device,
    )
    model.train()
    batch = materialized["batches"][0]
    selected = list(materialized["selected_plan"].get("1", materialized["selected_plan"].get(1, [])))
    images = batch["image"].to(device)
    pids = batch["pid"].to(device)
    cams = batch["camid"].to(device)
    clothes = batch["clothes_id"].to(device)
    captions = batch["caption_feature"].to(device)
    captures, handles = install_hooks(model)
    try:
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=args.mode == "amp"):
            score, feat = model(images, cam_label=cams)
            writer = v2.base.MetricCapture()
            base_total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=1)
            z = torch.nn.functional.normalize(feat[0].float(), dim=-1)
            relation_values = [1.0 - torch.sum(z[i] * z[j]) for i, j, _key, _state in selected]
            gpc = torch.stack(relation_values).mean() if relation_values else z.sum() * 0.0
            total = base_total + aa.LAMBDA_GPC * gpc
        feat[0].retain_grad()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if name.startswith(("patch_embed.proj", "blocks.0", "blocks.1", "blocks.11", "blocks.22", "norm", "bottleneck", "head"))}
        scale_before = float(scaler.get_scale())
        if args.mode == "amp":
            scaler.scale(total).backward()
            scaled_grads = parameter_summary(model)
            scaled_tensors = parameter_tensors(model)
            scaler.unscale_(optimizer)
            unscaled_grads = parameter_summary(model)
            unscaled_tensors = parameter_tensors(model)
            found_inf = aa.found_inf_value(scaler, optimizer)
            if args.no_optimizer_step:
                scale_after = scale_before
            else:
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
        else:
            total.backward()
            scaled_grads = parameter_summary(model)
            unscaled_grads = parameter_summary(model)
            scaled_tensors = parameter_tensors(model)
            unscaled_tensors = parameter_tensors(model)
            found_inf = 0.0
            if not args.no_optimizer_step:
                optimizer.step()
            scale_after = scale_before
        torch.cuda.synchronize(device)
        module_backward = {}
        for name, entry in captures.items():
            module_backward[name] = {
                "forward": entry["forward"],
                "output_grads": [tensor_summary(t.grad) for t in entry["tensors"]],
            }
        result = {
            "status": "PASS",
            "mode": args.mode,
            "seed": args.seed,
            "device_names": device_names,
            "checkpoint": materialized["state_path"],
            "checkpoint_sha256": materialized["state_sha256"],
            "source_phase": source["phase"],
            "materialized_batch_hash": materialized["batch_fingerprints"][0]["batch_hash"],
            "materialized_input_tensor_hashes": materialized["batch_fingerprints"][0]["tensor_hashes"],
            "selected_relation_sha256": aa.relation_digest(selected),
            "selected_relation_count": len(selected),
            "losses": {
                "base": float(base_total.detach().float().cpu()),
                "gpc": float(gpc.detach().float().cpu()),
                "total": float(total.detach().float().cpu()),
                "writer": safe(getattr(writer, "values", {})),
            },
            "forward": {
                "score": nested_summary(score),
                "feat": nested_summary(feat),
            },
            "feature_gradient": tensor_summary(feat[0].grad),
            "scaled_gradients": scaled_grads,
            "unscaled_gradients": unscaled_grads,
            "module_backward": module_backward,
            "amp": {
                "enabled": args.mode == "amp",
                "scale_before": scale_before,
                "found_inf": found_inf,
                "scale_after": scale_after,
                "optimizer_step_executed": bool((not args.no_optimizer_step) and found_inf == 0.0),
            },
            "optimizer_parameter_delta": optimizer_delta(before, model),
            "ddp": {
                "mode": "single_ppu_operator_probe",
                "all_reduce_before": "not_applicable",
                "all_reduce_after": "not_applicable",
                "note": "DDP all-reduce 需在独立双PPU探针中测量；本结果只隔离单卡反向路径。",
            },
        }
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scaled_gradients": scaled_tensors,
                "unscaled_gradients": unscaled_tensors,
                "module_output_grads": {
                    name: [t.grad.detach().cpu().clone() for t in entry["tensors"] if t.grad is not None]
                    for name, entry in captures.items()
                },
            },
            args.output / f"{args.label}.tensors.pt",
        )
        tmp = args.output / f"{args.label}.json.tmp"
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        tmp.replace(args.output / f"{args.label}.json")
        print(json.dumps({"status": "PASS", "mode": args.mode, "seed": args.seed, "found_inf": found_inf, "gpc_count": len(selected)}, ensure_ascii=False))
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--materialized", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=("amp", "fp32"), required=True)
    ap.add_argument("--no-optimizer-step", action="store_true", help="只做 forward/backward，不执行 optimizer.step")
    args = ap.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{args.label}.error.json").write_text(json.dumps({"status": "FAIL", "error": repr(exc)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
