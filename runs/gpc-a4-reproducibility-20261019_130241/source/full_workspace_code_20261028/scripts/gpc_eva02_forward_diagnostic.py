#!/usr/local/bin/python3.10
"""EVA02 LTCC 真实 PPU 前向/反向分阶段诊断。

该脚本是独立诊断 harness，不修改正式训练处理器、GPC-v1 或历史结果。
F0--F9 每一级在独立子进程运行；PPU 模型计算只在 cuda:0，CPU 仅用于构建、
checkpoint 读取、日志和指纹。前一级失败时停止后续阶段。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
CONFIG = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen/B0_seed1234/config.yml"
CHECKPOINT = ENGINE / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth"
SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
EXPECTED_DEVICE = "PPU-ZW810E"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finite(value: Any) -> bool:
    import torch

    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach().float()).all().item())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def tensor_meta(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "finite": finite(value),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, dict):
        return {str(key): tensor_meta(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [tensor_meta(item) for item in value]
    return str(value) if value is not None else None


def configure_imports():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ENGINE))
    sys.path.insert(0, str(ENGINE / "scripts"))
    import gpc_a4_mechanism_probe as probe  # noqa: E402
    import gpc_b0_first_divergence as divergence  # noqa: E402
    import phase6b_soft_relation_screen as b6  # noqa: E402
    import phase7i_relation_specificity as p7  # noqa: E402
    from model.build import factory  # noqa: E402
    return probe, divergence, b6, p7, factory


def configure_protocol(divergence, b6, p7, seed: int) -> None:
    import gpc_science_rng as science

    phase = f"{science.SCIENCE_RNG_NAMESPACE}|{science.SCIENCE_PROTOCOL_ID}"
    for module in (divergence, b6, p7):
        module.PHASE = phase
        module.SEED = seed
        module.WORLD_SIZE = 2


def load_cfg_and_batch(seed: int):
    import json as json_module
    import torch

    probe, divergence, b6, p7, factory = configure_imports()
    configure_protocol(divergence, b6, p7, seed)
    div = divergence
    cfg = probe.load_cfg(div.CONFIG)
    cfg.defrost()
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = seed
    cfg.freeze()
    dataset = b6.base.build_dataset(cfg)
    transform, _ = probe.build_img_transforms(cfg)
    split = json_module.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    state = probe.active_state(dataset, train_pids)
    index_map = b6.base.build_index_map(dataset)
    loader, meta = b6.base.make_epoch_loader(dataset, transform, state, index_map, 1, 0)
    batch = next(iter(loader))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return probe, divergence, b6, p7, factory, cfg, dataset, meta, batch, device


def load_state(factory, cfg):
    import torch

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint model state 不是 dict: {type(state).__name__}")
    state = {str(key).replace("module.", "", 1): value for key, value in state.items()}
    head = state.get("head.weight")
    if head is None:
        raise KeyError("checkpoint 缺少 head.weight")
    model = factory[cfg.MODEL.NAME](
        pretrained=False,
        config=cfg,
        num_classes=int(head.shape[0]),
        camera_num=12,
    )
    return model, state, payload


def model_cpu(seed: int, strict: bool):
    import torch

    probe, divergence, b6, p7, factory = configure_imports()
    configure_protocol(divergence, b6, p7, seed)
    cfg = probe.load_cfg(CONFIG)
    model, state, payload = load_state(factory, cfg)
    if strict:
        incompatible = model.load_state_dict(state, strict=True)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing, unexpected = list(missing), list(unexpected)
    return model, cfg, state, payload, missing, unexpected


def gpu_inputs(batch, device):
    import torch

    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    cams = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    torch.cuda.synchronize(device)
    return images, pids, cams, clothes, captions


def ensure_ppu(device):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available()=False")
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if len(names) != 2 or any(name != EXPECTED_DEVICE for name in names):
        raise RuntimeError(f"PPU 设备不符合要求: {names}")
    torch.cuda.set_device(device)
    return names


def run_trunk(model, images, cams, device):
    """逐模块执行 Eva.forward_features 的主干路径，返回最终 token 和 RoPE。"""
    import torch
    from timm.layers import apply_keep_indices_nlc

    def mark(name, value):
        torch.cuda.synchronize(device)
        if not finite(value):
            raise FloatingPointError(f"{name} 输出非 finite")
        print(json.dumps({"中文标记": f"PASS {name}", "输出": tensor_meta(value)}, ensure_ascii=False), flush=True)

    x = model.patch_embed(images)
    mark("patch_embed", x)
    if model.cls_token is not None:
        x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        mark("cls_concat", x)
    if model.pos_embed is not None:
        x = x + model.pos_embed + model.camera_xishu * model.camera_embed[cams]
        mark("position_camera", x)
    x = model.pos_drop(x)
    mark("position_dropout", x)
    rot_pos_embed = model.rope.get_embed() if model.rope is not None else None
    if rot_pos_embed is not None:
        mark("rope", rot_pos_embed)
    if model.patch_drop is not None:
        x, keep_indices = model.patch_drop(x)
        if rot_pos_embed is not None and keep_indices is not None:
            rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
        mark("patch_drop", x)
    for index, block in enumerate(model.blocks):
        x = block(x, rope=rot_pos_embed)
        mark(f"block_{index:02d}", x)
    return x, rot_pos_embed


def stage_f0(args):
    model, cfg, state, payload, missing, unexpected = model_cpu(args.seed, strict=False)
    return {
        "stage": "F0_build",
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_training_initial": bool(model.training),
        "blocks": len(getattr(model, "blocks", [])),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "bottleneck_config": bool(cfg.MODEL.BOTTEL_NECK),
        "factory_name": str(cfg.MODEL.NAME),
        "config_sha256": sha256_file(CONFIG),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "cpu_model_fallback": False,
        "note": "F0 只构建模型；未在 CPU 执行模型 forward/backward。",
    }


def stage_f1(args):
    model, cfg, state, payload, missing, unexpected = model_cpu(args.seed, strict=True)
    eval_result = model.eval()
    eval_self = eval_result is model
    train_result = model.train()
    train_self = train_result is model
    return {
        "stage": "F1_checkpoint_strict",
        "strict_load": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "model_eval_returns_self": eval_self,
        "model_train_returns_self": train_self,
        "training_after_train_call": bool(model.training),
        "state_key_count": len(state),
        "checkpoint_payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else None,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "cpu_model_fallback": False,
    }


def common_gpu_stage(args):
    import torch

    probe, divergence, b6, p7, factory, cfg, dataset, meta, batch, device = load_cfg_and_batch(args.seed)
    names = ensure_ppu(device)
    model, state, payload = load_state(factory, cfg)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.train()
    images, pids, cams, clothes, captions = gpu_inputs(batch, device)
    return {
        "probe": probe, "divergence": divergence, "b6": b6, "p7": p7,
        "factory": factory, "cfg": cfg, "dataset": dataset, "meta": meta,
        "batch": batch, "device": device, "names": names, "model": model,
        "images": images, "pids": pids, "cams": cams, "clothes": clothes,
        "captions": captions,
    }


def stage_f2(args):
    values = common_gpu_stage(args)
    import torch

    return {
        "stage": "F2_input_to_ppu",
        "device_names": values["names"],
        "batch_plan_sha256": values["meta"].get("plan_sha256"),
        "batch_size": len(values["batch"]["pid"]),
        "images": tensor_meta(values["images"]),
        "pid": tensor_meta(values["pids"]),
        "camera": tensor_meta(values["cams"]),
        "clothing": tensor_meta(values["clothes"]),
        "caption": tensor_meta(values["captions"]),
        "cuda_synchronized": True,
        "cpu_model_fallback": False,
    }


def stage_f3(args):
    values = common_gpu_stage(args)
    x, rope = run_trunk(values["model"], values["images"], values["cams"], values["device"])
    return {"stage": "F3_eva_backbone", "tokens": tensor_meta(x), "rope": tensor_meta(rope), "cpu_model_fallback": False}


def stage_f4(args):
    values = common_gpu_stage(args)
    model, images, cams, device = values["model"], values["images"], values["cams"], values["device"]
    x, rope = run_trunk(model, images, cams, device)
    x_bio = model.head_bio(x, rope=rope)
    torch_sync = __import__("torch").cuda.synchronize(device)
    x_bio = model.norm_bio(x_bio)
    __import__("torch").cuda.synchronize(device)
    return {
        "stage": "F4_bio_projection_norm",
        "outer_bottleneck_applicable": bool(values["cfg"].MODEL.BOTTEL_NECK),
        "outer_bottleneck_note": "配置 BOTTEL_NECK=false；实际 direct EVA 模型没有外层 BatchNorm。F4 检查 head_bio + norm_bio。",
        "bio_tokens": tensor_meta(x_bio),
        "cpu_model_fallback": False,
    }


def stage_f5(args):
    values = common_gpu_stage(args)
    model, images, cams, device = values["model"], values["images"], values["cams"], values["device"]
    x, rope = run_trunk(model, images, cams, device)
    x_bio = model.norm_bio(model.head_bio(x, rope=rope))
    feat_bio = model.forward_head(x_bio, pre_logits=True)
    cls_score = model.head(feat_bio)
    __import__("torch").cuda.synchronize(device)
    return {"stage": "F5_identity_head", "feature": tensor_meta(feat_bio), "identity_score": tensor_meta(cls_score), "cpu_model_fallback": False}


def stage_f6(args):
    values = common_gpu_stage(args)
    model, images, cams, device = values["model"], values["images"], values["cams"], values["device"]
    x, rope = run_trunk(model, images, cams, device)
    x_bio = model.norm_bio(model.head_bio(x, rope=rope))
    feat_bio = model.forward_head(x_bio, pre_logits=True)
    bio_score = model.head_clip_bio(feat_bio)
    __import__("torch").cuda.synchronize(device)
    return {"stage": "F6_bio_head", "feature": tensor_meta(feat_bio), "bio_score": tensor_meta(bio_score), "cpu_model_fallback": False}


def stage_f7(args):
    values = common_gpu_stage(args)
    with __import__("torch").autocast(device_type="cuda", enabled=True):
        score, feat = values["model"](values["images"], cam_label=values["cams"])
    __import__("torch").cuda.synchronize(values["device"])
    return {"stage": "F7_full_forward", "score": tensor_meta(score), "feature": tensor_meta(feat), "outputs_finite": finite(score) and finite(feat), "cpu_model_fallback": False}


def loss_components(args, include_gpc: bool = False):
    import torch

    values = common_gpu_stage(args)
    model, cfg, device = values["model"], values["cfg"], values["device"]
    loss_fn, center = values["b6"].base.make_loss(cfg, num_classes=values["b6"].base.NUM_CLASSES)
    writer = values["b6"].base.MetricCapture()
    with torch.autocast(device_type="cuda", enabled=True):
        score, feat = model(values["images"], cam_label=values["cams"])
    total = loss_fn(
        score, feat, values["pids"], values["cams"], values["captions"],
        clothes_ids=values["clothes"], train_writer=writer, step=1,
    )
    if include_gpc:
        raise NotImplementedError("F8/F9 按用户要求只验证 CE+Triplet+Bio；GPC 关系诊断保持离线。")
    return values, loss_fn, center, score, feat, total


def stage_f8(args):
    import torch

    values, loss_fn, _center, score, feat, total = loss_components(args)
    total.backward()
    torch.cuda.synchronize(values["device"])
    grads = [parameter.grad for parameter in values["model"].parameters() if parameter.grad is not None]
    return {
        "stage": "F8_ce_triplet_bio_backward",
        "loss": float(total.detach().float().cpu().item()),
        "loss_finite": finite(total),
        "gradient_tensor_count": len(grads),
        "gradient_finite": all(finite(grad) for grad in grads),
        "cpu_model_fallback": False,
    }


def stage_f9(args):
    import torch

    values, loss_fn, center, score, feat, total = loss_components(args)
    optimizer, _ = values["b6"].base.make_optimizer(values["cfg"], values["model"], center)
    # 与真实 Phase5B/Phase7I processor 一致；默认 65536 仅用于诊断时会
    # 人为改变数值条件，不能用来判断正式配置。
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    scale_before = float(scaler.get_scale())
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    grads = [parameter.grad for parameter in values["model"].parameters() if parameter.grad is not None]
    grad_finite = all(finite(grad) for grad in grads)
    if not grad_finite:
        raise FloatingPointError("F9 未缩放梯度非 finite，拒绝 optimizer.step")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(values["device"])
    return {
        "stage": "F9_optimizer_step",
        "loss": float(total.detach().float().cpu().item()),
        "loss_finite": finite(total),
        "gradient_finite": grad_finite,
        "scale_before": scale_before,
        "scale_after": float(scaler.get_scale()),
        "found_inf": False,
        "optimizer_step": True,
        "cpu_model_fallback": False,
    }


STAGES = [
    ("F0_build", stage_f0),
    ("F1_checkpoint_strict", stage_f1),
    ("F2_input_to_ppu", stage_f2),
    ("F3_eva_backbone", stage_f3),
    ("F4_bio_projection_norm", stage_f4),
    ("F5_identity_head", stage_f5),
    ("F6_bio_head", stage_f6),
    ("F7_full_forward", stage_f7),
    ("F8_ce_triplet_bio_backward", stage_f8),
    ("F9_optimizer_step", stage_f9),
]


def child_main(args):
    functions = dict(STAGES)
    payload = functions[args.child_stage](args)
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)


def parent_main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    status = "PASS_ALL_STAGES"
    for stage, _function in STAGES:
        stage_json = args.output / f"{stage}.json"
        stage_log = args.output / f"{stage}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--run-time", args.run_time,
                   "--child-stage", stage, "--output", str(stage_json), "--seed", str(args.seed)]
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, timeout=args.timeout)
            stage_log.write_text(completed.stdout, encoding="utf-8")
            signal_name = signal.Signals(-completed.returncode).name if completed.returncode < 0 else None
            row = {"stage": stage, "returncode": completed.returncode, "signal": signal_name,
                   "json_exists": stage_json.is_file(), "log": str(stage_log)}
            if completed.returncode == 0 and stage_json.is_file():
                row["payload"] = json.loads(stage_json.read_text(encoding="utf-8"))
                row["status"] = "PASS"
                rows.append(row)
                continue
            row["status"] = "FAIL"
            rows.append(row)
            status = f"STOPPED_AT_{stage}"
            break
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            stage_log.write_text(str(output) + "\n中文说明：阶段超时，已停止后续诊断。\n", encoding="utf-8")
            rows.append({"stage": stage, "status": "TIMEOUT", "json_exists": stage_json.is_file(), "log": str(stage_log)})
            status = f"STOPPED_AT_{stage}_TIMEOUT"
            break
    summary = {"run_time": args.run_time, "seed": args.seed, "status": status,
               "stages": rows, "config": str(CONFIG), "checkpoint": str(CHECKPOINT),
               "cpu_model_fallback": False}
    write_json(args.output / "diagnostic_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if status != "PASS_ALL_STAGES":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--child-stage", choices=[name for name, _ in STAGES])
    args = parser.parse_args()
    if args.child_stage:
        child_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
