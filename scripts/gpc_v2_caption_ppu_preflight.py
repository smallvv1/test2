#!/usr/bin/env python3
"""GPC-v2 恢复后的 Caption/Bio/GPC 双 PPU 实际训练前预检。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda import amp

ROOT = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
import phase5b_id_damage_ds as base  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rank, device = base.init_dist()
    result: dict = {"run_time": args.run_time, "rank": rank, "world_size": 2, "status": "FAIL", "checks": {}}
    try:
        if torch.cuda.device_count() != 2:
            raise RuntimeError(f"需要双 PPU，实际可见设备数={torch.cuda.device_count()}")
        cfg = base.configure()
        cfg.defrost()
        cfg.DATA.ROOT = "/home/data"
        cfg.DATA.CAPTION_DIR = "/data/data/DIFFER_TextCaption/LTCC_ReID"
        cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
        cfg.DATA.BIO_INDEX = "0"
        cfg.DATA.NOBIO_INDEX = [1, 2]
        cfg.DATA.SUMMERY_TEXT = True
        cfg.DATA.NUM_WORKERS = 0
        cfg.DATA.PIN_MEMORY = False
        cfg.SOLVER.SEED = 1234
        cfg.SOLVER.MAX_EPOCHS = 60
        cfg.freeze()
        result["config"] = {
            "dataset_root": cfg.DATA.ROOT,
            "caption_dir": cfg.DATA.CAPTION_DIR,
            "caption_model": cfg.DATA.TEXT_MODEL_VERSION,
            "bio_index": [0],
            "nonbio_index": [1, 2],
            "summary_text": True,
            "loss_type": list(cfg.MODEL.LOSS_TYPE),
            "optimizer": str(cfg.SOLVER.OPTIMIZER_NAME),
            "base_lr": float(cfg.SOLVER.BASE_LR),
            "max_epochs": int(cfg.SOLVER.MAX_EPOCHS),
        }
        result["device"] = {"device": str(device), "name": torch.cuda.get_device_name(device.index)}
        state, init_audit = b6.load_common_init()
        model, loss_fn, optimizer, scheduler, scaler = base.setup_runtime(cfg, state, device)
        model_parameter_devices = sorted({str(value.device) for value in model.parameters()})
        optimizer_parameter_devices = sorted({str(value.device) for group in optimizer.param_groups for value in group["params"]})
        if model_parameter_devices != [str(device)] or optimizer_parameter_devices != [str(device)]:
            raise RuntimeError(f"检测到模型或 optimizer 不在当前 PPU：model={model_parameter_devices}, optimizer={optimizer_parameter_devices}")
        result["checks"]["model_optimizer_on_ppu"] = True
        result["common_init"] = init_audit

        dataset = base.build_dataset(cfg)
        transform_train, _ = base.build_img_transforms(cfg)
        index_map = base.build_index_map(dataset)
        active_state = b6.full_active_state(dataset)
        loader, loader_meta = base.make_epoch_loader(dataset, transform_train, active_state, index_map, epoch=1, rank=rank)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        bank = b6.parse_relation_bank(dataset, cfg)
        bank["path_to_index"] = path_to_index
        selected_examples = []
        for batch_index, candidate_batch in enumerate(loader):
            entries = [{
                "pid": int(candidate_batch["pid"][i]),
                "clothes_id": int(candidate_batch["clothes_id"][i]),
                "image_path": str(candidate_batch["image_path"][i]),
            } for i in range(len(candidate_batch["pid"]))]
            chosen, matched_count, info = p7.selected_relation_pairs(entries, path_to_index, bank, "I3", 1, rank, batch_index)
            if matched_count > 0 and chosen["id"]:
                selected_info = {
                    "batch_index": batch_index,
                    "matched_count": matched_count,
                    "candidate_counts": info["candidate_counts"],
                    "selected_count": len(chosen["id"]),
                    "same_clothes_selected": sum(int(state[1] == state[2]) for _i, _j, _key, state in chosen["id"]),
                    "cross_clothes_selected": sum(int(state[1] != state[2]) for _i, _j, _key, state in chosen["id"]),
                    "pairs": chosen["id"],
                }
                selected_examples.append((candidate_batch, selected_info))
                if len(selected_examples) >= 3:
                    break
        if len(selected_examples) < 3:
            raise RuntimeError(f"只找到 {len(selected_examples)} 个有效 I3 batch，需要至少 3 个连续有效更新")
        selected_batch, selected_info = selected_examples[0]
        result["batch"] = {"loader_meta": loader_meta, "gpc_selector": "Phase7I I3", **selected_info}
        images = selected_batch["image"].to(device, non_blocking=False)
        pids = selected_batch["pid"].to(device, non_blocking=False)
        camids = selected_batch["camid"].to(device, non_blocking=False)
        clothes = selected_batch["clothes_id"].to(device, non_blocking=False)
        captions = selected_batch["caption_feature"].to(device, non_blocking=False)
        if not all(value.device == device for value in (images, pids, camids, clothes, captions)):
            raise RuntimeError("输入 batch 未全部进入 PPU")

        # 预检 A：独立 Bio forward/backward。此处不复用后续 GPC graph。
        model.train()
        independent_model = model.module
        optimizer.zero_grad(set_to_none=True)
        writer = base.MetricCapture()
        with amp.autocast(enabled=True):
            score_bio, feat_bio = independent_model(images, cam_label=camids)
        base_total = loss_fn(score_bio, feat_bio, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=1)
        components = getattr(loss_fn, "last_components", {})
        bio_loss = components.get("bio")
        if not finite_tensor(base_total) or bio_loss is None or not finite_tensor(bio_loss):
            raise RuntimeError("Bio loss forward 非有限或缺少组件")
        scaler.scale(bio_loss).backward()
        bio_gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not bio_gradients or not all(finite_tensor(gradient) for gradient in bio_gradients):
            raise RuntimeError("Bio loss backward 产生非有限梯度或没有梯度")
        result["bio_loss"] = {"value": scalar(bio_loss), "total_with_ce_triplet": scalar(base_total), "forward_finite": True, "backward_finite": True, "gradient_count": len(bio_gradients), "gradient_devices": sorted({str(gradient.device) for gradient in bio_gradients})}
        result["checks"]["bio_loss_forward_backward"] = True
        optimizer.zero_grad(set_to_none=True)

        # 预检 B：新的独立 forward，只计算 GPC-v1 relation。
        with amp.autocast(enabled=True):
            _score_gpc, feat_gpc = independent_model(images, cam_label=camids)
        normalized = torch.nn.functional.normalize(feat_gpc[0].float(), dim=-1)
        relation_values = [1.0 - torch.sum(normalized[i] * normalized[j]) for i, j, _key, _state in selected_info["pairs"]]
        relation_loss = torch.stack(relation_values).mean()
        if not finite_tensor(relation_loss):
            raise RuntimeError("GPC relation forward 非有限")
        scaler.scale(relation_loss).backward()
        relation_gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not relation_gradients or not all(finite_tensor(gradient) for gradient in relation_gradients):
            raise RuntimeError("GPC relation backward 产生非有限梯度或没有梯度")
        result["gpc_loss"] = {"value": scalar(relation_loss), "formula": "mean(1-cos(normalized feat[0]))", "forward_finite": True, "backward_finite": True, "gradient_count": len(relation_gradients), "gradient_devices": sorted({str(gradient.device) for gradient in relation_gradients}), "matched_pair_count": len(relation_values)}
        result["checks"]["gpc_loss_forward_backward"] = True
        result["checks"]["independent_checks_bypass_ddp_reducer"] = True
        optimizer.zero_grad(set_to_none=True)

        # 预检 C：每个 batch 都是正式 processor 同构的一次 forward/联合 loss/一次 backward/step。
        combined_steps = []
        for step_index, (step_batch, step_info) in enumerate(selected_examples, start=1):
            step_images = step_batch["image"].to(device, non_blocking=False)
            step_pids = step_batch["pid"].to(device, non_blocking=False)
            step_camids = step_batch["camid"].to(device, non_blocking=False)
            step_clothes = step_batch["clothes_id"].to(device, non_blocking=False)
            step_captions = step_batch["caption_feature"].to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            before = next(model.parameters()).detach().clone()
            writer = base.MetricCapture()
            with amp.autocast(enabled=True):
                score_step, feat_step = model(step_images, cam_label=step_camids)
            base_step = loss_fn(score_step, feat_step, step_pids, step_camids, step_captions, clothes_ids=step_clothes, train_writer=writer, step=step_index + 2)
            normalized_step = torch.nn.functional.normalize(feat_step[0].float(), dim=-1)
            relation_step = torch.stack([1.0 - torch.sum(normalized_step[i] * normalized_step[j]) for i, j, _key, _state in step_info["pairs"]]).mean()
            total = base_step + 0.1 * relation_step
            if not finite_tensor(total):
                raise RuntimeError(f"联合 loss forward 非有限：step={step_index}")
            scaler.scale(total).backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(finite_tensor(gradient) for gradient in gradients):
                raise RuntimeError(f"联合 loss backward 梯度非有限：step={step_index}")
            scaler.step(optimizer)
            found_inf = sum(float(value.item()) for value in scaler._per_optimizer_states.get(id(optimizer), {}).get("found_inf_per_device", {}).values())
            scaler.update()
            after = next(model.parameters()).detach().clone()
            changed = not torch.equal(before, after)
            if not changed:
                raise RuntimeError(f"optimizer.step 未改变模型参数：step={step_index}")
            if not math.isfinite(found_inf) or found_inf != 0.0 or not math.isfinite(float(scaler.get_scale())):
                raise RuntimeError(f"AMP found_inf/skipped 异常：step={step_index}, found_inf={found_inf}, scale={scaler.get_scale()}")
            if not all(finite_tensor(parameter) for parameter in model.parameters()):
                raise RuntimeError(f"optimizer.step 后模型参数非有限：step={step_index}")
            combined_steps.append({"step": step_index, "total": scalar(total), "bio": scalar(getattr(loss_fn, "last_components", {}).get("bio", base_step * 0.0)), "gpc": scalar(relation_step), "relation_lambda": 0.1, "optimizer_step_changed_parameters": changed, "found_inf": found_inf, "amp_scale": float(scaler.get_scale()), "skipped_step": False, "gradient_count": len(gradients)})
        result["combined_steps"] = combined_steps
        result["combined_step"] = combined_steps[-1]
        result["checks"].update({"combined_optimizer_step": True, "amp_stable": True, "no_nan_inf": True})
        dist.barrier()
        gathered = [None, None]
        dist.all_gather_object(gathered, {"rank": rank, "device": str(device), "name": torch.cuda.get_device_name(device.index), "status": "PASS"})
        result["ranks"] = gathered
        result["status"] = "PASS"
        if rank == 0:
            print("GPC_V2_CAPTION_BIO_GPC_PPU_PREFLIGHT_PASS " + json.dumps(result, ensure_ascii=False, sort_keys=True, default=str), flush=True)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print("中文结论：Caption/Bio/GPC 双 PPU 预检失败；禁止启动正式训练。", flush=True)
        print("原始错误：" + repr(exc), flush=True)
        if rank == 0:
            (output_root / "caption_bio_gpc_ppu_preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()
        return 2
    if rank == 0:
        (output_root / "caption_bio_gpc_ppu_preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    base.cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
