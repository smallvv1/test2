#!/usr/local/bin/python3.10
"""分进程、分级的 PPU 运行时预检。

每个 stage 都在独立 Python/PPU 进程中执行；前一阶段退出后才开始下一阶段。
一旦出现非零退出、SIGABRT、超时或没有 PASS JSON，父进程立即停止后续阶段。
模型计算始终使用 PPU，CPU 只负责调度和保存诊断文件。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
EXPECTED = "PPU-ZW810E"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_imports():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ENGINE))
    sys.path.insert(0, str(ENGINE / "scripts"))
    import gpc_a4_mechanism_probe as probe  # noqa: E402
    import gpc_b0_first_divergence as div  # noqa: E402
    import phase6b_soft_relation_screen as b6  # noqa: E402
    import phase7i_relation_specificity as p7  # noqa: E402
    return probe, div, b6, p7


def run_stage(stage: str, output: Path, seed: int) -> dict:
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
    if str(ENGINE / "scripts") not in sys.path:
        sys.path.insert(0, str(ENGINE / "scripts"))
    import torch

    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value.detach().float()).all().item())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return True

    if stage == "device":
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available()=False")
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if len(names) != 2 or any(name != EXPECTED for name in names):
            raise RuntimeError(f"需要两张 {EXPECTED}，实际为 {names}")
        return {"stage": stage, "device_count": len(names), "device_names": names, "cpu_model_fallback": False}

    if stage.startswith("tensor"):
        index = int(stage[-1])
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
        left = torch.randn(32, 16, device=device, requires_grad=True)
        right = torch.randn(16, 8, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([left, right], lr=1e-3)
        value = (left @ right).square().mean()
        value.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        return {"stage": stage, "device": index, "loss_finite": finite(value), "grad_finite": finite(left.grad) and finite(right.grad), "optimizer_step": True, "cpu_model_fallback": False}

    if stage == "matmul":
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        left = torch.randn(128, 128, device=device)
        right = torch.randn(128, 128, device=device)
        value = left @ right
        torch.cuda.synchronize(device)
        return {"stage": stage, "shape": list(value.shape), "finite": finite(value), "cpu_model_fallback": False}

    if stage in {"triplet_distance", "triplet_backward"}:
        from loss.triplet_loss import TripletLoss, euclidean_dist

        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        features = torch.randn(8, 32, device=device, requires_grad=(stage == "triplet_backward"))
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
        if stage == "triplet_distance":
            distances = euclidean_dist(features, features)
            torch.cuda.synchronize(device)
            return {"stage": stage, "shape": list(distances.shape), "finite": finite(distances), "cpu_model_fallback": False}
        loss, dist_ap, dist_an = TripletLoss()(features, labels)
        loss.backward()
        torch.cuda.synchronize(device)
        return {"stage": stage, "loss": float(loss.detach().float().cpu().item()), "loss_finite": finite(loss), "grad_finite": finite(features.grad), "dist_ap_finite": finite(dist_ap), "dist_an_finite": finite(dist_an), "cpu_model_fallback": False}

    if stage in {"real_forward", "real_triplet_backward", "real_full_optimizer"}:
        probe, div, b6, p7 = configure_imports()
        div.set_seed(seed)
        p7.PHASE = div.SCIENCE_PHASE
        b6.PHASE = div.SCIENCE_PHASE
        p7.SEED = seed
        b6.SEED = seed
        p7.WORLD_SIZE = 2
        b6.WORLD_SIZE = 2
        device = torch.device("cuda:0")
        cfg = probe.load_cfg(div.CONFIG)
        cfg.defrost(); cfg.DATA.NUM_WORKERS = 0; cfg.DATA.PIN_MEMORY = False; cfg.SOLVER.SEED = seed; cfg.freeze()
        dataset = b6.base.build_dataset(cfg)
        transform, _ = probe.build_img_transforms(cfg)
        split = json.loads(div.SPLIT.read_text(encoding="utf-8"))
        state = probe.active_state(dataset, set(map(int, split["train_pids"])))
        index_map = b6.base.build_index_map(dataset)
        loader, meta = b6.base.make_epoch_loader(dataset, transform, state, index_map, 1, 0)
        batch = next(iter(loader))
        model, _ = probe.load_model(div.CONFIG, div.COMMON_INIT, device)
        model.train()
        images = batch["image"].to(device, non_blocking=False)
        pids = batch["pid"].to(device, non_blocking=False)
        cams = batch["camid"].to(device, non_blocking=False)
        clothes = batch["clothes_id"].to(device, non_blocking=False)
        captions = batch["caption_feature"].to(device, non_blocking=False)
        if stage == "real_forward":
            with torch.autocast(device_type="cuda", enabled=True):
                score, feat = model(images, cam_label=cams)
            torch.cuda.synchronize(device)
            return {"stage": stage, "batch_size": len(batch["pid"]), "score_finite": finite(score), "feat0_finite": finite(feat[0]), "batch_plan_sha256": meta["plan_sha256"], "cpu_model_fallback": False}
        if stage == "real_triplet_backward":
            from loss.triplet_loss import TripletLoss
            with torch.autocast(device_type="cuda", enabled=True):
                _score, feat = model(images, cam_label=cams)
                loss, dist_ap, dist_an = TripletLoss()(feat[0], pids)
            loss.backward()
            torch.cuda.synchronize(device)
            return {"stage": stage, "loss": float(loss.detach().float().cpu().item()), "loss_finite": finite(loss), "dist_ap_finite": finite(dist_ap), "dist_an_finite": finite(dist_an), "cpu_model_fallback": False}
        loss_fn, center = b6.base.make_loss(cfg, num_classes=b6.base.NUM_CLASSES)
        optimizer, _ = b6.base.make_optimizer(cfg, model, center)
        scaler = torch.cuda.amp.GradScaler()
        writer = b6.base.MetricCapture()
        with torch.autocast(device_type="cuda", enabled=True):
            score, feat = model(images, cam_label=cams)
        total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=1)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        finite_grads = True
        for parameter in model.parameters():
            if parameter.grad is not None:
                finite_grads = finite_grads and finite(parameter.grad)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        return {"stage": stage, "total_loss": float(total.detach().float().cpu().item()), "loss_finite": finite(total), "grad_finite": bool(finite_grads), "amp_scale_after": float(scaler.get_scale()), "optimizer_step": True, "cpu_model_fallback": False}

    raise ValueError(f"未知 stage: {stage}")


def child_main(args: argparse.Namespace) -> None:
    payload = run_stage(args.stage, args.output, args.seed)
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def parent_main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    stages = ["device", "tensor0", "tensor1", "matmul", "triplet_distance", "triplet_backward", "real_forward", "real_triplet_backward", "real_full_optimizer"]
    results = []
    status = "PASS_ALL_STAGES"
    for stage in stages:
        stage_json = args.output / f"{stage}.json"
        stage_log = args.output / f"{stage}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--run-time", args.run_time, "--child-stage", stage, "--output", str(stage_json), "--seed", str(args.seed)]
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
            stage_log.write_text(completed.stdout, encoding="utf-8")
            signal_name = signal.Signals(-completed.returncode).name if completed.returncode < 0 else None
            row = {"stage": stage, "returncode": completed.returncode, "signal": signal_name, "json_exists": stage_json.is_file(), "log": str(stage_log)}
            if completed.returncode == 0 and stage_json.is_file():
                row["payload"] = json.loads(stage_json.read_text(encoding="utf-8"))
                row["status"] = "PASS"
                results.append(row)
                continue
            row["status"] = "FAIL"
            results.append(row)
            status = f"STOPPED_AT_{stage}"
            break
        except subprocess.TimeoutExpired as exc:
            stage_log.write_text((exc.stdout or "") + "\n中文说明：阶段超时，已停止后续 PPU 预检。\n", encoding="utf-8")
            results.append({"stage": stage, "status": "TIMEOUT", "json_exists": stage_json.is_file(), "log": str(stage_log)})
            status = f"STOPPED_AT_{stage}_TIMEOUT"
            break
    payload = {"status": status, "run_time": args.run_time, "seed": args.seed, "stages": results, "cpu_model_fallback": False}
    write_json(args.output / "staged_preflight_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status != "PASS_ALL_STAGES":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--child-stage")
    parser.add_argument("--stage-output", type=Path)
    args = parser.parse_args()
    if args.child_stage:
        if args.stage_output is None:
            args.stage_output = args.output
        child_main(argparse.Namespace(stage=args.child_stage, output=args.stage_output, seed=args.seed))
        return
    parent_main(args)


if __name__ == "__main__":
    main()
