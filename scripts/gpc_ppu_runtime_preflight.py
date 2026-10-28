#!/usr/local/bin/python3.10
"""PPU runtime 与真实 LTCC Triplet 单步预检。

只做最小 tensor forward/backward/optimizer.step 和一个真实 train-only
batch 的 CE+Triplet+Bio AMP 单步；不启动 epoch，不执行官方评估，不修改
任何 checkpoint 或正式训练处理器。
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
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))

import gpc_a4_mechanism_probe as probe  # noqa: E402
import gpc_b0_first_divergence as div  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402

base = p7.base
EXPECTED = "PPU-ZW810E"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach().float()).all().item())


def device_audit() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available()=False")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or any(name != EXPECTED for name in names):
        raise RuntimeError(f"需要两张 {EXPECTED}，实际为 {names}")
    return {"device_count": len(names), "device_names": names, "cpu_model_fallback": False}


def minimal_device_test(index: int) -> dict[str, Any]:
    torch.cuda.set_device(index)
    device = torch.device(f"cuda:{index}")
    left = torch.randn(32, 16, device=device, requires_grad=True)
    right = torch.randn(16, 8, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([left, right], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    value = (left @ right).square().mean()
    value.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    return {
        "device": index,
        "forward_finite": finite_tensor(value),
        "left_grad_finite": finite_tensor(left.grad),
        "right_grad_finite": finite_tensor(right.grad),
        "loss": float(value.detach().float().cpu().item()),
        "optimizer_step": True,
    }


def configure_science(seed: int) -> None:
    phase = div.SCIENCE_PHASE
    p7.PHASE = phase
    b6.PHASE = phase
    base.PHASE = phase
    p7.SEED = seed
    b6.SEED = seed
    base.SEED = seed
    p7.WORLD_SIZE = 2
    b6.WORLD_SIZE = 2
    base.WORLD_SIZE = 2
    div.set_seed(seed)


def triplet_batch_preflight(seed: int, device: torch.device) -> dict[str, Any]:
    configure_science(seed)
    cfg = probe.load_cfg(div.CONFIG)
    cfg.defrost()
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = seed
    cfg.freeze()
    dataset = base.build_dataset(cfg)
    transform, _ = probe.build_img_transforms(cfg)
    split = json.loads(div.SPLIT.read_text(encoding="utf-8"))
    state = probe.active_state(dataset, set(map(int, split["train_pids"])))
    index_map = base.build_index_map(dataset)
    loader, loader_meta = base.make_epoch_loader(dataset, transform, state, index_map, 1, 0)
    batch = next(iter(loader))
    model, _ = probe.load_model(div.CONFIG, div.COMMON_INIT, device)
    loss_fn, center = base.make_loss(cfg, num_classes=base.NUM_CLASSES)
    optimizer, _ = base.make_optimizer(cfg, model, center)
    scaler = amp.GradScaler(init_scale=base.INITIAL_SCALE)
    model.train()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    cams = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    writer = base.MetricCapture()
    optimizer.zero_grad(set_to_none=True)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=cams)
    total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=1)
    components = dict(getattr(loss_fn, "last_components", {}))
    triplet = components["triplet"]
    triplet_grad = torch.autograd.grad(triplet, feat[0], retain_graph=False, allow_unused=True)[0]
    if triplet_grad is None:
        raise RuntimeError("真实 Triplet 对身份特征没有梯度")
    if not finite_tensor(triplet_grad):
        raise RuntimeError("真实 Triplet 身份特征梯度非有限")
    if not finite_tensor(triplet):
        raise RuntimeError("真实 Triplet loss 非有限")
    triplet_grad_norm = float(torch.linalg.vector_norm(triplet_grad.detach().float()).cpu().item())
    # Use a fresh forward graph for the actual formal-style AMP update.  This
    # avoids reusing the graph consumed by the independent Triplet probe.
    del score, feat, total, triplet, triplet_grad
    torch.cuda.synchronize(device)
    writer = base.MetricCapture()
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=cams)
    total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=2)
    if not finite_tensor(total):
        raise RuntimeError("联合 CE+Triplet+Bio loss 非有限")
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.zeros((), device=device)
    finite_grads = True
    for parameter in model.parameters():
        if parameter.grad is not None:
            finite_grads = finite_grads and finite_tensor(parameter.grad)
            grad_norm = grad_norm + parameter.grad.detach().float().square().sum()
    found_inf = float(base.found_inf(scaler, optimizer))
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    return {
        "seed": seed,
        "batch_size": len(batch["pid"]),
        "batch_plan_sha256": loader_meta["plan_sha256"],
        "total_loss_finite": finite_tensor(total),
        "triplet_loss": float(components["triplet"].detach().float().cpu().item()),
        "triplet_grad_norm": triplet_grad_norm,
        "total_grad_norm": float(torch.sqrt(grad_norm).cpu().item()),
        "parameter_gradients_finite": bool(finite_grads),
        "found_inf": found_inf,
        "amp_scale_after": float(scaler.get_scale()),
        "optimizer_step_called": True,
        "cpu_model_fallback": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    audit = device_audit()
    minimal = [minimal_device_test(index) for index in range(2)]
    torch.cuda.empty_cache()
    real = triplet_batch_preflight(1234, torch.device("cuda:0"))
    payload = {
        "status": "PASS",
        "run_time": args.run_time,
        "device_audit": audit,
        "minimal_ppu_forward_backward": minimal,
        "real_triplet_batch": real,
        "source_sha256": {str(path): sha256_file(path) for path in (ROOT / "scripts/gpc_ppu_runtime_preflight.py", ROOT / "scripts/gpc_b0_first_divergence.py", ENGINE / "loss/make_loss.py", ENGINE / "loss/triplet_loss.py", ENGINE / "scripts/phase5b_id_damage_ds.py")},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ppu_runtime_preflight.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
