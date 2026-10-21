#!/usr/bin/env python3
"""PPU LayerNorm 的严格固定输入/上游梯度重复性探针。

旧版本在每次 repeat 内调用 ``randn_like`` 生成新的 upstream，旧的 LayerNorm
差异因此标记为 INVALID_UPSTREAM_CONTROL。本版本在循环外物化 x/up/参数，逐次
核对 hash，并分别测试通用 LayerNorm、EVA02 block22.norm1、AMP/FP32 和 CPU
参考。只建立反向图，不执行 optimizer.step。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
sys.path.insert(0, str(ROOT / "scripts"))
import ndp_real_trajectory_activation as act  # noqa: E402
import gpc_v2_train as v2  # noqa: E402


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_hash(module: torch.nn.Module) -> str:
    values = [(name, tensor_hash(value)) for name, value in module.state_dict().items()]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def finite_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    aa, bb = a.double().flatten(), b.double().flatten()
    diff = (aa - bb).abs()
    denom = torch.linalg.vector_norm(aa).clamp_min(1e-30)
    denom2 = torch.linalg.vector_norm(bb).clamp_min(1e-30)
    return {
        "max_abs": float(diff.max()),
        "relative_l2": float(torch.linalg.vector_norm(aa - bb) / denom),
        "cosine": float(torch.dot(aa, bb) / (denom * denom2)),
        "finite_a": bool(torch.isfinite(aa).all()),
        "finite_b": bool(torch.isfinite(bb).all()),
    }


def fixed_inputs(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=generator, dtype=torch.float32)
    upstream = torch.randn(shape, generator=generator, dtype=torch.float32)
    return x, upstream


def run_module(module, x_cpu, upstream_cpu, device, amp):
    module.zero_grad(set_to_none=True)
    x = x_cpu.to(device, non_blocking=False).clone().requires_grad_(True)
    upstream = upstream_cpu.to(device, non_blocking=False).clone()
    with torch.autocast(device_type="cuda", enabled=amp):
        output = module(x)
    output.backward(upstream)
    torch.cuda.synchronize()
    param_grads = {}
    for name, value in module.named_parameters():
        param_grads[name] = None if value.grad is None else value.grad.detach().cpu().clone()
    return {
        "input_hash": tensor_hash(x_cpu),
        "upstream_hash": tensor_hash(upstream_cpu),
        "output_hash": tensor_hash(output),
        "output": output.detach().cpu().clone(),
        "input_grad": x.grad.detach().cpu().clone(),
        "param_grads": param_grads,
    }


def compare_runs(runs):
    control = runs[0]
    rows = []
    for idx, run in enumerate(runs[1:], start=1):
        same_input = run["input_hash"] == control["input_hash"]
        same_upstream = run["upstream_hash"] == control["upstream_hash"]
        row = {"repeat": idx, "same_input": same_input, "same_upstream": same_upstream}
        if not (same_input and same_upstream):
            row["status"] = "INVALID_UPSTREAM_CONTROL"
            rows.append(row)
            continue
        row["status"] = "VALID_COMPARISON"
        row["output"] = finite_stats(control["output"], run["output"])
        row["input_grad"] = finite_stats(control["input_grad"], run["input_grad"])
        row["parameter_grads"] = {
            name: finite_stats(control["param_grads"][name], run["param_grads"][name])
            for name in control["param_grads"]
            if control["param_grads"][name] is not None and run["param_grads"].get(name) is not None
        }
        rows.append(row)
    return rows


def cpu_reference(shape, seed, eps, repeats=3):
    x_cpu, upstream_cpu = fixed_inputs(shape, seed)
    module = torch.nn.LayerNorm(shape[-1], eps=eps).float()
    module.weight.data.fill_(1.0); module.bias.data.zero_()
    results = []
    for _ in range(repeats):
        module.zero_grad(set_to_none=True)
        x = x_cpu.clone().requires_grad_(True)
        y = module(x); y.backward(upstream_cpu)
        results.append((y.detach(), x.grad.detach(), module.weight.grad.detach().clone()))
    return {
        "input_hash": tensor_hash(x_cpu), "upstream_hash": tensor_hash(upstream_cpu),
        "repeat_stats": [finite_stats(results[0][i], results[j][i]) for j in range(1, len(results)) for i in range(3)],
    }


def probe_group(name, module, shape, seed, device, repeats, amp):
    x_cpu, upstream_cpu = fixed_inputs(shape, seed)
    initial_state_hash = state_hash(module)
    runs = [run_module(module, x_cpu, upstream_cpu, device, amp) for _ in range(repeats)]
    return {
        "name": name, "shape": list(shape), "amp": amp, "dtype": "float32 inputs",
        "input_hash": tensor_hash(x_cpu), "upstream_hash": tensor_hash(upstream_cpu),
        "parameter_initial_hash": initial_state_hash,
        "input_strides": list(x_cpu.stride()), "upstream_strides": list(upstream_cpu.stride()),
        "repeat_hashes": [{"input": r["input_hash"], "upstream": r["upstream_hash"], "output": r["output_hash"]} for r in runs],
        "comparisons": compare_runs(runs),
        "parameter_unchanged": state_hash(module) == initial_state_hash,
    }


def load_eva_norm(device, seed):
    checkpoint = OLD_OUTPUT / f"training/two_seed/B0_seed{seed}/rank0_state_epoch060.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    class Args:
        output = OLD_OUTPUT
        run_time = "20261015_155822"
        arm = "B0"
        stage = "two_seed"
        epochs = 60
        full_data = False

    Args.seed = seed
    split = json.loads((OLD_OUTPUT / "development_split_manifest.json").read_text(encoding="utf-8"))
    v2.install_protocol(Args, split, ROOT / "outputs/ndp_layernorm_probe_model_load", None)
    cfg = v2.p7.configure()
    source = act.restore_numpy_pickle(checkpoint)
    state = source["model"] if isinstance(source, dict) and "model" in source else source
    model = act.load_model(cfg, state, device)
    model.eval()
    return model.blocks[22].norm1, checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint-seed", type=int, default=1234, choices=(1234, 4321))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--include-eva-norm", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU/CUDA 不可用，禁止 CPU model fallback")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    started = time.time()
    results = []
    for amp in (False, True):
        module = torch.nn.LayerNorm(1024).to(device).float()
        module.weight.data.fill_(1.0); module.bias.data.zero_()
        results.append(probe_group("independent_layernorm", module, (2, 64, 1024), 20261021, device, args.repeats, amp))
    checkpoint_meta = None
    if args.include_eva_norm:
        eva_module, checkpoint = load_eva_norm(device, args.checkpoint_seed)
        for amp in (False, True):
            results.append(probe_group("eva02_block22_norm1", eva_module, (2, 257, 1024), 20261022, device, args.repeats, amp))
        checkpoint_meta = {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    out = {
        "status": "PASS_FIXED_INPUT_UPSTREAM_PROBE",
        "device": str(device), "torch": torch.__version__, "python": sys.executable,
        "platform": platform.platform(), "repeats": args.repeats,
        "cpu_reference": cpu_reference((2, 64, 1024), 20261021, torch.nn.LayerNorm(1024).eps, args.repeats),
        "groups": results, "checkpoint": checkpoint_meta,
        "old_layernorm_result_status": "INVALID_UPSTREAM_CONTROL",
        "optimizer_step_executed": False,
        "scope_limit": "EVA02 direct norm1 is isolated; this is not a full block22 fused backward reproduction",
        "elapsed_seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
