#!/usr/local/bin/python3.10
"""受控 U0/U1 配对更新诊断。

U0 只使用 CE+Triplet+Bio，U1 额外使用 0.1*GPC。两个分支从同一
model/optimizer/scaler 和随机状态开始；任何一侧发生 AMP skip，该状态在
原始 AMP 条件下标为 INVALID_PAIR，不用不同更新数凑样本。fixed_amp 是
单独的安全 scale 对照，不改正式训练配置。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/data/DIFFER")
sys.path.insert(0, "/data/DIFFER/scripts")

import gpc_update_utility_diagnostic as old  # noqa: E402
from torch.cuda import amp  # noqa: E402


def set_all_seed(seed: int) -> None:
    old.configure_science(seed)
    old.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def rng_digest() -> str:
    values = [
        pickle.dumps(random.getstate(), protocol=4),
        pickle.dumps(np.random.get_state(), protocol=4),
        torch.get_rng_state().cpu().numpy().tobytes(),
    ]
    try:
        values.extend(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    except Exception:
        pass
    return hashlib.sha256(b"".join(values)).hexdigest()


def make_scaler(mode: str, payload: dict[str, Any] | None):
    if mode == "original_amp":
        scaler = amp.GradScaler(init_scale=old.base.INITIAL_SCALE)
        if payload is not None:
            scaler.load_state_dict(payload["scaler"])
        return scaler
    return amp.GradScaler(init_scale=1.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=100000)


def branch(cfg: Any, model_state: dict[str, torch.Tensor], payload: dict[str, Any] | None,
           batches: list[Any], selected: list[dict[str, Any]], device: torch.device,
           seed: int, use_gpc: bool, mode: str, updates: int, eval_batches: int) -> dict[str, Any]:
    set_all_seed(seed)
    model = old.model_from_state(cfg, model_state, device)
    loss_fn, center = old.base.make_loss(cfg, num_classes=old.base.NUM_CLASSES)
    optimizer, _center_optimizer = old.base.make_optimizer(cfg, model, center)
    scheduler = old.base.create_scheduler(cfg, optimizer)
    scaler = make_scaler(mode, payload)
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
    update_rows = []
    for index in range(updates):
        batch = batches[index]
        pairs = selected[index]["selected"]
        optimizer.zero_grad(set_to_none=True)
        score, feat, _formal, base_total, components = old.base_forward(model, loss_fn, batch, device, index + 1)
        relation, _z = old.relation_value(feat, pairs)
        total = base_total + (0.1 * relation if use_gpc else 0.0 * relation)
        scale_before = float(scaler.get_scale())
        if not bool(torch.isfinite(total.detach()).item()):
            raise RuntimeError(f"非有限损失：mode={mode} use_gpc={use_gpc} step={index + 1}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = old.gradient_norm(model)
        scaler.step(optimizer)
        found_inf = float(old.base.found_inf(scaler, optimizer))
        scaler.update()
        update_rows.append({
            "step": index + 1,
            "base_loss": float(base_total.detach().float().cpu()),
            "gpc_loss": float(relation.detach().float().cpu()),
            "total_loss": float(total.detach().float().cpu()),
            "selected_pair_count": len(pairs),
            "gradient_norm": grad_norm,
            "scale_before": scale_before,
            "found_inf": found_inf,
            "skipped": bool(found_inf > 0),
            "scale_after": float(scaler.get_scale()),
            "rng_after": rng_digest(),
            "loss_components": {key: float(value.detach().float().cpu()) for key, value in components.items() if torch.is_tensor(value)},
        })
        del score, feat, total, base_total, relation
    valid = not any(row["skipped"] for row in update_rows)
    evaluations = []
    if valid:
        model.train()
        set_all_seed(seed + 10000)
        with torch.no_grad():
            for index in range(updates, updates + eval_batches):
                batch = batches[index]
                pairs = selected[index]["selected"]
                score, feat, _formal, base_total, _components = old.base_forward(model, loss_fn, batch, device, 1000 + index)
                relation, z = old.relation_value(feat, pairs)
                identity_accuracy = float(
                    (score["cls_score"].argmax(dim=-1) == batch["pid"].to(device)).float().mean().cpu()
                )
                evaluations.append({
                    "batch_index": index,
                    "base_loss": float(base_total.detach().float().cpu()),
                    "positive_cosine_mean": 1.0 - float(relation.detach().float().cpu()) if pairs else None,
                    "identity_accuracy": identity_accuracy,
                    "selected_pair_count": len(pairs),
                })
                del score, feat, base_total, relation, z
    return {
        "use_gpc": use_gpc,
        "mode": mode,
        "updates": update_rows,
        "valid_pair": valid,
        "valid_updates": sum(not row["skipped"] for row in update_rows),
        "skipped_steps": sum(row["skipped"] for row in update_rows),
        "evaluations": evaluations,
    }


def paired_delta(u0: dict[str, Any], u1: dict[str, Any]) -> dict[str, Any]:
    if not (u0["valid_pair"] and u1["valid_pair"]):
        return {"status": "INVALID_PAIR", "reason": "至少一侧存在 AMP skip 或非有效更新"}
    loss_deltas = [b["base_loss"] - a["base_loss"] for a, b in zip(u0["evaluations"], u1["evaluations"])]
    pos_deltas = [
        b["positive_cosine_mean"] - a["positive_cosine_mean"]
        for a, b in zip(u0["evaluations"], u1["evaluations"])
        if a["positive_cosine_mean"] is not None and b["positive_cosine_mean"] is not None
    ]
    acc_deltas = [b["identity_accuracy"] - a["identity_accuracy"] for a, b in zip(u0["evaluations"], u1["evaluations"])]
    return {
        "status": "VALID_PAIR",
        "delta_utility_mean": sum(loss_deltas) / len(loss_deltas) if loss_deltas else None,
        "delta_utility_values": loss_deltas,
        "positive_cosine_delta_mean": sum(pos_deltas) / len(pos_deltas) if pos_deltas else None,
        "positive_cosine_delta_values": pos_deltas,
        "identity_accuracy_delta_mean": sum(acc_deltas) / len(acc_deltas) if acc_deltas else None,
        "identity_accuracy_delta_values": acc_deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--mode", choices=("original_amp", "fixed_amp"), default="original_amp")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 不可用")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not devices or any(name != "PPU-ZW810E" for name in devices):
        raise RuntimeError(f"设备不是 PPU-ZW810E：{devices}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    rows = []
    old.UPDATES = args.updates
    old.EVAL_BATCHES = args.eval_batches
    for seed in (1234, 4321):
        cfg = old.load_cfg()
        batches, selected, loader_meta = old.prepare_batches(cfg, seed)
        for case in ("initial", "E30"):
            state, payload = old.state_for_case(case, seed)
            u0 = branch(cfg, state, payload, batches, selected, device, seed, False, args.mode, args.updates, args.eval_batches)
            u1 = branch(cfg, state, payload, batches, selected, device, seed, True, args.mode, args.updates, args.eval_batches)
            rows.append({
                "seed": seed,
                "state": case,
                "loader_meta": loader_meta,
                "selected_counts": [int(item["matched_count"]) for item in selected[:args.updates + args.eval_batches]],
                "u0": u0,
                "u1": u1,
                "paired_delta": paired_delta(u0, u1),
            })
            del state, payload
            torch.cuda.empty_cache()
    valid = [row for row in rows if row["paired_delta"]["status"] == "VALID_PAIR"]
    delta = [row["paired_delta"]["delta_utility_mean"] for row in valid if row["paired_delta"]["delta_utility_mean"] is not None]
    pos = [row["paired_delta"]["positive_cosine_delta_mean"] for row in valid if row["paired_delta"]["positive_cosine_delta_mean"] is not None]
    payload = {
        "run_time": args.run_time,
        "mode": args.mode,
        "device_names": devices,
        "cpu_model_fallback": False,
        "train_only": True,
        "official_query_gallery_used": False,
        "development_validation_used_for_optimization": False,
        "status": "PASS_VALID_PAIRS" if len(valid) == len(rows) else "INVALID_PAIRS_PRESENT",
        "summary": {
            "total_states": len(rows),
            "valid_pairs": len(valid),
            "invalid_pairs": len(rows) - len(valid),
            "delta_utility_mean": sum(delta) / len(delta) if delta else None,
            "delta_utility_min": min(delta) if delta else None,
            "delta_utility_max": max(delta) if delta else None,
            "positive_cosine_delta_mean": sum(pos) / len(pos) if pos else None,
            "stable_delta_utility_direction": bool(delta) and all(value < 0 for value in delta),
            "stable_positive_cosine_direction": bool(pos) and all(value > 0 for value in pos),
        },
        "rows": rows,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "scripts/gpc_paired_update_control.py",
                ROOT / "scripts/gpc_update_utility_diagnostic.py",
                ENGINE / "scripts/phase7i_relation_specificity.py",
            )
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"paired_update_{args.mode}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "summary": payload["summary"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
