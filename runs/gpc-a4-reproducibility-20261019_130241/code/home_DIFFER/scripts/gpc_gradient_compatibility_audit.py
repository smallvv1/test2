#!/usr/local/bin/python3.10
"""GPC-v2 训练期梯度兼容性诊断。

本脚本是 train-only、diagnostic-only：不创建 optimizer，不调用 backward/step，
不改变任何 checkpoint。模型 forward 和 autograd.grad 在 PPU 上执行；CPU 只负责
读取清单、保存汇总和少量标量统计。

梯度比较在共享身份输出 feat[0] 层进行：g_base 是 CE+Triplet+Bio，g_gpc 是
原始 GPC-v1 的 1-cosine 关系损失。每个 component 使用独立的 autograd.grad 调用，
最后一个 GPC grad 释放当前 forward graph，避免复用已释放图。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp


ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
EXPECTED_DEVICE = "PPU-ZW810E"
TAU_PATH = ROOT / "outputs/gpc_v2_ltcc_20261015_155822/saturation_threshold.json"
SPLIT_PATH = ROOT / "gpc_v2/ltcc_dev_split_20261015_145338.json"
SPLIT_CSV = ROOT / "outputs/gpc_v2_ltcc_20261015_155822/development_split.csv"
RUN_ROOT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
TRAINING_ROOT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822/training"
PROBE_BATCHES_PER_PHASE = 4

sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))
from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_img_transforms  # noqa: E402
from model.build import factory  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402
import gpc_v2_train as v2  # noqa: E402

base = b6.base


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    base.SEED = seed
    v2.p7.SEED = seed


def load_cfg(config_path: Path):
    cfg = GLOBAL_CFG.clone()
    cfg.defrost()
    cfg.DATA.BIO_INDEX = 0
    cfg.MODEL.DEVICE_ID = 0
    cfg.merge_from_file(str(config_path))
    # yacs 将历史 YAML 的字符串 "0" 解码成标量整数，但 LTCC loader
    # 需要可迭代的语义因子索引；这里只做评估 harness 类型适配。
    cfg.defrost()
    cfg.DATA.BIO_INDEX = [0]
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()
    return cfg


def require_ppu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("PPU-only 梯度诊断拒绝执行：torch.cuda.is_available()=False；禁止 CPU model fallback")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not names or any(name != EXPECTED_DEVICE for name in names):
        raise RuntimeError(f"PPU 设备不符合要求：{names}")
    torch.cuda.set_device(0)
    print(f"中文说明：梯度诊断使用 PPU device=0；设备={names}；CPU model fallback=False", flush=True)
    return torch.device("cuda:0")


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state 类型错误：{type(state)}")
    return {str(key).replace("module.", "", 1): value for key, value in state.items()}


def model_from_checkpoint(config_path: Path, checkpoint: Path, device: torch.device):
    cfg = load_cfg(config_path)
    state = checkpoint_state(checkpoint)
    head = state.get("head.weight")
    if head is None:
        raise KeyError(f"checkpoint 缺少 head.weight：{checkpoint}")
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=int(head.shape[0]), camera_num=12)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load 失败：missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)
    model.train()
    return model, cfg


def load_split() -> tuple[dict[str, Any], list[dict[str, str]]]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    # 冻结 JSON 的 train_pids 是原始 PID；正式 gpc_v2_train 使用
    # development_split_manifest 中与 LTCC loader 一致的连续 label。两者
    # 通过同一冻结文件关联，但不能直接混用。
    manifest_path = RUN_ROOT / "development_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("frozen_split_sha256") != sha256_file(SPLIT_PATH):
        raise RuntimeError("development_split_manifest 与冻结 split SHA256 不一致")
    split["train_pids"] = manifest["train_pids"]
    split["validation_pids"] = manifest.get("validation_pids", [])
    split["train_raw_pids"] = manifest.get("train_raw_pids", [])
    with SPLIT_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return split, rows


def active_state(dataset: Any, train_pids: set[int]) -> dict[int, list[int]]:
    values: dict[int, set[int]] = defaultdict(set)
    for item in dataset.train:
        if int(item["pid"]) in train_pids:
            values[int(item["pid"])].add(int(item["clothes_id"]))
    if set(values) != train_pids:
        raise RuntimeError("梯度诊断 active_state 未覆盖冻结开发训练 PID")
    return {pid: sorted(clothes) for pid, clothes in sorted(values.items())}


def zero_like_or_none(value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference) if value is None else value


def scalar_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().float()).item())


def cosine_grad(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na = torch.linalg.vector_norm(a.detach().float())
    nb = torch.linalg.vector_norm(b.detach().float())
    if float(na.item()) <= 1e-12 or float(nb.item()) <= 1e-12:
        return None
    return float(torch.sum(a.detach().float() * b.detach().float()).div(na * nb).item())


def selected_relation(entries: list[dict[str, Any]], path_to_index: dict[str, int], bank: dict[str, Any], arm: str, epoch: int, iteration: int):
    v2.p7.PHASE = f"gpc_v2_ltcc_20261015_155822_two_seed_{arm}_gradient_audit"
    chosen, matched, info = v2.selected_v2(entries, path_to_index, bank, arm, epoch, 0, iteration)
    return chosen["id"], matched, info


def loss_and_gradients(model, loss_fn, batch: dict[str, Any], device: torch.device, selected, arm: str, tau: float, step: int) -> dict[str, Any]:
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    cams = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    model.zero_grad(set_to_none=True)
    writer = base.MetricCapture()
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=cams)
    base_total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=step)
    components = dict(loss_fn.last_components)
    feature = feat[0]
    representation = F.normalize(feature.float(), dim=-1)
    values = []
    cosine_values = []
    for i, j, _image_key, _state_key in selected:
        cosine = torch.sum(representation[i] * representation[j])
        cosine_values.append(cosine)
        values.append(F.relu(tau - cosine) if arm == "A5" else (1.0 - cosine))
    relation = torch.stack(values).mean() if values else representation.sum() * 0.0

    # 独立的 autograd.grad 序列：前几个 retain_graph，GPC 作为最后一次释放图。
    def grad_of(loss: torch.Tensor, retain_graph: bool) -> torch.Tensor:
        grad = torch.autograd.grad(loss, feature, retain_graph=retain_graph, allow_unused=True)[0]
        return zero_like_or_none(grad, feature)

    g_ce = grad_of(components["ce"], True)
    g_triplet = grad_of(components["triplet"], True)
    g_bio = grad_of(components["bio"], True)
    base_loss = components["ce"] + components["triplet"] + components["bio"]
    g_base = grad_of(base_loss, True)
    g_gpc = grad_of(relation, False)
    g_applied = g_gpc * 0.1

    with torch.no_grad():
        cos_values = torch.stack(cosine_values) if cosine_values else torch.empty(0, device=device)
        pid_values = [int(pids[i].detach().cpu()) for i, _j, _key, _state in selected]
        # batch-local 最近异身份相似度：只作为训练局部诊断，不是官方检索指标。
        sim = representation @ representation.T
        neg_mask = pids[:, None] != pids[None, :]
        neg_sim = sim.masked_fill(~neg_mask, -2.0).max(dim=1).values
        finite = [bool(torch.isfinite(x).all().item()) for x in (base_total, relation, g_base, g_gpc)]
    result = {
        "selected_pair_count": len(selected),
        "matched_pair_budget": int(len(selected)),
        "selected_pid_count": len(set(pid_values)),
        "selected_pid_histogram": dict(Counter(pid_values)),
        "relation_loss": float(relation.detach().float().item()),
        "relation_lambda": 0.1,
        "relation_applied_loss": float((0.1 * relation).detach().float().item()),
        "positive_cosine_mean": float(cos_values.mean().item()) if cos_values.numel() else None,
        "positive_cosine_min": float(cos_values.min().item()) if cos_values.numel() else None,
        "positive_cosine_max": float(cos_values.max().item()) if cos_values.numel() else None,
        "a5_relu_active_fraction": float((cos_values < tau).float().mean().item()) if cos_values.numel() else None,
        "nearest_negative_cosine_mean": float(neg_sim.mean().item()),
        "local_identity_margin": float((cos_values.mean() - neg_sim.mean()).item()) if cos_values.numel() else None,
        "ce_loss": float(components["ce"].detach().float().item()),
        "triplet_loss": float(components["triplet"].detach().float().item()),
        "bio_loss": float(components["bio"].detach().float().item()),
        "base_loss": float(base_loss.detach().float().item()),
        "ce_grad_norm": scalar_norm(g_ce),
        "triplet_grad_norm": scalar_norm(g_triplet),
        "bio_grad_norm": scalar_norm(g_bio),
        "base_grad_norm": scalar_norm(g_base),
        "gpc_grad_norm_raw": scalar_norm(g_gpc),
        "gpc_grad_norm_applied_lambda": scalar_norm(g_applied),
        "gpc_over_base_grad_ratio": float(scalar_norm(g_applied) / max(scalar_norm(g_base), 1e-12)),
        "cos_base_gpc": cosine_grad(g_base, g_gpc),
        "cos_triplet_gpc": cosine_grad(g_triplet, g_gpc),
        "finite": all(finite),
        "cpu_model_fallback": False,
        "gradient_space": "共享 feat[0] 输出层；GPC 梯度包含对归一化 z=normalize(feat[0]) 的链式梯度",
    }
    del score, feat, base_total, relation, g_ce, g_triplet, g_bio, g_base, g_gpc, g_applied
    model.zero_grad(set_to_none=True)
    return result


def phase_checkpoints(seed: int) -> list[tuple[str, Path, Path, int]]:
    base_dir = TRAINING_ROOT / "two_seed"
    values = [
        ("early", base_dir / f"B0_seed{seed}" / "model_best.pth", base_dir / f"B0_seed{seed}" / "config.yml", 1),
        ("late", base_dir / f"B0_seed{seed}" / "model_final.pth", base_dir / f"B0_seed{seed}" / "config.yml", 60),
    ]
    if seed == 1234:
        values.insert(1, ("middle", TRAINING_ROOT / "screen/B0_seed1234/rank0_state_epoch030.pt", TRAINING_ROOT / "screen/B0_seed1234/config.yml", 30))
    return values


def run_one_model(arm: str, seed: int, phase: str, checkpoint: Path, config_path: Path, epoch: int, device: torch.device, split: dict[str, Any], output_rows: list[dict[str, Any]]) -> None:
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"缺少梯度诊断 checkpoint/config：{checkpoint} / {config_path}")
    set_seed(seed)
    model, cfg = model_from_checkpoint(config_path, checkpoint, device)
    dataset = base.build_dataset(cfg)
    transform_train, _ = build_img_transforms(cfg)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    train_pids = set(map(int, split["train_pids"]))
    state = active_state(dataset, train_pids)
    index_map = base.build_index_map(dataset)
    base.SEED = seed
    loader, meta = base.make_epoch_loader(dataset, transform_train, state, index_map, epoch, 0)
    loss_fn, _center = __import__("loss.make_loss", fromlist=["make_loss"]).make_loss(cfg, num_classes=len(train_pids))
    tau = float(json.loads(TAU_PATH.read_text(encoding="utf-8"))["tau"])
    inspected = 0
    step = 0
    for iteration, batch in enumerate(loader, start=1):
        entries = [{"pid": int(batch["pid"][i]), "clothes_id": int(batch["clothes_id"][i]), "image_path": str(batch["image_path"][i])} for i in range(len(batch["pid"]))]
        selected, matched, info = selected_relation(entries, path_to_index, bank, arm, epoch, iteration)
        if not selected:
            continue
        step += 1
        row = loss_and_gradients(model, loss_fn, batch, device, selected, arm, tau, step)
        row.update({
            "created_utc": utc_now(), "arm": arm, "seed": seed, "phase": phase, "epoch": epoch,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "selector": info["selector"], "candidate_counts": info["candidate_counts"],
            "relation_active": bool(info["active"]), "loader_batches": meta["batches"],
        })
        output_rows.append(row)
        inspected += 1
        print(f"中文结果：梯度 probe arm={arm} seed={seed} phase={phase} epoch={epoch} batch={iteration} base_norm={row['base_grad_norm']:.6g} gpc_norm={row['gpc_grad_norm_raw']:.6g} cos={row['cos_base_gpc']}", flush=True)
        if inspected >= PROBE_BATCHES_PER_PHASE:
            break
    if inspected < PROBE_BATCHES_PER_PHASE:
        raise RuntimeError(f"梯度诊断有效 batch 不足：{arm} seed={seed} phase={phase} got={inspected}")
    del loader, loss_fn, dataset, model, bank
    gc = __import__("gc")
    gc.collect()
    torch.cuda.empty_cache()


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["arm"], int(row["seed"]), row["phase"])].append(row)
    result = []
    for (arm, seed, phase), values in sorted(groups.items()):
        def mean(key: str) -> float | None:
            numbers = [float(row[key]) for row in values if row[key] is not None]
            return float(np.mean(numbers)) if numbers else None
        def frac(key: str) -> float | None:
            numbers = [float(row[key]) for row in values if row[key] is not None]
            return float(np.mean(numbers)) if numbers else None
        result.append({
            "arm": arm, "seed": seed, "phase": phase, "probe_count": len(values),
            "positive_cosine_mean": mean("positive_cosine_mean"),
            "nearest_negative_cosine_mean": mean("nearest_negative_cosine_mean"),
            "local_identity_margin": mean("local_identity_margin"),
            "gpc_loss": mean("relation_loss"),
            "gpc_grad_norm_raw": mean("gpc_grad_norm_raw"),
            "gpc_grad_norm_applied_lambda": mean("gpc_grad_norm_applied_lambda"),
            "base_grad_norm": mean("base_grad_norm"),
            "gpc_over_base_grad_ratio": mean("gpc_over_base_grad_ratio"),
            "cos_base_gpc": mean("cos_base_gpc"),
            "cos_triplet_gpc": mean("cos_triplet_gpc"),
            "ce_grad_norm": mean("ce_grad_norm"),
            "triplet_grad_norm": mean("triplet_grad_norm"),
            "bio_grad_norm": mean("bio_grad_norm"),
            "a5_relu_active_fraction": frac("a5_relu_active_fraction"),
            "all_finite": all(bool(row["finite"]) for row in values),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    device = require_ppu()
    split, _rows = load_split()
    rows: list[dict[str, Any]] = []
    for seed in (1234, 4321):
        checkpoints = phase_checkpoints(seed)
        for phase, checkpoint, config_path, epoch in checkpoints:
            # 每个冻结阶段都分析 B0；A5 作为饱和梯度的附加对照。
            for arm in ("B0", "A5"):
                actual_checkpoint = checkpoint
                if arm == "A5":
                    actual_checkpoint = checkpoint.parent.parent / f"A5_seed{seed}" / checkpoint.name
                    if phase == "middle":
                        actual_checkpoint = TRAINING_ROOT / "screen/A5_seed1234/rank0_state_epoch030.pt"
                actual_config = actual_checkpoint.parent / "config.yml"
                run_one_model(arm, seed, phase, actual_checkpoint, actual_config, epoch, device, split, rows)
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "gradient_probe_rows.csv", rows)
    write_csv(args.output / "gradient_summary.csv", summary)
    payload = {
        "created_utc": utc_now(), "train_only": True, "official_query_gallery_used": False,
        "cpu_model_fallback": False, "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "gradient_space": "共享 feat[0] 输出层；不是完整参数梯度；GPC 为归一化身份特征 z 的链式梯度",
        "forward_graph_policy": "一次 forward；CE/Triplet/Bio/GPC 使用 autograd.grad 分量，前项 retain_graph，最后 GPC 释放图",
        "probe_batches_per_phase": PROBE_BATCHES_PER_PHASE,
        "tau": float(json.loads(TAU_PATH.read_text(encoding="utf-8"))["tau"]),
        "rows": rows, "summary": summary,
    }
    write_json(args.output / "gradient_compatibility.json", payload)
    lines = [
        "# GPC 梯度兼容性诊断",
        "",
        "本诊断为 train-only、小规模、无参数更新的可行性检查。模型 forward 与 autograd.grad 使用 PPU；没有调用 optimizer、backward 或 step。",
        "",
        f"- 诊断输出：`{args.output}`。",
        f"- GPC-v1 lambda：`0.1`；A5 tau：`{payload['tau']:.10f}`。",
        "- 梯度空间：共享 `feat[0]` 输出层，不是完整模型参数梯度；因此结论只代表身份特征层的方向兼容性。",
        "- 计算图：一次 forward 后，CE、Triplet、Bio 和 GPC 分别使用正确的 `autograd.grad`；最后一次 GPC grad 释放 graph。",
        "",
        "## 汇总",
        "",
        "| arm | seed | 阶段 | probes | 正样本cos | 最近异身份cos | 局部margin | GPC grad | base grad | λGPC/base | cos(base,GPC) | cos(triplet,GPC) | A5激活比例 | finite |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        def fmt(key: str) -> str:
            value = row.get(key)
            return "NA" if value is None else f"{float(value):.6f}"
        lines.append(f"| {row['arm']} | {row['seed']} | {row['phase']} | {row['probe_count']} | {fmt('positive_cosine_mean')} | {fmt('nearest_negative_cosine_mean')} | {fmt('local_identity_margin')} | {fmt('gpc_grad_norm_raw')} | {fmt('base_grad_norm')} | {fmt('gpc_over_base_grad_ratio')} | {fmt('cos_base_gpc')} | {fmt('cos_triplet_gpc')} | {fmt('a5_relu_active_fraction')} | {row['all_finite']} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "- `cos(base,GPC)<0` 是特征层梯度方向冲突的观测证据；它不能单独证明冲突导致检索下降。",
        "- `cos(base,GPC)` 接近 0 表示近似正交；正值表示方向一致。需要结合两个 seed、训练阶段、正样本相似度、最近异身份相似度和局部 margin 解读。",
        "- A5 的激活比例是 `cos(z_i,z_j)<tau` 的比例；低比例说明 hinge 大部分已饱和，但不等同于原始 GPC 没有训练影响。",
        "- 逐 probe 明细：`gradient_probe_rows.csv`；机器可读汇总：`gradient_compatibility.json`。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "report": str(args.report), "summary_rows": len(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
