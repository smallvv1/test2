#!/usr/local/bin/python3.10
"""对真实 continuation 节点的自然激活 NDP 关系做无更新 PPU 梯度探针。

该脚本只读取已经保存的 Bio+GPC-v1 轨迹节点，不调用 optimizer.step，
不修改 GPC、Triplet、Bio 或任何历史 checkpoint。它用阶段3同一批固定
关系，在自然激活的 anchor 上比较局部 Triplet、参考 margin、缓存邻居
NDP、实时邻居 NDP 和 SupCon/N-pair 控制的实际特征/参数梯度。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
RECOVERY = ROOT / "outputs/ndp_overnight_recovery_20261021_024248"
TAU = 0.2
EPSILON = 0.02
EXPECTED_DEVICE = "PPU-ZW810E"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE / "scripts"))
import gpc_v2_train as v2  # noqa: E402
from ndp_real_trajectory_activation import (  # noqa: E402
    load_model,
    restore_numpy_pickle,
)
from ndp_vladr_relation import (  # noqa: E402
    adaptive_margin_relation_loss,
    local_advantage_preservation,
    local_identity_advantage,
    multi_positive_supcon_control,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def scalar(value) -> float:
    return float(value.detach().float().cpu().item())


def finite(value) -> bool:
    return bool(torch.isfinite(value.detach().float()).all().item())


def norm(value) -> float:
    return float(torch.linalg.vector_norm(value.detach().float()).cpu().item())


def cosine(a, b):
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    na = torch.linalg.vector_norm(aa)
    nb = torch.linalg.vector_norm(bb)
    if float(na.cpu()) <= 1e-12 or float(nb.cpu()) <= 1e-12:
        return None
    return float(torch.dot(aa, bb).div(na * nb).cpu().item())


def load_cfg_and_dataset(split, seed, run_dir):
    class Args:
        output = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
        run_time = "20261015_155822"
        arm = "B0"
        stage = "two_seed"
        epochs = 60
        full_data = False

    protocol_args = Args()
    protocol_args.seed = int(seed)
    v2.install_protocol(protocol_args, split, run_dir, None)
    cfg = v2.p7.configure()
    dataset = v2.base.build_dataset(cfg)
    return cfg, dataset


def image_batch(dataset, indices, transform, device):
    from PIL import Image

    tensors = []
    image_hashes = []
    cams = []
    for idx in indices:
        path = Path(str(dataset.train[int(idx)]["image_path"]))
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB"))
        tensors.append(tensor)
        image_hashes.append(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest())
        cams.append(int(dataset.train[int(idx)]["camid"]))
    return (
        torch.stack(tensors, dim=0).to(device, non_blocking=False),
        torch.tensor(cams, dtype=torch.long, device=device),
        image_hashes,
    )


def forward_feature(model, images, cams):
    with torch.autocast(device_type="cuda", enabled=True):
        output = model(images, cam_label=cams)
    feature = output[1][0] if isinstance(output, tuple) else output
    if feature.ndim != 2 or feature.shape[1] != 1024:
        raise RuntimeError(f"身份特征形状异常: {tuple(feature.shape)}")
    return feature


def selected_params(model):
    wanted = []
    for name, parameter in model.named_parameters():
        low = name.lower()
        if any(token in low for token in (
            "patch_embed.proj.weight", "blocks.0.attn.qkv.weight",
            "blocks.22.attn.qkv.weight", "head.weight",
        )):
            wanted.append((name, parameter))
    return wanted or list(model.named_parameters())[:4]


def grad_record(model, loss, feature, baseline_grad, parameters):
    grads = torch.autograd.grad(
        loss, (feature,) + tuple(parameter for _, parameter in parameters),
        allow_unused=True,
    )
    feature_grad = grads[0] if grads[0] is not None else torch.zeros_like(feature)
    parameter_rows = {}
    for (name, _), grad in zip(parameters, grads[1:]):
        parameter_rows[name] = {
            "present": grad is not None,
            "finite": finite(grad) if grad is not None else None,
            "norm": norm(grad) if grad is not None else 0.0,
        }
    return {
        "loss": scalar(loss),
        "loss_finite": finite(loss),
        "feature_grad_norm": norm(feature_grad),
        "feature_grad_finite": finite(feature_grad),
        "parameter_grad_norm_sum": float(sum(row["norm"] for row in parameter_rows.values())),
        "parameter_gradients": parameter_rows,
        "cosine_with_batch_hard_grad": cosine(feature_grad, baseline_grad),
    }


def candidate_loss(candidate, current_z, reference_z, anchor_pos, positive_pos, negative_pos):
    cur_a = current_z[anchor_pos].unsqueeze(0)
    cur_p = current_z[positive_pos].unsqueeze(0)
    cur_n = current_z[negative_pos].unsqueeze(0)
    ref_a = reference_z[0].unsqueeze(0)
    ref_p = reference_z[1:1 + len(positive_pos)].unsqueeze(0)
    ref_n = reference_z[1 + len(positive_pos):].unsqueeze(0)
    ref_adv, _ = local_identity_advantage(ref_a, ref_p, ref_n, temperature=TAU)
    if candidate == "C1_reference_margin_triplet":
        loss, _ = adaptive_margin_relation_loss(
            local_identity_advantage(cur_a, cur_p, cur_n, temperature=TAU)[0],
            ref_adv,
            epsilon=EPSILON,
            reliable_mask=torch.ones(1, dtype=torch.bool, device=current_z.device),
        )
    elif candidate == "C2_ndp_detached_cache":
        loss, _ = local_advantage_preservation(
            cur_a, ref_p, ref_n, ref_a, ref_p, ref_n,
            temperature=TAU, epsilon=EPSILON, reference_reliability=0.05,
            reliable_mask=torch.ones(1, dtype=torch.bool, device=current_z.device),
        )
    elif candidate == "C3_ndp_live_neighbors":
        loss, _ = local_advantage_preservation(
            cur_a, cur_p, cur_n, ref_a, ref_p, ref_n,
            temperature=TAU, epsilon=EPSILON, reference_reliability=0.05,
            reliable_mask=torch.ones(1, dtype=torch.bool, device=current_z.device),
        )
    elif candidate == "C4_supcon_npair":
        loss, _ = multi_positive_supcon_control(cur_a, cur_p, cur_n, temperature=TAU)
    else:
        raise ValueError(candidate)
    return loss


def run(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("自然激活梯度探针需要两张PPU，禁止CPU回退")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if any(name != EXPECTED_DEVICE for name in names):
        raise RuntimeError(f"PPU设备不符合要求: {names}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    split = json.loads((ROOT / "outputs/gpc_v2_ltcc_20261015_155822/development_split_manifest.json").read_text())
    trajectory = RECOVERY / f"trajectory_seed{args.seed}"
    groups_path = trajectory / "activation_audit/fixed_diagnostic_groups.json"
    rows_path = trajectory / "activation_audit/activation_rows.csv"
    groups_doc = json.loads(groups_path.read_text())
    groups = {int(group["anchor"]): group for group in groups_doc["groups"]}
    all_active_rows = []
    with rows_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["update"]) == args.update and row["active"].lower() == "true":
                all_active_rows.append(row)
    all_active_rows.sort(key=lambda row: int(row["anchor"]))
    # NDP 只保护参考身份优势可靠的关系。阶段3的旧汇总保留了
    # delta>epsilon 的宽松诊断；这里不把 reference_advantage<=0.05
    # 的错误/不可靠参考结构当作NDP激活证据。
    active_rows = [row for row in all_active_rows if float(row["reference_advantage"]) > 0.05]
    if not active_rows:
        raise RuntimeError(f"seed{args.seed} update{args.update} 没有自然激活 anchor")

    run_dir = out / "protocol_install"
    cfg, dataset = load_cfg_and_dataset(split, args.seed, run_dir)
    _transform_train, transform_test = v2.base.build_img_transforms(cfg)
    source_path = args.resume_dir / "rank0_state_epoch060.pt"
    node_path = trajectory / f"node_update{args.update:04d}/model.pth"
    source_payload = restore_numpy_pickle(source_path)
    source_state = source_payload["model"]
    node_state = restore_numpy_pickle(node_path)
    source_model = load_model(cfg, source_state, device)
    source_model.eval()
    all_ids = sorted({int(x) for group in groups.values() for x in [group["anchor"], *group["positives"], *group["negatives"]]})
    with torch.no_grad():
        source_images, source_cams, _ = image_batch(dataset, all_ids, transform_test, device)
        source_feature = forward_feature(source_model, source_images, source_cams)
        source_feature = F.normalize(source_feature.float(), dim=-1).detach().cpu()
    del source_model, source_images, source_cams
    torch.cuda.empty_cache()

    model = load_model(cfg, node_state, device)
    model.eval()
    parameters = selected_params(model)
    candidate_names = (
        "C0_batch_hard_triplet", "C1_reference_margin_triplet",
        "C2_ndp_detached_cache", "C3_ndp_live_neighbors", "C4_supcon_npair",
    )
    result_rows = []
    for active_row in active_rows:
        group = groups[int(active_row["anchor"])]
        batch_ids = [int(x) for x in group["batch_indices"]]
        relation_ids = [int(group["anchor"]), *map(int, group["positives"]), *map(int, group["negatives"])]
        positions = [batch_ids.index(index) for index in relation_ids]
        anchor_pos = positions[0]
        positive_pos = positions[1:1 + len(group["positives"])]
        negative_pos = positions[1 + len(group["positives"]):]
        images, cams, image_hashes = image_batch(dataset, batch_ids, transform_test, device)
        labels = torch.tensor([int(dataset.train[index]["pid"]) for index in batch_ids], dtype=torch.long, device=device)
        reference_z = F.normalize(source_feature[[all_ids.index(index) for index in relation_ids]].to(device), dim=-1).detach()

        # 先独立前向得到 batch-hard 基线梯度，之后每个候选都重新前向。
        model.zero_grad(set_to_none=True)
        baseline_feature = forward_feature(model, images, cams)
        from loss.triplet_loss import TripletLoss
        baseline_loss = TripletLoss()(baseline_feature, labels, normalize_feature=False)[0]
        base_grads = torch.autograd.grad(baseline_loss, baseline_feature, allow_unused=True)
        baseline_grad = base_grads[0] if base_grads[0] is not None else torch.zeros_like(baseline_feature)
        for candidate in candidate_names:
            model.zero_grad(set_to_none=True)
            start = time.perf_counter()
            feature = forward_feature(model, images, cams)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            current_z = F.normalize(feature.float(), dim=-1)
            if candidate == "C0_batch_hard_triplet":
                loss = TripletLoss()(feature, labels, normalize_feature=False)[0]
            else:
                loss = candidate_loss(candidate, current_z, reference_z, anchor_pos, positive_pos, negative_pos)
            rec = grad_record(model, loss, feature, baseline_grad.detach(), parameters)
            rec.update({
                "seed": args.seed,
                "update": args.update,
                "anchor": int(group["anchor"]),
                "pid": int(group["pid"]),
                "candidate": candidate,
                "active_by_stage3": True,
                "reference_advantage": float(active_row["reference_advantage"]),
                "current_advantage": float(active_row["current_advantage"]),
                "advantage_drop": float(active_row["advantage_drop"]),
                "positive_count": len(group["positives"]),
                "negative_count": len(group["negatives"]),
                "batch_size": len(batch_ids),
                "relation_sha256": group["relation_sha256"],
                "batch_image_tensor_sha256": image_hashes,
                "current_feature_requires_grad": bool(feature.requires_grad),
                "reference_feature_requires_grad": bool(reference_z.requires_grad),
                "forward_ms": elapsed_ms,
                "optimizer_step_executed": False,
                "amp_found_inf": False,
            })
            result_rows.append(rec)
            del feature, current_z, loss
            torch.cuda.empty_cache()
        del images, cams, labels, reference_z
        torch.cuda.empty_cache()

    payload = {
        "seed": args.seed,
        "update": args.update,
        "trajectory_dir": str(trajectory),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "node_checkpoint": str(node_path),
        "node_checkpoint_sha256": sha256_file(node_path),
        "groups_sha256": groups_doc["groups_sha256"],
        "all_delta_active_anchor_count_at_node": len(all_active_rows),
        "reliable_active_anchor_count_at_node": len(active_rows),
        "active_anchor_sample_rule": "按固定 groups.json 的 anchor 升序全部纳入；不是按梯度结果筛选",
        "ppu_devices": names,
        "cpu_model_fallback": False,
        "optimizer_step_executed": False,
        "candidate_rows": result_rows,
        "note": "真实 continuation 节点的无 optimizer.step 参数梯度探针；不是NDP效果训练。",
    }
    (out / "natural_activation_gradient_probe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    with (out / "natural_activation_gradient_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in result_rows for key in row if not isinstance(row[key], (dict, list))})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result_rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({"seed": args.seed, "update": args.update, "active": len(active_rows), "rows": len(result_rows)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1234, 4321), required=True)
    parser.add_argument("--update", type=int, choices=(20, 50, 100, 200, 400), required=True)
    parser.add_argument("--resume-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
