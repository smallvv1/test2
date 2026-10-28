#!/usr/local/bin/python3.10
"""NDP-VLADR 原型的真实 EVA02/PPU 多邻居无更新探针。

只使用 LTCC train-only 的一个预先固定 sampler batch；不执行 optimizer.step，
不改变 GPC 代码。目的仅是检查多正/多负局部目标在真实模型图上能否反向，
以及它是否与 Triplet/SupCon 控制产生可测量的梯度差异。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))

from ndp_live_ppu_gradient_probe import (  # noqa: E402
    CHECKPOINT,
    EXPECTED_DEVICE,
    BANK,
    finite,
    image_batch,
    load_cfg,
    load_dataset_and_plan,
    load_model,
    sha256_file,
    tensor_cos,
    tensor_norm,
)
from ndp_vladr_relation import (  # noqa: E402
    adaptive_margin_relation_loss,
    local_advantage_preservation,
    local_identity_advantage,
    multi_positive_supcon_control,
)


def scalar(x) -> float:
    return float(x.detach().float().cpu().item())


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def selected_params(model):
    wanted = []
    for name, param in model.named_parameters():
        low = name.lower()
        if any(token in low for token in ("patch_embed.proj.weight", "blocks.0.attn.qkv.weight", "blocks.22.attn.qkv.weight", "head.weight")):
            wanted.append((name, param))
    return wanted or list(model.named_parameters())[:4]


def grad_report(model, loss, feat, base_feat_grad, params):
    import torch
    grads = torch.autograd.grad(loss, (feat,) + tuple(p for _, p in params), allow_unused=True)
    fg = grads[0] if grads[0] is not None else torch.zeros_like(feat)
    p_rows = {}
    for (name, _), g in zip(params, grads[1:]):
        p_rows[name] = {
            "present": g is not None,
            "finite": bool(finite(g)) if g is not None else None,
            "norm": tensor_norm(g) if g is not None else 0.0,
        }
    return {
        "loss": scalar(loss),
        "loss_finite": finite(loss),
        "feature_grad_norm": tensor_norm(fg),
        "feature_grad_finite": finite(fg),
        "parameter_grad_norm_sum": float(sum(v["norm"] for v in p_rows.values())),
        "parameter_gradients": p_rows,
        "cosine_with_base_feature_gradient": tensor_cos(fg, base_feat_grad),
    }


def find_fixed_batch(dataset, plan):
    """选rank0第20个迭代附近的多样本 batch，规则固定且不看特征效果。"""
    pids = np.asarray([int(x["pid"]) for x in dataset.train])
    clothes = np.asarray([int(x["clothes_id"]) for x in dataset.train])
    for rank in (0, 1):
        for iteration, batch in enumerate(plan["ranks"][rank]["batches"], start=1):
            groups = {}
            for idx in batch:
                groups.setdefault(int(pids[idx]), []).append(int(idx))
            eligible = [(pid, values) for pid, values in sorted(groups.items()) if len(values) >= 3]
            if not eligible or len(groups) < 2:
                continue
            pid, values = eligible[0]
            anchor = values[0]
            positives = [x for x in values[1:] if clothes[x] != clothes[anchor]][:2]
            if len(positives) < 2:
                positives = values[1:3]
            negatives = [x for x in batch if int(pids[x]) != pid][:4]
            if len(positives) >= 2 and len(negatives) >= 2:
                return {
                    "rank": rank, "iteration": iteration, "batch_indices": [int(x) for x in batch],
                    "anchor": int(anchor), "positive": [int(x) for x in positives],
                    "negative": [int(x) for x in negatives],
                    "pid": int(pid),
                    "positive_cross_clothes": [bool(clothes[x] != clothes[anchor]) for x in positives],
                }
    raise RuntimeError("冻结采样计划中没有至少2正/2负的合法多邻居batch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-time", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("PPU/CUDA设备不可用，禁止CPU模型回退")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if any(name != EXPECTED_DEVICE for name in names):
        raise RuntimeError(f"PPU设备不符合要求: {names}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(1234)
    cfg = load_cfg()
    _base, dataset, _path_to_index, plan = load_dataset_and_plan(cfg)
    fixed = find_fixed_batch(dataset, plan)
    bank = np.load(BANK, allow_pickle=False)["features"].astype(np.float32)
    transform_train, transform_test = __import__("data", fromlist=["build_img_transforms"]).build_img_transforms(cfg)
    params_info = None
    model, load_info = load_model(cfg, device)
    model.train()
    params = selected_params(model)
    ids = fixed["batch_indices"]
    images, image_hashes = image_batch(dataset, ids, transform_test, device)
    cams = torch.tensor([int(dataset.train[i]["camid"]) for i in ids], dtype=torch.long, device=device)
    labels = torch.tensor([int(dataset.train[i]["pid"]) for i in ids], dtype=torch.long, device=device)
    clothes = torch.tensor([int(dataset.train[i]["clothes_id"]) for i in ids], dtype=torch.long, device=device)
    pos_local = [ids.index(x) for x in fixed["positive"]]
    neg_local = [ids.index(x) for x in fixed["negative"]]
    anchor_local = ids.index(fixed["anchor"])
    ref_all = F.normalize(torch.from_numpy(bank[ids]).to(device), dim=-1).detach()
    rows = []

    # 每个候选独立 forward，禁止复用已释放 graph。
    for candidate in ("C0_batch_hard_triplet", "C1_adaptive_margin", "C2_ndp_cache", "C3_ndp_live", "C4_supcon_npair"):
        model.zero_grad(set_to_none=True)
        score, feat = model(images, cam_label=cams)
        if feat[0].shape[-1] != 1024:
            raise RuntimeError(f"身份特征维度异常: {tuple(feat[0].shape)}")
        z = F.normalize(feat[0].float(), dim=-1)
        if candidate == "C0_batch_hard_triplet":
            from loss.triplet_loss import TripletLoss
            loss = TripletLoss()(feat[0], labels, normalize_feature=False)[0]
        elif candidate == "C2_ndp_cache":
            cur_a = z[anchor_local:anchor_local + 1]
            cur_p = ref_all[pos_local].unsqueeze(0)
            cur_n = ref_all[neg_local].unsqueeze(0)
            ref_a = ref_all[anchor_local:anchor_local + 1]
            ref_p = ref_all[pos_local].unsqueeze(0)
            ref_n = ref_all[neg_local].unsqueeze(0)
            loss, info = local_advantage_preservation(cur_a, cur_p, cur_n, ref_a, ref_p, ref_n, temperature=0.2, epsilon=0.02, reference_reliability=0.05, reliable_mask=torch.ones(1, dtype=torch.bool, device=device))
        elif candidate == "C3_ndp_live":
            cur_a = z[anchor_local:anchor_local + 1]
            cur_p = z[pos_local].unsqueeze(0)
            cur_n = z[neg_local].unsqueeze(0)
            ref_a = ref_all[anchor_local:anchor_local + 1]
            ref_p = ref_all[pos_local].unsqueeze(0)
            ref_n = ref_all[neg_local].unsqueeze(0)
            loss, info = local_advantage_preservation(cur_a, cur_p, cur_n, ref_a, ref_p, ref_n, temperature=0.2, epsilon=0.02, reference_reliability=0.05, reliable_mask=torch.ones(1, dtype=torch.bool, device=device))
        elif candidate == "C1_adaptive_margin":
            ref_adv, _ = local_identity_advantage(ref_all[anchor_local:anchor_local + 1], ref_all[pos_local].unsqueeze(0), ref_all[neg_local].unsqueeze(0), temperature=0.2)
            cur_adv, _ = local_identity_advantage(z[anchor_local:anchor_local + 1], z[pos_local].unsqueeze(0), z[neg_local].unsqueeze(0), temperature=0.2)
            loss, info = adaptive_margin_relation_loss(cur_adv, ref_adv, epsilon=0.02, reliable_mask=torch.ones(1, dtype=torch.bool, device=device))
        else:
            loss, info = multi_positive_supcon_control(z[anchor_local:anchor_local + 1], z[pos_local].unsqueeze(0), z[neg_local].unsqueeze(0), temperature=0.2)
        # 基础梯度来自同一图上的一个独立重算，避免 retain_graph 作为默认修复。
        model.zero_grad(set_to_none=True)
        score_b, feat_b = model(images, cam_label=cams)
        from loss.make_loss import make_loss
        loss_fn, _ = make_loss(cfg, num_classes=77)
        writer = type("Writer", (), {"add_scalar": lambda self, *a, **k: None})()
        captions = torch.stack([torch.from_numpy(np.asarray(dataset.train[i]["caption_feature"])) for i in ids], dim=0).to(device)
        base_loss = loss_fn(score_b, [feat_b[0]], labels, cams, captions, clothes_ids=clothes, train_writer=writer, step=1)
        base_grad = torch.autograd.grad(base_loss, feat_b[0], allow_unused=True)[0]
        if base_grad is None:
            base_grad = torch.zeros_like(feat_b[0])
        # 上面的候选 graph 尚未反向；单次候选 grad 由独立 graph 取得，二者不混用。
        rec = grad_report(model, loss, feat[0], base_grad.detach(), params)
        rec.update({"candidate": candidate, "active_anchor_count": 1, "positive_count": len(pos_local), "negative_count": len(neg_local), "current_feature_requires_grad": bool(feat[0].requires_grad), "reference_requires_grad": bool(ref_all.requires_grad), "optimizer_step_executed": False})
        rows.append(rec)
        del score, feat, score_b, feat_b, loss, base_loss
        torch.cuda.empty_cache()
    result = {
        "run_time": args.run_time,
        "python": sys.executable,
        "torch": torch.__version__,
        "ppu_devices": names,
        "cpu_model_fallback": False,
        "optimizer_step_executed": False,
        "train_only": True,
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "reference_bank": str(BANK), "reference_bank_sha256": sha256_file(BANK),
        "model_load": load_info,
        "fixed_batch": fixed,
        "batch_image_tensor_sha256": image_hashes,
        "batch_plan_sha256": plan["plan_sha256"],
        "relation_sha256": digest({"anchor": fixed["anchor"], "positive": fixed["positive"], "negative": fixed["negative"]}),
        "candidate_rows": rows,
        "note": "这是无optimizer.step的PPU梯度机制探针；不是训练效果或自然轨迹激活证据。",
        "real_trajectory_activation": "UNVERIFIED",
    }
    (out / "ndp_vladr_ppu_gradient_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run_time": args.run_time, "fixed_batch": fixed, "candidates": rows}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
