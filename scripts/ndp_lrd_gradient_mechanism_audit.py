#!/usr/bin/env python3
"""在冻结 train-only 特征上比较 LRD-A/B/C 与 Triplet/SupCon 的特征梯度。

这是离线机制审计，不是模型训练，也不把特征梯度当完整 EVA02 参数梯度。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ndp_lrd_abc import lrd_a_positive_mass, lrd_b_full_kl, lrd_c_cross_identity_kl, local_advantage


def top(scores, pool, k):
    return pool[np.argsort(-scores[pool], kind="stable")[: min(k, len(pool))]]


def cosine64(x: torch.Tensor, y: torch.Tensor):
    """用显式 float64 点积/范数计算诊断余弦；不以 clamp 隐藏超界。"""
    x = x.to(torch.float64).flatten()
    y = y.to(torch.float64).flatten()
    nx = torch.linalg.vector_norm(x)
    ny = torch.linalg.vector_norm(y)
    if float(nx) == 0.0 or float(ny) == 0.0:
        return None
    return torch.dot(x, y) / (nx * ny)


def lme(x, tau):
    return torch.logsumexp(x / tau, dim=1) - np.log(x.shape[1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--anchors", type=int, default=256); ap.add_argument("--tau", type=float, default=0.07)
    args = ap.parse_args()
    root = Path("/home/DIFFER/outputs/ndp_local_intrusion_20261020_155558/feature_banks") / f"seed{args.seed}"
    bio, gpc = np.load(root / "bio_feature_bank.npz", allow_pickle=True), np.load(root / "gpc_feature_bank.npz", allow_pickle=True)
    ref = bio["features"].astype(np.float32); cur = gpc["features"].astype(np.float32)
    ref /= np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12); cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(20261021 + args.seed); ids = np.sort(rng.choice(len(ref), min(args.anchors, len(ref)), replace=False))
    rs = ref[ids] @ ref.T
    pos_idx, neg_idx = [], []
    for row, idx in enumerate(ids):
        same = np.flatnonzero(bio["pid"] == bio["pid"][idx]); same = same[same != idx]
        diff = np.flatnonzero(bio["pid"] != bio["pid"][idx])
        pos_idx.append(top(rs[row], same, 4)); neg_idx.append(top(rs[row], diff, 20))
    # 等长固定关系；LTCC 训练集每个 anchor 均有合法候选。
    # 余弦、点积、范数和梯度比较全部用 float64；这不是模型训练精度设置，
    # 只是避免诊断矩阵出现 cosine>1 的报告伪影。
    a0 = torch.tensor(cur[ids], dtype=torch.float64)
    p0 = torch.tensor(np.stack([cur[x] for x in pos_idx]), dtype=torch.float64)
    n0 = torch.tensor(np.stack([cur[x] for x in neg_idx]), dtype=torch.float64)
    ar = torch.tensor(ref[ids], dtype=torch.float64); pr = torch.tensor(np.stack([ref[x] for x in pos_idx]), dtype=torch.float64); nr = torch.tensor(np.stack([ref[x] for x in neg_idx]), dtype=torch.float64)
    with torch.no_grad():
        rr0, rn0 = torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(pr, dim=-1)), torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(nr, dim=-1))
        ref_adv0 = lme(rr0, args.tau) - lme(rn0, args.tau)
        reliable = ref_adv0 > 0.05

    def run(name, fn):
        a, p, n = a0.clone().requires_grad_(True), p0.clone().requires_grad_(True), n0.clone().requires_grad_(True)
        loss, aux = fn(a, p, n)
        loss.backward()
        g = torch.cat([a.grad.flatten(), p.grad.flatten(), n.grad.flatten()])
        return {"loss": float(loss.detach()), "grad_norm_all_current_features": float(torch.linalg.vector_norm(g)), "grad_norm_anchor": float(torch.linalg.vector_norm(a.grad)), "active_anchor_count": int(aux.get("active_anchor_count", aux.get("violating_anchor_count", aux.get("valid_anchor_count", torch.tensor(len(ids)))))), "finite": bool(torch.isfinite(g).all()), "gradient": g.detach()}

    def margin_fn(a, p, n):
        with torch.no_grad():
            rr, rn = torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(pr, dim=-1)), torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(nr, dim=-1))
            r = lme(rr, args.tau) - lme(rn, args.tau)
        aa, pp, nn = F.normalize(a, dim=-1), F.normalize(p, dim=-1), F.normalize(n, dim=-1)
        cp, cn = torch.einsum("ad,amd->am", aa, pp), torch.einsum("ad,amd->am", aa, nn)
        c = lme(cp, args.tau) - lme(cn, args.tau)
        v = F.relu(r - c - 0.02); v = v[reliable]; return v.mean(), {"active_anchor_count": (v > 0).sum()}

    def supcon_fn(a, p, n):
        aa, pp, nn = F.normalize(a, dim=-1), F.normalize(p, dim=-1), F.normalize(n, dim=-1)
        ps, ns = torch.einsum("ad,amd->am", aa, pp) / args.tau, torch.einsum("ad,amd->am", aa, nn) / args.tau
        value = F.softplus(torch.logsumexp(ns, 1) - torch.log(torch.tensor(ns.shape[1], dtype=ns.dtype)) - (torch.logsumexp(ps, 1) - torch.log(torch.tensor(ps.shape[1], dtype=ps.dtype))))
        return value.mean(), {"valid_anchor_count": torch.tensor(len(ids))}

    def lrd_a_fn(a, p, n):
        aa, pp, nn = F.normalize(a, dim=-1), F.normalize(p, dim=-1), F.normalize(n, dim=-1)
        cp, cn = torch.einsum("ad,amd->am", aa, pp), torch.einsum("ad,amd->am", aa, nn)
        rr, rn = torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(pr, dim=-1)), torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(nr, dim=-1))
        loss = lrd_a_positive_mass(rr, rn, cp, cn, tau=args.tau, epsilon=0.02, reliable=reliable)
        with torch.no_grad():
            ref_mass = torch.softmax(torch.cat([rr, rn], 1) / args.tau, 1)[:, :rr.shape[1]].sum(1)
            cur_mass = torch.softmax(torch.cat([cp, cn], 1) / args.tau, 1)[:, :cp.shape[1]].sum(1)
            active = ((ref_mass - cur_mass - 0.02) > 0) & reliable
        return loss, {"active_anchor_count": active.sum()}

    def lrd_b_fn(a, p, n):
        aa, pp, nn = F.normalize(a, dim=-1), F.normalize(p, dim=-1), F.normalize(n, dim=-1)
        cp, cn = torch.einsum("ad,amd->am", aa, pp), torch.einsum("ad,amd->am", aa, nn)
        rr, rn = torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(pr, dim=-1)), torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(nr, dim=-1))
        loss = lrd_b_full_kl(rr, rn, cp, cn, tau=args.tau, reliable=reliable)
        return loss, {"active_anchor_count": reliable.sum()}

    def lrd_c_fn(a, p, n):
        aa, pp, nn = F.normalize(a, dim=-1), F.normalize(p, dim=-1), F.normalize(n, dim=-1)
        cp, cn = torch.einsum("ad,amd->am", aa, pp), torch.einsum("ad,amd->am", aa, nn)
        rr, rn = torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(pr, dim=-1)), torch.einsum("ad,amd->am", F.normalize(ar, dim=-1), F.normalize(nr, dim=-1))
        loss = lrd_c_cross_identity_kl(rr, rn, cp, cn, tau=args.tau, reliable=reliable)
        return loss, {"active_anchor_count": reliable.sum()}

    results, vectors = {}, {}
    funcs = {
        "adaptive_margin_triplet": margin_fn,
        "lrd_a_positive_mass": lrd_a_fn,
        "lrd_b_full_kl": lrd_b_fn,
        "lrd_c_cross_identity_kl": lrd_c_fn,
        "supcon_npair_control": supcon_fn,
    }
    # A/C/B 依赖正负局部关系，但为审计公平使用同一关系索引。
    for name, fn in funcs.items():
        out = run(name, fn); vectors[name] = out.pop("gradient"); results[name] = out
    cos = {}
    for x, gx in vectors.items():
        cos[x] = {y: (None if (v := cosine64(gx, gy)) is None else float(v)) for y, gy in vectors.items()}
    finite_cos = [v for row in cos.values() for v in row.values() if v is not None]
    excess_over_one = max((v - 1.0 for v in finite_cos), default=None)
    excess_below_minus_one = min((v + 1.0 for v in finite_cos), default=None)
    result = {"status": "PASS", "seed": args.seed, "sample_anchor_count": len(ids), "positive_count": 4, "negative_count": 20, "tau": args.tau, "feature_gradient_only_warning": True, "results": results, "gradient_cosine": cos}
    result["cosine_audit"] = {
        "dtype": "float64",
        "raw_dot_and_norm": True,
        "clamped": False,
        "max_excess_over_one": max(0.0, excess_over_one) if excess_over_one is not None else None,
        "max_excess_below_minus_one": min(0.0, excess_below_minus_one) if excess_below_minus_one is not None else None,
        "max_cosine": max(finite_cos) if finite_cos else None,
        "min_cosine": min(finite_cos) if finite_cos else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status":"PASS", "seed":args.seed, "results":results, "gradient_cosine":cos}, ensure_ascii=False))


if __name__ == "__main__": main()
