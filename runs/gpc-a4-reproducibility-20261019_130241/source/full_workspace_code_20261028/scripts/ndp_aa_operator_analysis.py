#!/usr/bin/env python3
"""比较 A/A 单卡算子探针保存的真实梯度张量。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def digest(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    x = a.to(dtype=torch.float64).reshape(-1)
    y = b.to(dtype=torch.float64).reshape(-1)
    d = x - y
    nx = float(torch.linalg.vector_norm(x))
    ny = float(torch.linalg.vector_norm(y))
    nd = float(torch.linalg.vector_norm(d))
    raw = float(torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))) if nx and ny else None
    bounded = None if raw is None else max(-1.0, min(1.0, raw))
    return {
        "shape": list(a.shape), "hash_a": digest(a), "hash_b": digest(b),
        "max_abs": float(d.abs().max()) if d.numel() else 0.0,
        "mean_abs": float(d.abs().mean()) if d.numel() else 0.0,
        "diff_l2": nd, "reference_l2_a": nx, "reference_l2_b": ny,
        "relative_l2_to_a": nd / max(nx, 1e-12), "relative_l2_to_b": nd / max(ny, 1e-12),
        "cosine_raw": raw, "cosine": bounded,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    labels = {
        "amp": (f"seed{args.seed}_amp0.tensors.pt", f"seed{args.seed}_amp1.tensors.pt"),
        "fp32_vs_amp0": (f"seed{args.seed}_fp32.tensors.pt", f"seed{args.seed}_amp0.tensors.pt"),
    }
    result = {"seed": args.seed, "root": str(args.root), "comparisons": {}}
    rows = []
    for mode, (fa, fb) in labels.items():
        pa, pb = args.root / fa, args.root / fb
        if not pa.exists() or not pb.exists():
            continue
        da = torch.load(pa, map_location="cpu", weights_only=False)
        db = torch.load(pb, map_location="cpu", weights_only=False)
        result["comparisons"][mode] = {}
        for group in ("scaled_gradients", "unscaled_gradients", "module_output_grads"):
            result["comparisons"][mode][group] = {}
            for name in sorted(set(da[group]) & set(db[group])):
                va, vb = da[group][name], db[group][name]
                if isinstance(va, list):
                    entries = [compare(x, y) for x, y in zip(va, vb)]
                    value = {"entries": entries}
                    for i, entry in enumerate(entries):
                        rows.append({"seed": args.seed, "comparison": mode, "group": group, "name": f"{name}[{i}]", **entry})
                else:
                    value = compare(va, vb)
                    rows.append({"seed": args.seed, "comparison": mode, "group": group, "name": name, **value})
                result["comparisons"][mode][group][name] = value
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "seed": args.seed, "rows": len(rows), "json": str(args.out), "csv": str(csv_path)}))


if __name__ == "__main__":
    main()
