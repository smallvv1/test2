#!/usr/bin/env python3
"""NDP train-only 风险审计共享工具。

本模块只处理冻结特征和标签，不导入正式训练 processor，也不读取 query/gallery。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


REQUIRED_FIELDS = {"features", "pid", "clothes", "camid", "image_path"}
TRAIN_PREFIX = "/home/data/LTCC_ReID/train/"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def load_feature_bank(path: str | Path) -> dict:
    path = Path(path)
    bank = np.load(path, allow_pickle=True)
    missing = REQUIRED_FIELDS.difference(bank.files)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    out = {key: bank[key] for key in bank.files}
    n = int(out["features"].shape[0])
    if out["features"].ndim != 2 or out["features"].shape[1] != 1024:
        raise ValueError(f"{path} 特征不是 [N,1024]: {out['features'].shape}")
    for key in ("pid", "clothes", "camid", "image_path"):
        if len(out[key]) != n:
            raise ValueError(f"{path} 字段长度不一致: {key}")
    if not np.isfinite(out["features"]).all():
        raise ValueError(f"{path} 特征含非有限值")
    paths = out["image_path"].astype(str)
    if not np.all(np.char.startswith(paths, TRAIN_PREFIX)):
        raise ValueError(f"{path} 含非 LTCC train 路径")
    out["path"] = str(path)
    out["sha256"] = sha256_file(path)
    return out


def assert_aligned(reference: dict, other: dict) -> None:
    for key in ("pid", "clothes", "camid", "image_path"):
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"样本映射不一致: {key}")
    if reference["features"].shape != other["features"].shape:
        raise ValueError("特征形状不一致")


def normalize(features: np.ndarray) -> np.ndarray:
    x = features.astype(np.float32, copy=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def fixed_anchor_indices(n: int, seed: int, count: int, namespace: int = 20261021) -> np.ndarray:
    rng = np.random.default_rng(namespace + int(seed))
    return np.sort(rng.choice(n, min(int(count), n), replace=False)).astype(np.int64)


def stable_top_indices(values: np.ndarray, count: int) -> np.ndarray:
    if values.size == 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-values, kind="stable")
    return order[: min(count, len(order))].astype(np.int64)


def top_negative_indices(
    scores: np.ndarray,
    anchor_index: int,
    pids: np.ndarray,
    k: int,
) -> np.ndarray:
    """固定 anchor 的异 PID Top-K，排除自引用和同 PID。"""
    valid = pids != pids[anchor_index]
    valid[anchor_index] = False
    candidates = np.flatnonzero(valid)
    if len(candidates) <= k:
        return candidates[np.argsort(-scores[candidates], kind="stable")]
    local = np.argpartition(-scores[candidates], k - 1)[:k]
    chosen = candidates[local]
    return chosen[np.argsort(-scores[chosen], kind="stable")]


def positive_indices(
    scores: np.ndarray,
    anchor_index: int,
    pids: np.ndarray,
    count: int,
) -> np.ndarray:
    valid = (pids == pids[anchor_index])
    valid[anchor_index] = False
    candidates = np.flatnonzero(valid)
    chosen = stable_top_indices(scores[candidates], count)
    return candidates[chosen]


def batched_scores(anchors: np.ndarray, features: np.ndarray, chunk: int = 128) -> Iterable[tuple[int, np.ndarray]]:
    for start in range(0, len(anchors), chunk):
        stop = min(start + chunk, len(anchors))
        yield start, anchors[start:stop] @ features.T


def quantile_or_nan(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")
