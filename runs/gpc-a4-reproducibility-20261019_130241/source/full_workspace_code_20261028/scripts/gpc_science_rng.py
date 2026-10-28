#!/usr/bin/env python3
"""GPC 科学随机命名空间（独立于历史 Phase7I）。

本模块只描述可复现实验协议，不读取 RUN_TIME、输出目录、日志路径或
显示用 arm 名称。历史 Phase7I 的 PHASE/SEED 逻辑保持不变；新诊断脚本
显式使用本模块生成 batch augmentation 和候选关系选择键。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence


SCIENCE_RNG_NAMESPACE = "DIFFER_GPC_SCIENCE_RNG_V1"
SCIENCE_PROTOCOL_ID = "LTCC_DEV_SPLIT_20261015_145338"
FROZEN_SPLIT_SHA256 = "bf869c3226aea7b87fa367c69b83f868538e37cdf8166a8b43a09cf5582b84e9"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def science_payload(
    *,
    seed: int,
    epoch: int,
    rank: int,
    iteration: int,
    kind: str,
    parts: Sequence[Any] = (),
) -> dict[str, Any]:
    """Build the complete key payload.

    ``kind`` is a fixed scientific operation (for example ``id`` or
    ``augmentation``), not an experiment display name.  The caller must not
    pass RUN_TIME, log paths, output paths, or arm labels in ``parts``.
    """
    return {
        "namespace": SCIENCE_RNG_NAMESPACE,
        "protocol_id": SCIENCE_PROTOCOL_ID,
        "frozen_split_sha256": FROZEN_SPLIT_SHA256,
        "seed": int(seed),
        "epoch": int(epoch),
        "rank": int(rank),
        "iteration": int(iteration),
        "kind": str(kind),
        "parts": list(parts),
    }


def science_hash(
    *,
    seed: int,
    epoch: int,
    rank: int,
    iteration: int,
    kind: str,
    parts: Sequence[Any] = (),
) -> str:
    payload = science_payload(
        seed=seed, epoch=epoch, rank=rank, iteration=iteration, kind=kind, parts=parts
    )
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def science_seed(
    *,
    seed: int,
    epoch: int,
    rank: int,
    iteration: int,
    kind: str,
    parts: Sequence[Any] = (),
) -> int:
    return int(science_hash(
        seed=seed, epoch=epoch, rank=rank, iteration=iteration, kind=kind, parts=parts
    )[:16], 16) & 0x7FFFFFFF


def science_selection(
    candidates: Iterable[Any],
    count: int,
    *,
    seed: int,
    epoch: int,
    rank: int,
    iteration: int,
    kind: str,
) -> list[Any]:
    """Deterministically select a fixed budget from a candidate list."""
    values = list(candidates)
    if count <= 0:
        return []
    if count > len(values):
        raise ValueError(f"science selection budget {count} exceeds candidates {len(values)}")

    def key(item: Any) -> str:
        # Relation tuples use image_key and pair indices.  The JSON fallback
        # also makes this helper usable by the protocol unit tests.
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            parts = [item[2], item[0], item[1]]
        else:
            parts = [item]
        return science_hash(
            seed=seed, epoch=epoch, rank=rank, iteration=iteration, kind=kind, parts=parts
        )

    return sorted(values, key=key)[:count]


def list_sha256(values: Iterable[Any]) -> str:
    return hashlib.sha256(_canonical(list(values)).encode("utf-8")).hexdigest()


def relation_key(item: Any) -> Any:
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        return [item[0], item[1], item[2], item[3]]
    return item


def relation_sha256(values: Iterable[Any]) -> str:
    return list_sha256([relation_key(value) for value in values])


def protocol_digest() -> str:
    return science_hash(
        seed=0, epoch=0, rank=0, iteration=0, kind="protocol", parts=[]
    )
