"""Small, train-only online graph helper for the low-interference NDP_SAFE arm."""

from __future__ import annotations

from typing import Any

import torch


def _item_value(item: Any, *names: str, default: int = -1) -> int:
    if isinstance(item, dict):
        for name in names:
            if name in item and item[name] is not None:
                return int(item[name])
    return int(default)


def build_safe_online_neighbor_graph(
    fresh_features: torch.Tensor,
    dataset_train: list[dict[str, Any]],
    device: torch.device,
    max_negative_pids: int = 3,
    block_size: int = 512,
    previous_signature: list[tuple[int, ...]] | None = None,
):
    """Build the existing GPU danger graph with a cross-clothes positive preference.

    Negative PID retrieval remains the validated blockwise implementation.  The
    only change is the positive endpoint: choose one legal same-PID image from a
    different clothes id when available, otherwise the first legal same-PID
    image.  All endpoints stay detached and the graph is train-only.
    """
    import ndp_v5_prcc_runner as runner

    if len(dataset_train) != int(fresh_features.shape[0]):
        raise ValueError("dataset/features length mismatch")
    pids = [
        _item_value(item, "pid", "label") for item in dataset_train
    ]
    clothes = [
        _item_value(item, "clothes_id", "clothes", "cloth_id") for item in dataset_train
    ]
    rows, _old_combined, summary, signature = runner.build_online_neighbor_graph(
        fresh_features,
        torch.as_tensor(pids, dtype=torch.long, device=device),
        device,
        max_negative_pids=max_negative_pids,
        block_size=block_size,
        previous_signature=previous_signature,
    )
    by_pid: dict[int, list[int]] = {}
    for index, pid in enumerate(pids):
        by_pid.setdefault(pid, []).append(index)
    positive_indices: list[int] = []
    cross_clothes = 0
    for anchor_index, pid in enumerate(pids):
        candidates = [index for index in by_pid.get(pid, []) if index != anchor_index]
        different_clothes = [index for index in candidates if clothes[index] >= 0 and clothes[index] != clothes[anchor_index]]
        chosen = (different_clothes or candidates)
        positive_indices.append(chosen[0] if chosen else anchor_index)
        if different_clothes:
            cross_clothes += 1
    positive_bank = fresh_features.detach().to(device=device, dtype=torch.float32).clone()
    if positive_indices:
        positive_bank = fresh_features.detach().to(device=device, dtype=torch.float32)[positive_indices]
    combined = torch.cat((fresh_features.detach().to(device=device, dtype=torch.float32), positive_bank), dim=0).detach()
    for anchor_index, anchor_rows in rows.items():
        for row in anchor_rows:
            row["positive_local"] = [int(fresh_features.shape[0]) + int(anchor_index)]
    summary = dict(summary)
    summary.update({
        "safe_positive_definition": "same_pid_cross_clothes_first_else_same_pid_first",
        "safe_cross_clothes_positive_count": int(cross_clothes),
        "safe_positive_fallback_count": int(len(pids) - cross_clothes),
    })
    return rows, combined, summary, signature
