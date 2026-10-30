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
    robust_positive: bool = False,
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
    positive_reference_indices: list[list[int]] = []
    reliable_positive_count = 0
    cross_clothes = 0
    if robust_positive:
        # Avoid one accelerator->CPU synchronization per candidate image.
        # Score padded same-PID candidates in accelerator batches, then copy
        # only the small score table for deterministic clothes-aware selection.
        candidate_lists: list[list[int]] = []
        for anchor_index, pid in enumerate(pids):
            candidates = [index for index in by_pid.get(pid, []) if index != anchor_index]
            candidate_lists.append(candidates or [anchor_index])
        candidate_width = max(len(values) for values in candidate_lists)
        candidate_index_tensor = torch.empty(
            (len(candidate_lists), candidate_width), dtype=torch.long
        )
        candidate_valid_tensor = torch.zeros(
            (len(candidate_lists), candidate_width), dtype=torch.bool
        )
        for anchor_index, candidates in enumerate(candidate_lists):
            candidate_index_tensor[anchor_index, :len(candidates)] = torch.as_tensor(
                candidates, dtype=torch.long
            )
            candidate_valid_tensor[anchor_index, :len(candidates)] = True
            if len(candidates) < candidate_width:
                candidate_index_tensor[anchor_index, len(candidates):] = candidates[0]
        feature_bank = fresh_features.detach().to(device=device, dtype=torch.float32)
        for start in range(0, len(candidate_lists), 512):
            stop = min(start + 512, len(candidate_lists))
            anchor_features = feature_bank[start:stop]
            candidate_indices = candidate_index_tensor[start:stop].to(device=device)
            candidate_features = feature_bank[candidate_indices]
            scores = (anchor_features.unsqueeze(1) * candidate_features).sum(dim=-1)
            scores = scores.detach().float().cpu().tolist()
            valid_rows = candidate_valid_tensor[start:stop].tolist()
            for local_index, (row_scores, row_valid) in enumerate(zip(scores, valid_rows)):
                anchor_index = start + local_index
                candidates = candidate_lists[anchor_index]
                valid_positions = [
                    position for position, valid in enumerate(row_valid) if valid
                ]
                ranked_positions = sorted(
                    valid_positions, key=lambda position: row_scores[position], reverse=True
                )
                different_positions = [
                    position for position in ranked_positions
                    if clothes[candidates[position]] >= 0
                    and clothes[candidates[position]] != clothes[anchor_index]
                ]
                chosen: list[int] = []
                chosen_clothes: set[int] = set()
                for position in different_positions:
                    candidate = candidates[position]
                    if clothes[candidate] in chosen_clothes:
                        continue
                    chosen.append(candidate)
                    chosen_clothes.add(clothes[candidate])
                    if len(chosen) == 2:
                        break
                if len(chosen) < 2:
                    for position in ranked_positions:
                        candidate = candidates[position]
                        if candidate not in chosen:
                            chosen.append(candidate)
                            if len(chosen) == 2:
                                break
                if len(different_positions) >= 2 and len({
                    clothes[candidates[position]] for position in different_positions[:2]
                }) == 2:
                    reliable_positive_count += 1
                if not chosen:
                    chosen = [anchor_index]
                positive_indices.append(chosen[0])
                positive_reference_indices.append(chosen)
                if any(
                    clothes[candidate] >= 0 and clothes[candidate] != clothes[anchor_index]
                    for candidate in candidates
                ):
                    cross_clothes += 1
    else:
        for anchor_index, pid in enumerate(pids):
            candidates = [index for index in by_pid.get(pid, []) if index != anchor_index]
            different_clothes = [
                index for index in candidates
                if clothes[index] >= 0 and clothes[index] != clothes[anchor_index]
            ]
            chosen = (different_clothes or candidates)[:1]
            if not chosen:
                chosen = [anchor_index]
            positive_indices.append(chosen[0])
            positive_reference_indices.append(chosen)
            if different_clothes:
                cross_clothes += 1
    flat_positive_indices = [index for values in positive_reference_indices for index in values]
    positive_bank = fresh_features.detach().to(device=device, dtype=torch.float32)[flat_positive_indices]
    combined = torch.cat((fresh_features.detach().to(device=device, dtype=torch.float32), positive_bank), dim=0).detach()
    positive_offsets: list[list[int]] = []
    offset = 0
    for values in positive_reference_indices:
        positive_offsets.append(list(range(offset, offset + len(values))))
        offset += len(values)
    for anchor_index, anchor_rows in rows.items():
        for row in anchor_rows:
            row["positive_local"] = [int(fresh_features.shape[0]) + int(index) for index in positive_offsets[anchor_index]]
            row["safe_positive_cross_clothes"] = bool(
                clothes[anchor_index] >= 0
                and positive_indices[anchor_index] < len(clothes)
                and clothes[positive_indices[anchor_index]] != clothes[anchor_index]
            )
            row["safe_positive_reliable"] = bool(
                robust_positive
                and len(positive_reference_indices[anchor_index]) >= 2
                and len({clothes[index] for index in positive_reference_indices[anchor_index][:2]}) == 2
            )
    summary = dict(summary)
    summary.update({
        "safe_positive_definition": (
            "same_pid_top2_distinct_clothes_by_current_similarity_else_best_legal"
            if robust_positive
            else "same_pid_cross_clothes_first_else_same_pid_first"
        ),
        "safe_cross_clothes_positive_count": int(cross_clothes),
        "safe_positive_fallback_count": int(len(pids) - cross_clothes),
        "safe_positive_reliable_count": int(reliable_positive_count),
        "safe_positive_robust": bool(robust_positive),
    })
    return rows, combined, summary, signature
