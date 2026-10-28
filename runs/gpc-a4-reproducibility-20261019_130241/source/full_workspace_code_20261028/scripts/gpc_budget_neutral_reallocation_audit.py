#!/usr/local/bin/python3.10
"""S0/S1 固定总关系预算的 train-only 离线模拟。

S0 是原始 Phase7I I3 matched-budget 计划；S1 在相同 seed、epoch、rank
和固定 batch 计划上，从合法同 PID/不同图片候选中确定性随机放置恰好相同
数量的 relation，默认每个被选 batch 放一个 relation，以观察关系机会的
时间分布变化。两者都不执行模型计算，因此不会使用跨 batch 旧特征反向传播。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))

import gpc_science_rng as science  # noqa: E402
import gpc_a4_mechanism_probe as probe  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402

base = p7.base
SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
CONFIG = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen/B0_seed1234/config.yml"
EPOCH = 60
WORLD_SIZE = 2
SEEDS = (1234, 4321)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_protocol(seed: int) -> None:
    phase = f"{science.SCIENCE_RNG_NAMESPACE}|{science.SCIENCE_PROTOCOL_ID}"
    p7.PHASE = phase
    b6.PHASE = phase
    base.PHASE = phase
    p7.SEED = seed
    b6.SEED = seed
    base.SEED = seed
    p7.WORLD_SIZE = WORLD_SIZE
    b6.WORLD_SIZE = WORLD_SIZE
    base.WORLD_SIZE = WORLD_SIZE


def active_state(dataset: Any, train_pids: set[int]) -> dict[int, list[int]]:
    values: dict[int, set[int]] = {}
    for item in dataset.train:
        pid = int(item["pid"])
        if pid in train_pids:
            values.setdefault(pid, set()).add(int(item["clothes_id"]))
    if set(values) != train_pids:
        raise RuntimeError("冻结开发训练 PID 未完整覆盖")
    return {pid: sorted(clothes) for pid, clothes in sorted(values.items())}


def relation_sort_key(item: tuple, seed: int, rank: int, iteration: int) -> str:
    return science.science_hash(
        seed=seed, epoch=EPOCH, rank=rank, iteration=iteration,
        kind="budget_reallocation_relation", parts=[item[2], item[0], item[1]],
    )


def batch_sort_key(iteration: int, seed: int, rank: int) -> str:
    return science.science_hash(
        seed=seed, epoch=EPOCH, rank=rank, iteration=iteration,
        kind="budget_reallocation_batch", parts=[],
    )


def relation_key(item: tuple) -> str:
    return json.dumps([item[2], item[3]], separators=(",", ":"), sort_keys=True)


def selected_stats(selected_by_batch: dict[int, list[tuple]], mode: str, seed: int, rank: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [item for values in selected_by_batch.values() for item in values]
    pid_counts = Counter(int(item[3][0]) for item in selected)
    repeats = Counter(relation_key(item) for item in selected)
    same_clothes = sum(int(item[3][1] == item[3][2]) for item in selected)
    cross_clothes = sum(int(item[3][1] != item[3][2]) for item in selected)
    return {
        "seed": seed,
        "rank": rank,
        "mode": mode,
        "all_batches": len(rows),
        "active_batches": len(selected_by_batch),
        "relation_total": len(selected),
        "pid_counts": {str(key): value for key, value in sorted(pid_counts.items())},
        "pid_count": len(pid_counts),
        "same_clothes_relations": same_clothes,
        "cross_clothes_relations": cross_clothes,
        "same_clothes_fraction": same_clothes / max(len(selected), 1),
        "cross_clothes_fraction": cross_clothes / max(len(selected), 1),
        "unique_relation_keys": len(repeats),
        "relation_repeat_count": sum(value - 1 for value in repeats.values() if value > 1),
        "relation_repeat_rate": sum(value - 1 for value in repeats.values() if value > 1) / max(len(selected), 1),
        "active_batch_rate": len(selected_by_batch) / max(len(rows), 1),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    allocation_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        set_protocol(seed)
        cfg = probe.load_cfg(CONFIG)
        cfg.defrost(); cfg.DATA.NUM_WORKERS = 0; cfg.DATA.PIN_MEMORY = False; cfg.SOLVER.SEED = seed; cfg.freeze()
        dataset = base.build_dataset(cfg)
        transform, _ = probe.build_img_transforms(cfg)
        state = active_state(dataset, train_pids)
        index_map = base.build_index_map(dataset)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        bank = b6.parse_relation_bank(dataset, cfg)
        bank["path_to_index"] = path_to_index
        seed_summary = {"seed": seed, "rank_summaries": []}
        for rank in range(WORLD_SIZE):
            loader, meta = base.make_epoch_loader(dataset, transform, state, index_map, EPOCH, rank)
            batch_rows: list[dict[str, Any]] = []
            s0_by_batch: dict[int, list[tuple]] = {}
            candidate_by_batch: dict[int, list[tuple]] = {}
            for iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
                entries = [loader.dataset.samples[index][0] for index in batch_indices]
                candidates = p7.candidate_lists(entries, path_to_index, bank)
                budget = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
                s0 = p7.stable_selection(candidates["id"], budget, EPOCH, rank, iteration, "id")
                candidate_by_batch[iteration] = list(candidates["id"])
                if s0:
                    s0_by_batch[iteration] = s0
                batch_rows.append({
                    "seed": seed, "rank": rank, "epoch": EPOCH, "iteration": iteration,
                    "n_cross": len(candidates["cross"]), "n_same": len(candidates["same"]),
                    "n_id": len(candidates["id"]), "budget": budget,
                    "s0_selected": len(s0), "plan_sha256": meta["plan_sha256"],
                })
            total_budget = sum(int(row["budget"]) for row in batch_rows)
            eligible = sorted(candidate_by_batch, key=lambda iteration: batch_sort_key(iteration, seed, rank))
            if total_budget > len(eligible):
                raise RuntimeError("S1 无法在不重复 batch 的条件下放置固定总预算")
            chosen_batches = set(eligible[:total_budget])
            s1_by_batch: dict[int, list[tuple]] = {}
            for iteration in sorted(chosen_batches):
                candidates = candidate_by_batch[iteration]
                chosen = sorted(candidates, key=lambda item: relation_sort_key(item, seed, rank, iteration))[:1]
                s1_by_batch[iteration] = chosen
            for row in batch_rows:
                iteration = int(row["iteration"])
                row["s0_active"] = int(iteration in s0_by_batch)
                row["s1_active"] = int(iteration in s1_by_batch)
                row["s0_selected_relation_keys"] = json.dumps([relation_key(item) for item in s0_by_batch.get(iteration, [])], ensure_ascii=False)
                row["s1_selected_relation_keys"] = json.dumps([relation_key(item) for item in s1_by_batch.get(iteration, [])], ensure_ascii=False)
                allocation_rows.append(row)
            s0_summary = selected_stats(s0_by_batch, "S0", seed, rank, batch_rows)
            s1_summary = selected_stats(s1_by_batch, "S1", seed, rank, batch_rows)
            s0_summary["plan_sha256"] = meta["plan_sha256"]
            s1_summary["plan_sha256"] = meta["plan_sha256"]
            seed_summary["rank_summaries"].extend([s0_summary, s1_summary])
        summaries.append(seed_summary)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "s0_s1_allocation_by_batch.csv", allocation_rows)
    payload = {
        "status": "PASS_OFFLINE_SIMULATION",
        "run_time": args.run_time,
        "epoch": EPOCH,
        "train_only": True,
        "official_query_gallery_used": False,
        "validation_pid_used_for_weighting": False,
        "cpu_model_fallback": False,
        "S0": "原始 Phase7I I3 matched-budget；B=min(N_cross,N_same,N_id)",
        "S1": "固定每个 seed/rank 的 S0 relation 总数；按科学命名空间确定性随机选择相同数量 batch，每个 S1 batch 放一个合法同 PID/不同图片关系",
        "budget_matching": "每个 seed/rank 的 relation_total、基础 batch 计划和 augmentation 计划保持不变；实际梯度范数未在本次离线模拟中测量",
        "split_sha256": sha256_file(SPLIT),
        "config_sha256": sha256_file(CONFIG),
        "summaries": summaries,
        "output_csv": str(args.output / "s0_s1_allocation_by_batch.csv"),
        "relation_cosine_distribution": "未计算：本阶段禁止使用跨 batch 旧特征，且 PPU 模型路径当前 runtime SIGABRT；不得用离线特征替代",
        "gradient_scale_estimate": "未计算：关系数量相同不等于梯度预算相同，必须等 PPU 技术门禁恢复后用同状态 forward/autograd 直接测量",
        "source_sha256": {str(path): sha256_file(path) for path in (ROOT / "scripts/gpc_budget_neutral_reallocation_audit.py", ROOT / "scripts/gpc_science_rng.py", ENGINE / "scripts/phase5b_id_damage_ds.py", ENGINE / "scripts/phase7i_relation_specificity.py")},
    }
    (args.output / "budget_neutral_reallocation_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summaries": summaries, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
