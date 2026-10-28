#!/usr/local/bin/python3.10
"""原始 GPC matched-budget 关系机会审计。

只读取冻结 LTCC 开发训练 PID 的一个完整 epoch，按照实际 Phase7I I3 的
candidate_lists 和 B=min(N_cross,N_same,N_id) 统计关系机会。F 项是离线
反事实覆盖率，绝不替换正式 GPC 预算，也不执行模型 forward/backward。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))

import gpc_a4_mechanism_probe as probe  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402

base = p7.base
SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
CONFIG = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen/B0_seed1234/config.yml"
EPOCH = 1
WORLD_SIZE = 2
SEEDS = (1234, 4321)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_protocol(seed: int) -> None:
    split_id = "LTCC_DEV_SPLIT_20261015_145338"
    phase = f"DIFFER_GPC_SCIENCE_RNG_V1|{split_id}"
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
    values: dict[int, set[int]] = defaultdict(set)
    for item in dataset.train:
        pid = int(item["pid"])
        if pid in train_pids:
            values[pid].add(int(item["clothes_id"]))
    if set(values) != train_pids:
        raise RuntimeError("冻结开发训练 PID 未完整覆盖")
    return {pid: sorted(clothes) for pid, clothes in sorted(values.items())}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pid_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {str(pid): count for pid, count in sorted(Counter(int(item["pid"]) for item in entries).items())}


def clothing_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {str(cloth): count for cloth, count in sorted(Counter(int(item["clothes_id"]) for item in entries).items())}


def selected_pid_counts(selected: list[tuple]) -> dict[str, int]:
    return {str(pid): count for pid, count in sorted(Counter(int(item[3][0]) for item in selected).items())}


def batch_row(batch_indices: list[int], loader: Any, path_to_index: dict[str, int], bank: dict[str, Any], seed: int, rank: int, iteration: int) -> dict[str, Any]:
    entries = [loader.dataset.samples[index][0] for index in batch_indices]
    candidates = p7.candidate_lists(entries, path_to_index, bank)
    n_cross = len(candidates["cross"])
    n_same = len(candidates["same"])
    n_id = len(candidates["id"])
    matched_budget = min(n_cross, n_same, n_id)
    selected = p7.stable_selection(candidates["id"], matched_budget, EPOCH, rank, iteration, "id")
    same_id_exists = n_id > 0
    no_same_id = not same_id_exists
    n_id_only_active = int(n_id > 0)
    same_clothes_selected = sum(int(item[3][1] == item[3][2]) for item in selected)
    cross_clothes_selected = sum(int(item[3][1] != item[3][2]) for item in selected)
    return {
        "seed": seed,
        "rank": rank,
        "epoch": EPOCH,
        "iteration": iteration,
        "batch_size": len(entries),
        "unique_pid_count": len(set(int(item["pid"]) for item in entries)),
        "pid_counts": json.dumps(pid_counts(entries), sort_keys=True),
        "clothing_counts": json.dumps(clothing_counts(entries), sort_keys=True),
        "n_cross": n_cross,
        "n_same": n_same,
        "n_id": n_id,
        "matched_budget": matched_budget,
        "no_same_id_positive": int(no_same_id),
        "same_id_exists_n_cross_zero": int(same_id_exists and n_cross == 0),
        "same_id_exists_n_same_zero": int(same_id_exists and n_same == 0),
        "same_id_exists_matched_budget_zero": int(same_id_exists and matched_budget == 0),
        "original_active": int(bool(selected)),
        "counterfactual_n_id_only_active": n_id_only_active,
        "counterfactual_n_id_only_relations": n_id,
        "selected_count": len(selected),
        "selected_pid_counts": json.dumps(selected_pid_counts(selected), sort_keys=True),
        "selected_same_clothes_count": same_clothes_selected,
        "selected_cross_clothes_count": cross_clothes_selected,
        "candidate_hash": p7.base.sha256_bytes(json.dumps({key: value for key, value in candidates.items()}, default=str, sort_keys=True).encode()),
        "selected_hash": p7.base.sha256_bytes(json.dumps(selected, default=str, sort_keys=True).encode()),
    }


def aggregate(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    fields = (
        "no_same_id_positive", "same_id_exists_n_cross_zero", "same_id_exists_n_same_zero",
        "same_id_exists_matched_budget_zero", "original_active", "counterfactual_n_id_only_active",
    )
    numeric = {field: sum(int(row[field]) for row in rows) for field in fields}
    all_batches = len(rows)
    same_id_batches = all_batches - numeric["no_same_id_positive"]
    return {
        "seed": seed,
        "all_batches": all_batches,
        "world_size": WORLD_SIZE,
        "counts": numeric,
        "fractions_all_batches": {field: numeric[field] / max(all_batches, 1) for field in fields},
        "same_id_batches": same_id_batches,
        "fractions_given_same_id": {
            field: numeric[field] / max(same_id_batches, 1)
            for field in fields if field != "no_same_id_positive"
        },
        "sum_candidate_cross": sum(int(row["n_cross"]) for row in rows),
        "sum_candidate_same": sum(int(row["n_same"]) for row in rows),
        "sum_candidate_id": sum(int(row["n_id"]) for row in rows),
        "sum_matched_budget": sum(int(row["matched_budget"]) for row in rows),
        "sum_selected": sum(int(row["selected_count"]) for row in rows),
        "sum_counterfactual_id_only_relations": sum(int(row["counterfactual_n_id_only_relations"]) for row in rows),
        "active_pid_exposure": dict(sorted(Counter(
            int(pid)
            for row in rows
            for pid in json.loads(row["selected_pid_counts"])
        ).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()
    global EPOCH
    EPOCH = int(args.epoch)
    if EPOCH < 1:
        raise ValueError("epoch 必须为正整数")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        set_protocol(seed)
        cfg = probe.load_cfg(CONFIG)
        cfg.defrost()
        cfg.DATA.NUM_WORKERS = 0
        cfg.DATA.PIN_MEMORY = False
        cfg.SOLVER.SEED = seed
        cfg.freeze()
        dataset = base.build_dataset(cfg)
        transform_train, _ = probe.build_img_transforms(cfg)
        state = active_state(dataset, train_pids)
        index_map = base.build_index_map(dataset)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        bank = b6.parse_relation_bank(dataset, cfg)
        bank["path_to_index"] = path_to_index
        seed_rows: list[dict[str, Any]] = []
        for rank in range(WORLD_SIZE):
            loader, meta = base.make_epoch_loader(dataset, transform_train, state, index_map, EPOCH, rank)
            for iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
                row = batch_row(batch_indices, loader, path_to_index, bank, seed, rank, iteration)
                row["plan_sha256"] = meta["plan_sha256"]
                rows.append(row)
                seed_rows.append(row)
        summaries.append({"summary": aggregate(seed_rows, seed), "loader_batches_per_rank": 596})
        print(f"中文进度：seed={seed} 完成 {len(seed_rows)} 个 rank-batch", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "matched_budget_by_batch.csv", rows)
    payload = {
        "status": "PASS",
        "run_time": args.run_time,
        "protocol": f"LTCC 冻结开发训练 PID；每个 seed 的 epoch{EPOCH}、rank0/rank1 全部计划 batch；不执行模型计算。",
        "budget_formula": "B=min(N_cross,N_same,N_id)",
        "counterfactual_F": "仅统计 N_id>0 的理论关系机会，不替换正式预算。",
        "train_pids": sorted(train_pids),
        "split_sha256": sha256_file(SPLIT),
        "config_sha256": sha256_file(CONFIG),
        "rows": len(rows),
        "summaries": summaries,
        "output_csv": str(args.output / "matched_budget_by_batch.csv"),
        "source_sha256": {str(path): sha256_file(path) for path in (ROOT / "scripts/gpc_matched_budget_opportunity_audit.py", ENGINE / "scripts/phase5b_id_damage_ds.py", ENGINE / "scripts/phase7i_relation_specificity.py")},
        "official_query_gallery_used": False,
        "cpu_model_fallback": False,
    }
    (args.output / "matched_budget_opportunity_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summaries": summaries, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
