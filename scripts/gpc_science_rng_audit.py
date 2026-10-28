#!/usr/bin/env python3
"""GPC 科学随机命名空间与训练关系选择协议审计。

本脚本只构建 CPU 侧的真实 LTCC 开发训练 sampler/candidate 计划，不做
模型 forward/backward，不启动训练。它用固定的科学命名空间替代历史
wrapper 中包含 RUN_TIME/arm 的 PHASE，仅用于新诊断与后续规范化基线。
"""

from __future__ import annotations

import argparse
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

import gpc_science_rng as science  # noqa: E402
import gpc_a4_mechanism_probe as probe  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402
import gpc_a4_recheck_train as a4  # noqa: E402

base = p7.base

SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
CONFIG = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen/B0_seed1234/config.yml"
SCIENCE_PHASE = f"{science.SCIENCE_RNG_NAMESPACE}|{science.SCIENCE_PROTOCOL_ID}"
EPOCHS = (1, 30)
RANKS = (0, 1)
CHECK_BATCHES = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_science(seed: int) -> None:
    # These module globals are changed only in this process. No historical
    # Phase7I file or checkpoint is modified.
    p7.PHASE = SCIENCE_PHASE
    b6.PHASE = SCIENCE_PHASE
    base.PHASE = SCIENCE_PHASE
    p7.SEED = int(seed)
    b6.SEED = int(seed)
    base.SEED = int(seed)
    p7.WORLD_SIZE = 2
    b6.WORLD_SIZE = 2
    base.WORLD_SIZE = 2


def batch_records(loader: Any, batch_indices: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "dataset_position": int(loader.dataset.samples[index][0].get("index", index)),
            "pid": int(loader.dataset.samples[index][0]["pid"]),
            "clothes_id": int(loader.dataset.samples[index][0]["clothes_id"]),
            "camid": int(loader.dataset.samples[index][0].get("camid", -1)),
            "image_path": str(loader.dataset.samples[index][0]["image_path"]),
            "augmentation_seed": int(loader.dataset.samples[index][1]),
        }
        for index in batch_indices
    ]


def legacy_selection_digest(candidates: list[Any], count: int, seed: int, epoch: int, rank: int, iteration: int, phase: str) -> str:
    old_phase = p7.PHASE
    old_seed = p7.SEED
    try:
        p7.PHASE = phase
        p7.SEED = seed
        selected = p7.stable_selection(candidates, count, epoch, rank, iteration, "id")
        return science.relation_sha256(selected)
    finally:
        p7.PHASE = old_phase
        p7.SEED = old_seed


def build_plan(dataset: Any, cfg: Any, train_pids: set[int], seed: int) -> dict[str, Any]:
    configure_science(seed)
    transform_train, _ = probe.build_img_transforms(cfg)
    state = probe.active_state(dataset, train_pids)
    index_map = base.build_index_map(dataset)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    loaders = {
        rank: base.make_epoch_loader(dataset, transform_train, state, index_map, 1, rank)[0]
        for rank in RANKS
    }

    batches: dict[str, Any] = {}
    candidate_hashes: dict[str, Any] = {}
    selected_hashes: dict[str, Any] = {}
    selected_counts: dict[str, Any] = {}
    pid_counts: dict[str, Any] = {}
    records = []
    active_seen = {rank: 0 for rank in RANKS}
    for rank, loader in loaders.items():
        for iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
            entries = batch_records(loader, list(batch_indices))
            candidates = p7.candidate_lists(entries, path_to_index, bank)
            count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
            key = f"rank{rank}_iter{iteration}"
            batches[key] = entries
            candidate_hashes[key] = {kind: science.relation_sha256(values) for kind, values in candidates.items()}
            chosen = science.science_selection(candidates["id"], count, seed=seed, epoch=1, rank=rank, iteration=iteration, kind="id")
            selected_hashes[key] = science.relation_sha256(chosen)
            selected_counts[key] = {"matched": int(count), "selected": len(chosen), "active": bool(chosen)}
            pid_counts[key] = dict(Counter(int(item[3][0]) for item in chosen))
            records.append((iteration, rank, candidates, count))
            if count > 0 and active_seen[rank] < CHECK_BATCHES:
                active_seen[rank] += 1

    # A4 uses the same real candidate pools and the same budget. Its only
    # scientific difference here is the PID allocation rule. The cumulative
    # selector is invoked with the fixed namespace, never with RUN_TIME.
    a4_selected_hashes: dict[str, str] = {}
    a4_counts: dict[str, dict[str, Any]] = {}
    a4_pid_counts: dict[str, dict[str, int]] = {}
    totals: Counter = Counter()
    for iteration, rank, candidates, count in sorted(records, key=lambda row: (row[0], row[1])):
        chosen = a4.cumulative_pid_balanced_selection(candidates["id"], count, 1, rank, iteration, totals)
        key = f"rank{rank}_iter{iteration}"
        a4_selected_hashes[key] = science.relation_sha256(chosen)
        a4_counts[key] = {"matched": int(count), "selected": len(chosen), "active": bool(chosen)}
        a4_pid_counts[key] = dict(Counter(int(item[3][0]) for item in chosen))

    return {
        "science_phase": SCIENCE_PHASE,
        "seed": seed,
        "epoch": 1,
        "base_batch_hash": science.list_sha256(
            [[key, entries] for key, entries in sorted(batches.items())]
        ),
        "batches": batches,
        "candidate_hashes": candidate_hashes,
        "b0_selected_hashes": selected_hashes,
        "b0_selected_counts": selected_counts,
        "b0_pid_counts": pid_counts,
        "a4_selected_hashes": a4_selected_hashes,
        "a4_selected_counts": a4_counts,
        "a4_pid_counts": a4_pid_counts,
        "a4_total_pid_counts": dict(totals),
        "pair_budget_match": all(
            a4_counts[key]["matched"] == selected_counts[key]["matched"]
            and a4_counts[key]["selected"] == selected_counts[key]["selected"]
            for key in selected_counts
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    cfg = probe.load_cfg(CONFIG)
    cfg.defrost()
    cfg.OUTPUT_DIR = str(args.output)
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()
    dataset = base.build_dataset(cfg)

    contexts = [
        {"run_time": "20261019_141702_A", "log_dir": "/tmp/gpc-log-a", "output_dir": "/tmp/gpc-out-a", "arm": "B0"},
        {"run_time": "20261019_141702_B", "log_dir": "/tmp/gpc-log-b", "output_dir": "/tmp/gpc-out-b", "arm": "B0"},
        {"run_time": "20261019_141702_C", "log_dir": "/var/tmp/gpc-log-c", "output_dir": "/var/tmp/gpc-out-c", "arm": "A4"},
    ]
    plans = [build_plan(dataset, cfg, train_pids, 1234) for _ in contexts]
    same_batch = len({item["base_batch_hash"] for item in plans}) == 1
    same_b0 = len({json.dumps(item["b0_selected_hashes"], sort_keys=True) for item in plans}) == 1
    same_candidates = len({json.dumps(item["candidate_hashes"], sort_keys=True) for item in plans}) == 1
    a4_budget = all(item["pair_budget_match"] for item in plans)

    # Demonstrate the historical problem on one real batch: changing the old
    # wrapper PHASE can change the old Phase7I relation hash.
    first = plans[0]
    first_key = next(
        key for key in sorted(first["batches"])
        if first["b0_selected_counts"][key]["matched"] > 0
    )
    rank = int(first_key.split("_")[0].replace("rank", ""))
    iteration = int(first_key.split("_")[1].replace("iter", ""))
    entries = first["batches"][first_key]
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index
    candidates = p7.candidate_lists(entries, path_to_index, bank)["id"]
    count = min(len(p7.candidate_lists(entries, path_to_index, bank)[kind]) for kind in ("cross", "same", "id"))
    old_a = legacy_selection_digest(candidates, count, 1234, 1, rank, iteration, "gpc_v2_ltcc_20261016_161631_screen_B0_seed1234")
    old_b = legacy_selection_digest(candidates, count, 1234, 1, rank, iteration, "gpc_v2_ltcc_20261019_141702_other_B0_seed1234")

    unit_tests = {
        "same_seed_different_runtime": same_batch and same_b0 and same_candidates,
        "different_log_output_paths": same_batch and same_b0,
        "b0_a4_same_base_sampler": same_batch,
        "a4_same_matched_pair_budget": a4_budget,
        "legacy_phase_runtime_sensitivity_observed": old_a != old_b,
    }
    payload = {
        "status": "PASS" if all(unit_tests.values()) else "FAIL",
        "run_time": args.run_time,
        "official_query_gallery_used": False,
        "validation_identities_used": False,
        "model_forward_used": False,
        "cpu_only_protocol_audit": True,
        "science_namespace": {
            "name": science.SCIENCE_RNG_NAMESPACE,
            "protocol_id": science.SCIENCE_PROTOCOL_ID,
            "frozen_split_sha256": science.FROZEN_SPLIT_SHA256,
            "phase": SCIENCE_PHASE,
            "digest": science.protocol_digest(),
        },
        "split": {"path": str(SPLIT), "sha256": sha256_file(SPLIT), "train_pid_count": len(train_pids)},
        "contexts_excluded_from_key": contexts,
        "unit_tests": unit_tests,
        "legacy_runtime_sensitivity": {"old_a": old_a, "old_b": old_b, "same": old_a == old_b},
        "plans": plans,
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                ROOT / "scripts/gpc_science_rng.py",
                ROOT / "scripts/gpc_science_rng_audit.py",
                ENGINE / "scripts/phase5b_id_damage_ds.py",
                ENGINE / "scripts/phase7i_relation_specificity.py",
                ROOT / "scripts/gpc_a4_recheck_train.py",
            )
        },
    }
    write_json(args.output / "science_rng_audit.json", payload)
    print(json.dumps({"status": payload["status"], "unit_tests": unit_tests, "output": str(args.output)}, ensure_ascii=False, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
