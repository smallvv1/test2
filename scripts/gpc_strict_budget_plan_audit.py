#!/usr/local/bin/python3.10
"""严格审计 S0/S1/S2 的 train-only 固定关系预算计划。

S0 保留原始 Phase 7I I3 的每 batch matched budget 与 selector。
S1 保留 S0 的 active batch 集合及每 batch 关系数，只在同一 batch 内
选择合法同 PID/不同图片关系，并要求全局关系键不重复。
S2 保留整轮关系总数和 active batch 数，在其它合法 batch 中重排关系，
使按 active batch 计的累计 GPC loss 系数与 S0 相同。

本脚本只构造关系计划，不执行模型 forward/backward；不得用跨 batch 旧特征。
"""

from __future__ import annotations

import argparse
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

import gpc_a4_mechanism_probe as probe  # noqa: E402
import gpc_science_rng as science  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402
import phase7i_relation_specificity as p7  # noqa: E402

base = p7.base
SPLIT = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/development_split_manifest.json"
CONFIG = ROOT / "outputs/gpc_a4_ltcc_20261016_161631/training/screen/B0_seed1234/config.yml"
EPOCH = 60
WORLD_SIZE = 2
SEEDS = (1234, 4321)
LAMBDA_GPC = 0.1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, tuple):
        return [canonical(item) for item in value]
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return value


def digest(value: Any) -> str:
    raw = json.dumps(canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def relation_key(item: tuple) -> str:
    # Include endpoints and state metadata. This catches the same image pair
    # recurring in different batches, including repeated sampler exposure.
    return json.dumps(canonical(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def relation_hash_rows(selected: dict[int, list[tuple]]) -> str:
    rows = []
    for iteration in sorted(selected):
        rows.append({"iteration": iteration, "relations": sorted(relation_key(item) for item in selected[iteration])})
    return digest(rows)


def candidate_hash_rows(candidates: dict[int, dict[str, list[tuple]]]) -> str:
    rows = []
    for iteration in sorted(candidates):
        rows.append({
            "iteration": iteration,
            "candidates": {
                kind: sorted(relation_key(item) for item in candidates[iteration][kind])
                for kind in ("cross", "same", "id")
            },
        })
    return digest(rows)


def configure(seed: int) -> None:
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


def order_key(kind: str, seed: int, rank: int, iteration: int, item: tuple) -> str:
    return science.science_hash(
        seed=seed, epoch=EPOCH, rank=rank, iteration=iteration, kind=kind,
        parts=[item[0], item[1], item[2], item[3]],
    )


def batch_order(seed: int, rank: int, iteration: int, kind: str) -> str:
    return science.science_hash(seed=seed, epoch=EPOCH, rank=rank, iteration=iteration, kind=kind, parts=[])


def unique_ordered(items: list[tuple], seed: int, rank: int, iteration: int, kind: str, used: set[str]) -> list[tuple]:
    result = []
    local = set()
    for item in sorted(items, key=lambda value: order_key(kind, seed, rank, iteration, value)):
        key = relation_key(item)
        if key in used or key in local:
            continue
        local.add(key)
        result.append(item)
    return result


def choose_s1(
    s0: dict[int, list[tuple]], candidates: dict[int, dict[str, list[tuple]]], seed: int, rank: int,
) -> tuple[dict[int, list[tuple]], dict[str, Any]]:
    selected: dict[int, list[tuple]] = {}
    used: set[str] = set()
    unchanged = 0
    failure = []
    for iteration in sorted(s0):
        need = len(s0[iteration])
        pool = unique_ordered(candidates[iteration]["id"], seed, rank, iteration, "S1_local", used)
        chosen = pool[:need]
        if len(chosen) != need:
            failure.append({"iteration": iteration, "needed": need, "available_unique": len(pool)})
        selected[iteration] = chosen
        unchanged += sum(relation_key(a) == relation_key(b) for a, b in zip(
            sorted(s0[iteration], key=relation_key), sorted(chosen, key=relation_key)))
        used.update(relation_key(item) for item in chosen)
    return selected, {
        "same_active_batches": sorted(selected) == sorted(s0),
        "same_per_batch_counts": all(len(selected.get(k, [])) == len(v) for k, v in s0.items()),
        "duplicate_free": len(used) == sum(len(v) for v in selected.values()),
        "unchanged_relation_count": unchanged,
        "failures": failure,
    }


def choose_s2(
    all_candidates: dict[int, dict[str, list[tuple]]], s0: dict[int, list[tuple]], seed: int, rank: int,
) -> tuple[dict[int, list[tuple]], dict[str, Any]]:
    total = sum(len(v) for v in s0.values())
    active_need = len(s0)
    unique_by_batch = {
        iteration: unique_ordered(all_candidates[iteration]["id"], seed, rank, iteration, "S2_pool", set())
        for iteration in all_candidates
    }
    eligible = [iteration for iteration in all_candidates if unique_by_batch[iteration]]
    ordered = sorted(eligible, key=lambda iteration: batch_order(seed, rank, iteration, "S2_batch"))
    # Start with distinct batches and one relation per batch. Then distribute
    # remaining relation slots while preserving the same active-batch count.
    selected_batches = ordered[:active_need]
    selected: dict[int, list[tuple]] = {iteration: [] for iteration in selected_batches}
    used: set[str] = set()
    failure = []
    for iteration in selected_batches:
        pool = unique_ordered(unique_by_batch[iteration], seed, rank, iteration, "S2_relation", used)
        if not pool:
            failure.append({"iteration": iteration, "reason": "no globally unique relation"})
            continue
        selected[iteration].append(pool[0])
        used.add(relation_key(pool[0]))
    remaining = total - len(used)
    progress = True
    while remaining > 0 and progress:
        progress = False
        for iteration in sorted(selected, key=lambda value: batch_order(seed, rank, value, "S2_extra")):
            if remaining <= 0:
                break
            pool = unique_ordered(unique_by_batch[iteration], seed, rank, iteration, "S2_extra_relation", used)
            if not pool:
                continue
            selected[iteration].append(pool[0])
            used.add(relation_key(pool[0]))
            remaining -= 1
            progress = True
    if remaining:
        failure.append({"remaining_relations": remaining, "reason": "target batches lack enough unique legal relations"})
    selected = {iteration: values for iteration, values in selected.items() if values}
    return selected, {
        "target_relation_total": total,
        "target_active_batches": active_need,
        "same_relation_total": sum(len(v) for v in selected.values()) == total,
        "same_active_batches": len(selected) == active_need,
        "duplicate_free": len(used) == sum(len(v) for v in selected.values()),
        "failures": failure,
    }


def stats(mode: str, seed: int, rank: int, rows: list[dict[str, Any]], selected: dict[int, list[tuple]], candidates: dict[int, dict[str, list[tuple]]], batch_hash: str, candidate_hash: str) -> dict[str, Any]:
    flat = [item for values in selected.values() for item in values]
    relation_keys = [relation_key(item) for item in flat]
    repeats = Counter(relation_keys)
    same = sum(int(item[3][1] == item[3][2]) for item in flat)
    cross = sum(int(item[3][1] != item[3][2]) for item in flat)
    selected_hash = relation_hash_rows(selected)
    composite = digest({"batch_plan_sha256": batch_hash, "candidate_relation_sha256": candidate_hash, "selected_relation_sha256": selected_hash})
    per_batch = {
        str(iteration): {
            "selected_pairs": len(selected.get(iteration, [])),
            "loss_coefficient_per_pair": (LAMBDA_GPC / len(selected[iteration])) if selected.get(iteration) else 0.0,
            "batch_loss_coefficient": LAMBDA_GPC if selected.get(iteration) else 0.0,
        }
        for iteration in range(1, len(rows) + 1)
    }
    return {
        "seed": seed, "rank": rank, "mode": mode, "all_batches": len(rows),
        "active_batches": len(selected), "relation_total": len(flat),
        "pid_counts": {str(k): v for k, v in sorted(Counter(int(item[3][0]) for item in flat).items())},
        "pid_count": len(set(int(item[3][0]) for item in flat)),
        "same_clothes_relations": same, "cross_clothes_relations": cross,
        "same_clothes_fraction": same / max(len(flat), 1), "cross_clothes_fraction": cross / max(len(flat), 1),
        "unique_relation_keys": len(repeats), "relation_repeat_count": sum(v - 1 for v in repeats.values() if v > 1),
        "relation_repeat_rate": sum(v - 1 for v in repeats.values() if v > 1) / max(len(flat), 1),
        "batch_plan_sha256": batch_hash, "candidate_relation_sha256": candidate_hash,
        "selected_relation_sha256": selected_hash, "plan_sha256": composite,
        "active_batch_rate": len(selected) / max(len(rows), 1),
        "cumulative_loss_coefficient": LAMBDA_GPC * len(selected),
        "per_batch_loss_coefficients": per_batch,
        "candidate_batches_with_id": sum(bool(candidates[i]["id"]) for i in candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    summaries = []
    csv_rows = []
    for seed in SEEDS:
        configure(seed)
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
            rows = []
            candidates: dict[int, dict[str, list[tuple]]] = {}
            s0: dict[int, list[tuple]] = {}
            for iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
                entries = [loader.dataset.samples[index][0] for index in batch_indices]
                current = p7.candidate_lists(entries, path_to_index, bank)
                budget = min(len(current["cross"]), len(current["same"]), len(current["id"]))
                selected = p7.stable_selection(current["id"], budget, EPOCH, rank, iteration, "id")
                candidates[iteration] = {kind: list(current[kind]) for kind in ("cross", "same", "id")}
                if selected:
                    s0[iteration] = selected
                rows.append({
                    "seed": seed, "rank": rank, "epoch": EPOCH, "iteration": iteration,
                    "batch_indices": json.dumps([int(index) for index in batch_indices]),
                    "n_cross": len(current["cross"]), "n_same": len(current["same"]), "n_id": len(current["id"]),
                    "budget": budget, "s0_selected": len(selected),
                })
            batch_hash = digest([{"iteration": int(row["iteration"]), "batch_indices": json.loads(row["batch_indices"])} for row in rows])
            candidate_hash = candidate_hash_rows(candidates)
            s1, s1_gate = choose_s1(s0, candidates, seed, rank)
            s2, s2_gate = choose_s2(candidates, s0, seed, rank)
            gate = {
                "S1": s1_gate,
                "S2": s2_gate,
                "S0_duplicate_free": len({relation_key(item) for values in s0.values() for item in values}) == sum(len(v) for v in s0.values()),
            }
            for mode, selected in (("S0", s0), ("S1", s1), ("S2", s2)):
                summary = stats(mode, seed, rank, rows, selected, candidates, batch_hash, candidate_hash)
                summary["protocol_gate"] = gate[mode] if mode in gate else {"source": "S0 original selector"}
                seed_summary["rank_summaries"].append(summary)
            for row in rows:
                iteration = int(row["iteration"])
                row.update({
                    "s0_active": int(iteration in s0), "s1_active": int(iteration in s1), "s2_active": int(iteration in s2),
                    "s0_selected_relation_keys": json.dumps([relation_key(item) for item in s0.get(iteration, [])], ensure_ascii=False),
                    "s1_selected_relation_keys": json.dumps([relation_key(item) for item in s1.get(iteration, [])], ensure_ascii=False),
                    "s2_selected_relation_keys": json.dumps([relation_key(item) for item in s2.get(iteration, [])], ensure_ascii=False),
                    "batch_plan_sha256": batch_hash,
                    "candidate_relation_sha256": candidate_hash,
                })
                csv_rows.append(row)
        summaries.append(seed_summary)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "s0_s1_s2_allocation_by_batch.csv"
    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(csv_rows)
    all_gates = [summary["protocol_gate"] for item in summaries for summary in item["rank_summaries"] if summary["mode"] in ("S1", "S2")]
    s1_ok = all(gate.get("same_active_batches") and gate.get("same_per_batch_counts") and gate.get("duplicate_free") and not gate.get("failures") for gate in [summary["protocol_gate"] for item in summaries for summary in item["rank_summaries"] if summary["mode"] == "S1"])
    s2_ok = all(gate.get("same_relation_total") and gate.get("same_active_batches") and gate.get("duplicate_free") and not gate.get("failures") for gate in [summary["protocol_gate"] for item in summaries for summary in item["rank_summaries"] if summary["mode"] == "S2"])
    payload = {
        "status": "PASS_OFFLINE_PLAN_AUDIT" if s1_ok and s2_ok else "FAIL_OFFLINE_PLAN_AUDIT",
        "run_time": args.run_time, "epoch": EPOCH, "train_only": True,
        "official_query_gallery_used": False, "validation_pid_used_for_weighting": False, "cpu_model_fallback": False,
        "lambda_gpc": LAMBDA_GPC,
        "definitions": {
            "S0": "原始 Phase7I I3 matched-budget 与 stable selector",
            "S1": "保持 S0 active batch 与每 batch 关系数，仅改变当前 batch 内合法同 PID/不同图片关系；全局去重门禁",
            "S2": "保持整轮 relation_total 与 active_batches，跨可用 batch 重排；累计 batch loss coefficient 与 S0 匹配",
        },
        "hash_definition": {
            "batch_plan_sha256": "有序 batch image indices",
            "candidate_relation_sha256": "每 batch 全部 cross/same/id 候选关系键",
            "selected_relation_sha256": "每 batch 选中关系键",
            "plan_sha256": "以上三种 hash 的组合 hash",
        },
        "split_sha256": sha256_file(SPLIT), "config_sha256": sha256_file(CONFIG),
        "summaries": summaries, "output_csv": str(csv_path),
        "gradient_measurement": "未执行模型 forward/backward；需 PPU 门禁通过后对 S0/S1/S2 同状态直接测量",
        "source_sha256": {str(path): sha256_file(path) for path in (ROOT / "scripts/gpc_strict_budget_plan_audit.py", ROOT / "scripts/gpc_science_rng.py", ENGINE / "scripts/phase7i_relation_specificity.py")},
    }
    (args.output / "strict_budget_plan_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "s1_ok": s1_ok, "s2_ok": s2_ok}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
