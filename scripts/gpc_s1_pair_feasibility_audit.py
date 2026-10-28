#!/usr/local/bin/python3.10
"""离线审计严格 S1-Pair 可辨识性。

S1-Pair 保持 S0 的 active batch、每批关系数、PID 和同衣/跨衣类型，
只允许替换当前 batch 内的具体图像对。脚本不执行模型 forward/backward，
不读取 query/gallery，不使用验证效果，也不修改历史 S0/S1 结果。
"""

from __future__ import annotations

import argparse
import csv
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

import gpc_strict_budget_plan_audit as audit  # noqa: E402


def relation_key(item: tuple) -> str:
    return json.dumps(audit.canonical(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def selected_hash(selected: dict[int, list[tuple]]) -> str:
    rows = []
    for iteration in sorted(selected):
        rows.append({"iteration": iteration, "relations": sorted(relation_key(x) for x in selected[iteration])})
    return audit.digest(rows)


def pair_type(item: tuple) -> str:
    return "same" if int(item[3][1]) == int(item[3][2]) else "cross"


def pair_candidates(target: tuple, candidates: list[tuple], excluded: set[str]) -> list[tuple]:
    target_pid = int(target[3][0])
    target_type = pair_type(target)
    result = []
    for item in candidates:
        key = relation_key(item)
        if key == relation_key(target) or key in excluded:
            continue
        if int(item[3][0]) != target_pid:
            continue
        if pair_type(item) != target_type:
            continue
        result.append(item)
    return result


def order_key(seed: int, rank: int, iteration: int, item: tuple) -> str:
    return audit.science.science_hash(
        seed=seed, epoch=audit.EPOCH, rank=rank, iteration=iteration,
        kind="S1_PAIR_ALTERNATIVE", parts=[item[0], item[1], item[2], item[3]],
    )


def build_seed_rank(seed: int, rank: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit.configure(seed)
    cfg = audit.probe.load_cfg(audit.CONFIG)
    cfg.defrost(); cfg.DATA.NUM_WORKERS = 0; cfg.DATA.PIN_MEMORY = False; cfg.SOLVER.SEED = seed; cfg.freeze()
    dataset = audit.base.build_dataset(cfg)
    transform, _ = audit.probe.build_img_transforms(cfg)
    split = json.loads(audit.SPLIT.read_text(encoding="utf-8"))
    train_pids = set(map(int, split["train_pids"]))
    state = audit.active_state(dataset, train_pids)
    index_map = audit.base.build_index_map(dataset)
    loader, _meta = audit.base.make_epoch_loader(dataset, transform, state, index_map, audit.EPOCH, rank)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    bank = audit.b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = path_to_index

    rows: list[dict[str, Any]] = []
    batch_records: dict[int, dict[str, Any]] = {}
    s0: dict[int, list[tuple]] = {}
    s1: dict[int, list[tuple]] = {}
    all_s0_keys: set[str] = set()
    for iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
        batch_indices = [int(index) for index in batch_indices]
        entries = [loader.dataset.samples[index][0] for index in batch_indices]
        candidates = audit.p7.candidate_lists(entries, path_to_index, bank)
        budget = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
        selected = audit.p7.stable_selection(candidates["id"], budget, audit.EPOCH, rank, iteration, "id")
        batch_records[iteration] = {
            "batch_indices": batch_indices,
            "candidates": candidates,
            "budget": budget,
        }
        if selected:
            s0[iteration] = list(selected)
            all_s0_keys.update(relation_key(item) for item in selected)

    used_s1: set[str] = set()
    for iteration in sorted(batch_records):
        record = batch_records[iteration]
        original = s0.get(iteration, [])
        current_candidates = record["candidates"]
        chosen = []
        pairings = []
        batch_used_s1 = set(used_s1)
        local_replaceable = 0
        globally_replaceable = 0
        alternatives_by_target = {}
        for target in sorted(original, key=relation_key):
            target_key = relation_key(target)
            local_alternatives = pair_candidates(target, current_candidates["id"], {target_key})
            global_alternatives = pair_candidates(target, current_candidates["id"], all_s0_keys | batch_used_s1)
            local_replaceable += int(bool(local_alternatives))
            globally_replaceable += int(bool(global_alternatives))
            alternatives_by_target[target_key] = {
                "local_legal_count": len(local_alternatives),
                "globally_available_count": len(global_alternatives),
            }
            ordered = sorted(global_alternatives, key=lambda item: order_key(seed, rank, iteration, item))
            replacement = ordered[0] if ordered else target
            chosen.append(replacement)
            batch_used_s1.add(relation_key(replacement))
            pairings.append({"target": target, "replacement": replacement, "replaced": relation_key(target) != relation_key(replacement)})
        if chosen:
            s1[iteration] = chosen
            used_s1.update(relation_key(item) for item in chosen)
        rows.append({
            "seed": seed, "rank": rank, "epoch": audit.EPOCH, "iteration": iteration,
            "batch_size": len(record["batch_indices"]),
            "n_cross": len(current_candidates["cross"]), "n_same": len(current_candidates["same"]),
            "n_id": len(current_candidates["id"]), "s0_budget": len(original),
            "s0_active": bool(original), "s1_active": bool(chosen),
            "s0_relation_keys": [relation_key(item) for item in original],
            "s1_relation_keys": [relation_key(item) for item in chosen],
            "s0_pid_types": sorted([[int(item[3][0]), pair_type(item)] for item in original]),
            "s1_pid_types": sorted([[int(item[3][0]), pair_type(item)] for item in chosen]),
            "target_alternative_counts": alternatives_by_target,
            "local_replaceable_s0_count": local_replaceable,
            "globally_replaceable_s0_count": globally_replaceable,
            "actual_replaced_count": sum(int(item["replaced"]) for item in pairings),
            "batch_indices_sha256": audit.digest(record["batch_indices"]),
        })

    flat0 = [item for values in s0.values() for item in values]
    flat1 = [item for values in s1.values() for item in values]
    profile = lambda values: {
        "relation_total": len(values),
        "active_batches": sum(bool(v) for v in (s0 if values is flat0 else s1).values()),
        "pid_counts": {str(k): v for k, v in sorted(Counter(int(item[3][0]) for item in values).items())},
        "same_clothes_relations": sum(pair_type(item) == "same" for item in values),
        "cross_clothes_relations": sum(pair_type(item) == "cross" for item in values),
        "unique_relations": len({relation_key(item) for item in values}),
        "duplicate_relations": len(values) - len({relation_key(item) for item in values}),
    }
    summary = {
        "seed": seed, "rank": rank, "epoch": audit.EPOCH, "all_batches": len(rows),
        "s0": profile(flat0), "s1_pair": profile(flat1),
        "s0_selected_relation_sha256": selected_hash(s0),
        "s1_pair_selected_relation_sha256": selected_hash(s1),
        "s0_total_relations": len(flat0), "s1_pair_total_relations": len(flat1),
        "s0_active_batches": len(s0), "s1_pair_active_batches": len(s1),
        "locally_replaceable_s0_relations": sum(int(row["local_replaceable_s0_count"]) for row in rows),
        "globally_replaceable_s0_relations": sum(int(row["globally_replaceable_s0_count"]) for row in rows),
        "unreplaceable_s0_relations": sum(int(row["s0_budget"] - row["local_replaceable_s0_count"]) for row in rows),
        "actual_replaced_relations": sum(int(row["actual_replaced_count"]) for row in rows),
        "same_pid_and_type_preserved": all(row["s0_pid_types"] == row["s1_pid_types"] for row in rows),
        "same_active_batches": len(s0) == len(s1),
        "same_per_batch_counts": all(len(s0.get(i, [])) == len(s1.get(i, [])) for i in range(1, len(rows) + 1)),
        "duplicate_free_s1": len({relation_key(item) for item in flat1}) == len(flat1),
        "model_forward_executed": False,
        "cpu_model_fallback": False,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    rows = []
    for seed in (1234, 4321):
        for rank in (0, 1):
            summary, batch_rows = build_seed_rank(seed, rank)
            summaries.append(summary); rows.extend(batch_rows)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    by_seed = []
    for seed in (1234, 4321):
        group = [x for x in summaries if x["seed"] == seed]
        by_seed.append({
            "seed": seed,
            "ranks": group,
            "s0_total_relations": sum(x["s0_total_relations"] for x in group),
            "locally_replaceable_s0_relations": sum(x["locally_replaceable_s0_relations"] for x in group),
            "globally_replaceable_s0_relations": sum(x["globally_replaceable_s0_relations"] for x in group),
            "unreplaceable_s0_relations": sum(x["unreplaceable_s0_relations"] for x in group),
            "actual_replaced_relations": sum(x["actual_replaced_relations"] for x in group),
            "replacement_fraction_of_s0": sum(x["actual_replaced_relations"] for x in group) / max(sum(x["s0_total_relations"] for x in group), 1),
            "locally_replaceable_fraction_of_s0": sum(x["locally_replaceable_s0_relations"] for x in group) / max(sum(x["s0_total_relations"] for x in group), 1),
            "globally_replaceable_fraction_of_s0": sum(x["globally_replaceable_s0_relations"] for x in group) / max(sum(x["s0_total_relations"] for x in group), 1),
        })
    payload = {
        "status": "PASS_OFFLINE_S1_PAIR_AUDIT",
        "run_time": args.run_time, "source_run_time": "20261020_112219",
        "epoch": audit.EPOCH, "train_only": True, "official_query_gallery_used": False,
        "validation_pid_used_for_weighting": False, "model_forward_executed": False,
        "cpu_model_fallback": False, "lambda_gpc": audit.LAMBDA_GPC,
        "definition": "S1-Pair保持S0 active batch、每批关系数、PID及同衣/跨衣类型，只替换当前batch内同PID不同图片关系；无合法替代则保留S0关系。",
        "source_sha256": {
            str(ROOT / "scripts/gpc_s1_pair_feasibility_audit.py"): audit.sha256_file(ROOT / "scripts/gpc_s1_pair_feasibility_audit.py"),
            str(ROOT / "scripts/gpc_strict_budget_plan_audit.py"): audit.sha256_file(ROOT / "scripts/gpc_strict_budget_plan_audit.py"),
            str(ENGINE / "scripts/phase7i_relation_specificity.py"): audit.sha256_file(ENGINE / "scripts/phase7i_relation_specificity.py"),
        },
        "split_sha256": audit.sha256_file(audit.SPLIT), "config_sha256": audit.sha256_file(audit.CONFIG),
        "summaries_by_seed": by_seed, "rank_summaries": summaries, "batch_rows": rows,
        "decision": "S1_PAIR_FEASIBILITY_LIMITED_UNLESS_REPLACEMENT_FRACTION_IS_SUFFICIENT",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "s1_pair_feasibility_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output / "s1_pair_feasibility_by_batch.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    print(json.dumps({"status": payload["status"], "output": str(args.output), "summaries_by_seed": by_seed}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
