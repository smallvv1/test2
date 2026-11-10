#!/usr/bin/env python3.10
"""Query-level audit for the frozen Bio+GPC/NDP paper models.

This script only evaluates already completed checkpoints.  It reproduces the
repository's official distance, camera, and clothes filtering rules, then
exports one row per query plus paired changes against Bio+GPC.  It is an audit
artifact, not a tuning or model-selection tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import train_ndp_listwise_full60 as runner
from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from processor.processor import test as processor_test
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_safe) + "\n", encoding="utf-8")


def ap_and_rank(orig_cmc: np.ndarray) -> tuple[bool, int | None, float]:
    positive = np.flatnonzero(orig_cmc)
    if positive.size == 0:
        return False, None, float("nan")
    cumulative = orig_cmc.cumsum()
    precision = cumulative / np.arange(1, len(orig_cmc) + 1, dtype=np.float64)
    ap = float((precision * orig_cmc).sum() / float(orig_cmc.sum()))
    return bool(orig_cmc[0]), int(positive[0]) + 1, ap


def query_rows(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    q_clothes: np.ndarray | None,
    g_clothes: np.ndarray | None,
    ltcc: bool,
) -> list[dict[str, Any]]:
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)
    rows: list[dict[str, Any]] = []
    for q_idx in range(len(q_pids)):
        order = indices[q_idx]
        remove = (g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx])
        if ltcc:
            remove = remove | ((g_pids[order] == q_pids[q_idx]) & (g_clothes[order] == q_clothes[q_idx]))
        keep = ~remove
        orig_cmc = matches[q_idx][keep]
        rank1, first_rank, ap = ap_and_rank(orig_cmc)
        rows.append({
            "query_index": int(q_idx),
            "pid": int(q_pids[q_idx]),
            "camid": int(q_camids[q_idx]),
            "cloth_id": int(q_clothes[q_idx]) if q_clothes is not None else None,
            "valid": bool(np.any(orig_cmc)),
            "rank1": rank1 if np.any(orig_cmc) else None,
            "first_positive_rank": first_rank,
            "ap": ap,
        })
    return rows


def evaluate_one(dataset_name: str, label: str, model_path: Path, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = runner.cfg_for(dataset_name, output_dir / "runtime" / label)
    dataset = build_dataset(cfg)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model = runner.build_model_for_evaluation(cfg, dataset)
    model.load_state_dict(state, strict=True)
    model.to("cuda:0").eval()
    _, transform_test = build_img_transforms(cfg)

    if dataset_name == "ltcc":
        query = dataset.query
        gallery = dataset.gallery
        evaluator = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=True)
        ltcc = True
    else:
        query = dataset.query_diff
        gallery = dataset.gallery
        evaluator = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=True)
        ltcc = False

    loader = DataLoader(
        ConcatDataset([ImageDataset(query, transform=transform_test), ImageDataset(gallery, transform=transform_test)]),
        batch_size=int(cfg.TEST.TEST_BATCH if hasattr(cfg.TEST, "TEST_BATCH") else 128),
        shuffle=False,
        num_workers=int(cfg.DATA.NUM_WORKERS),
        pin_memory=False,
        drop_last=False,
    )
    evaluator.reset()
    # The official evaluator writes to the same output directory and uses the
    # same processor/model path as train_ndp_listwise_full60.py.
    processor_test(cfg, model, evaluator, loader, runner.logger_for(output_dir / f"{label}.eval.log"), torch.device("cuda:0"), test=True, ltcc=ltcc, clothChanging=True)
    cmc, mAP, distmat, pids, camids, _qf, _gf, imgpaths = evaluator.compute()
    pids = np.asarray(pids)
    camids = np.asarray(camids)
    q_count = len(query)
    q_pids, g_pids = pids[:q_count], pids[q_count:]
    q_camids, g_camids = camids[:q_count], camids[q_count:]
    q_clothes = g_clothes = None
    if ltcc:
        clothes = np.asarray(evaluator.cloth_ids)
        q_clothes, g_clothes = clothes[:q_count], clothes[q_count:]
    rows = query_rows(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes, g_clothes, ltcc)
    for row, path in zip(rows, imgpaths[:q_count]):
        row["query_path"] = str(path)
    valid_rows = [row for row in rows if row["valid"]]
    rank1_count = sum(bool(row["rank1"]) for row in valid_rows)
    aps = np.asarray([row["ap"] for row in valid_rows], dtype=np.float64)
    ranks = np.asarray([row["first_positive_rank"] for row in valid_rows], dtype=np.float64)
    summary = {
        "dataset": dataset_name,
        "label": label,
        "model": str(model_path),
        "model_sha256": runner.sha256_file(model_path),
        "query_count": len(rows),
        "valid_query_count": len(valid_rows),
        "rank1_correct_count": int(rank1_count),
        "rank1": float(cmc[0] * 100.0),
        "mAP": float(mAP * 100.0),
        "mean_first_positive_rank": float(np.mean(ranks)) if len(ranks) else None,
        "median_first_positive_rank": float(np.median(ranks)) if len(ranks) else None,
        "ap_q25": float(np.quantile(aps, 0.25)) if len(aps) else None,
        "ap_median": float(np.median(aps)) if len(aps) else None,
        "ap_q75": float(np.quantile(aps, 0.75)) if len(aps) else None,
        "protocol": "LTCC clothes-changing" if ltcc else "PRCC C-to-A",
    }
    return summary, rows


def paired_summary(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_query = {int(row["query_index"]): row for row in baseline}
    pairs = [row for row in candidate if int(row["query_index"]) in by_query]
    base = [by_query[int(row["query_index"])] for row in pairs]
    valid = [i for i, row in enumerate(pairs) if row["valid"] and base[i]["valid"]]
    gain = sum((not bool(base[i]["rank1"])) and bool(pairs[i]["rank1"]) for i in valid)
    loss = sum(bool(base[i]["rank1"]) and (not bool(pairs[i]["rank1"])) for i in valid)
    ap_delta = np.asarray([pairs[i]["ap"] - base[i]["ap"] for i in valid], dtype=np.float64)
    return {
        "baseline": "Bio+GPC",
        "candidate": label,
        "paired_valid_queries": len(valid),
        "rank1_gained_queries": int(gain),
        "rank1_lost_queries": int(loss),
        "net_rank1_change_queries": int(gain - loss),
        "mean_ap_delta_pp": float(np.mean(ap_delta) * 100.0) if len(ap_delta) else None,
        "median_ap_delta_pp": float(np.median(ap_delta) * 100.0) if len(ap_delta) else None,
        "ap_improved_queries": int(np.sum(ap_delta > 1e-12)),
        "ap_decreased_queries": int(np.sum(ap_delta < -1e-12)),
        "ap_unchanged_queries": int(np.sum(np.abs(ap_delta) <= 1e-12)),
        "sum_ap_delta_pp": float(np.sum(ap_delta) * 100.0) if len(ap_delta) else None,
    }


def write_rows(path: Path, model_rows: dict[str, list[dict[str, Any]]]) -> None:
    labels = list(model_rows)
    base = model_rows[labels[0]]
    fields = ["query_index", "query_path", "pid", "camid", "cloth_id", "valid"]
    for label in labels:
        fields += [f"{label}_rank1", f"{label}_first_positive_rank", f"{label}_ap"]
    by_label = {label: {int(row["query_index"]): row for row in rows} for label, rows in model_rows.items()}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for base_row in base:
            idx = int(base_row["query_index"])
            out = {key: base_row.get(key) for key in ["query_index", "query_path", "pid", "camid", "cloth_id", "valid"]}
            for label in labels:
                row = by_label[label][idx]
                out[f"{label}_rank1"] = row["rank1"]
                out[f"{label}_first_positive_rank"] = row["first_positive_rank"]
                out[f"{label}_ap"] = row["ap"]
            writer.writerow(out)


def parse_models(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, sep, raw_path = value.partition("=")
        if not sep or not label or not raw_path:
            raise ValueError(f"model must be LABEL=PATH, got {value!r}")
        if not label.replace("_", "").isalnum():
            raise ValueError(f"model label must be shell-safe, got {label!r}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[label] = path
    if "Bio_GPC" not in result:
        raise ValueError("Bio_GPC is required as the paired baseline")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ltcc", "prcc"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="LABEL=MODEL_FINAL.PTH")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.model)
    summaries: list[dict[str, Any]] = []
    model_rows: dict[str, list[dict[str, Any]]] = {}
    for label, path in models.items():
        summary, rows = evaluate_one(args.dataset, label, path, args.output_dir)
        summaries.append(summary)
        model_rows[label] = rows
        print(json.dumps(summary, ensure_ascii=False))
    write_rows(args.output_dir / "query_metrics.csv", model_rows)
    baseline_rows = model_rows["Bio_GPC"]
    paired = [paired_summary(baseline_rows, model_rows[label], label) for label in models if label != "Bio_GPC"]
    write_json(args.output_dir / "summary.json", {"dataset": args.dataset, "models": summaries, "paired_vs_bio_gpc": paired})
    write_json(args.output_dir / "model_paths.json", {label: str(path) for label, path in models.items()})
    print(json.dumps({"dataset": args.dataset, "paired_vs_bio_gpc": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
