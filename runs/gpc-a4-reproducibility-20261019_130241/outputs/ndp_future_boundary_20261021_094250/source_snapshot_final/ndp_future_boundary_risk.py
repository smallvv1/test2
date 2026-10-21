#!/usr/bin/env python3
"""同一 continuation 上的未来局部边界风险重构。

只读取阶段3在 PPU eval/no_grad 下保存的固定关系优势值；不重新选择未来关系，
不执行 optimizer.step。输出累计持续、首次触发、增量退化、真实排序翻转和
四状态标签，并按不等长更新区间分层。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ndp_risk_benchmark import auc_rank, average_precision, top_fraction_metrics


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("seed", "update", "anchor", "pid", "positive_count", "negative_count"):
                r[k] = int(r[k])
            for k in ("reference_advantage", "current_advantage", "advantage_drop", "epsilon"):
                r[k] = float(r[k])
            r["active"] = r["active"].lower() == "true"
            rows.append(r)
    return rows


def load_groups(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(g["anchor"]): g for g in data["groups"]}


def load_meta(path: Path) -> dict[int, dict]:
    import numpy as np
    bank = np.load(path, allow_pickle=True)
    # NpzFile 字段访问会重复解压；先物化一次，避免 9576×字段数的重复 I/O。
    pid = bank["pid"]; clothes = bank["clothes"]; camid = bank["camid"]; paths = bank["image_path"]
    return {i: {"pid": int(pid[i]), "clothes": int(clothes[i]), "camid": int(camid[i]), "image_path": str(paths[i])} for i in range(len(pid))}


def metric_block(rows: list[dict], label: str, score_names: list[str]) -> dict:
    y = np.asarray([bool(r[label]) for r in rows])
    result = {"rows": len(rows), "positive_count": int(y.sum()), "event_rate": float(y.mean()) if len(y) else None, "scores": {}}
    for name in score_names:
        score = np.asarray([float(r[name]) for r in rows], dtype=np.float64)
        result["scores"][name] = {
            "roc_auc": auc_rank(y, score),
            "average_precision": average_precision(y, score),
            "top10": top_fraction_metrics(y, score),
        }
    return result


def classify_transition(prev_margin: float, target_margin: float, prev_drop: float,
                        target_drop: float, epsilon: float,
                        epsilon_increment: float) -> dict[str, bool | str]:
    """纯标签定义；风险分数不得读取 target 状态。"""
    prev_correct = prev_margin > 0
    target_correct = target_margin > 0
    return {
        "prev_correct": prev_correct,
        "target_correct": target_correct,
        "state": (
            "correct_to_correct" if prev_correct and target_correct else
            "correct_to_wrong" if prev_correct and not target_correct else
            "wrong_to_correct" if not prev_correct and target_correct else
            "wrong_to_wrong"
        ),
        "y_persistent": target_drop > epsilon,
        "y_onset": prev_drop <= epsilon and target_drop > epsilon,
        "y_increment": target_drop - prev_drop > epsilon_increment,
        "y_flip": prev_correct and not target_correct,
        "y_margin_drop": prev_margin - target_margin > epsilon,
    }


def normalized_change(delta: float, interval_steps: int) -> float:
    """按有效更新区间长度归一化；零长度区间返回0。"""
    return delta / interval_steps if interval_steps > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs=2, required=True)
    ap.add_argument("--groups", type=Path, nargs=2, required=True)
    ap.add_argument("--metadata", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--epsilon-increment", type=float, default=0.02)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summaries = []
    for row_path, group_path, meta_path in zip(args.rows, args.groups, args.metadata):
        source_rows = load_rows(row_path)
        seed = source_rows[0]["seed"]
        groups = load_groups(group_path)
        meta = load_meta(meta_path)
        by_anchor: dict[int, list[dict]] = {}
        for r in source_rows:
            by_anchor.setdefault(r["anchor"], []).append(r)
        seed_rows: list[dict] = []
        for anchor, values in by_anchor.items():
            values.sort(key=lambda x: x["update"])
            group = groups[anchor]
            negs = [int(x) for x in group.get("negatives", [])]
            poss = [int(x) for x in group.get("positives", [])]
            negative_pids = sorted({meta[n]["pid"] for n in negs if n in meta})
            exact_single_pair = (
                len(poss) == 1 and len(negs) == 1 and
                anchor in meta and negs[0] in meta and
                meta[anchor]["pid"] != meta[negs[0]]["pid"]
            )
            for i in range(1, len(values)):
                prev, target = values[i - 1], values[i]
                prevprev = values[i - 2] if i >= 2 else None
                steps = target["update"] - prev["update"]
                prev_drop, target_drop = prev["advantage_drop"], target["advantage_drop"]
                delta_drop = target_drop - prev_drop
                prev_m, target_m = prev["current_advantage"], target["current_advantage"]
                margin_drop = prev_m - target_m
                prev_correct, target_correct = prev_m > 0, target_m > 0
                labels = classify_transition(
                    prev_m, target_m, prev_drop, target_drop,
                    args.epsilon, args.epsilon_increment,
                )
                row = {
                    "seed": seed, "anchor": anchor, "pid": prev["pid"],
                    "positive_indices": ";".join(map(str, poss)),
                    "negative_indices": ";".join(map(str, negs)),
                    "negative_pids": ";".join(map(str, negative_pids)),
                    "negative_pid_count": len(negative_pids),
                    "negative_pid_decomposable": exact_single_pair,
                    "image_path": meta.get(anchor, {}).get("image_path", ""),
                    "clothes": meta.get(anchor, {}).get("clothes", ""),
                    "camid": meta.get(anchor, {}).get("camid", ""),
                    "feature_update": prev["update"], "target_update": target["update"],
                    "interval_steps": steps,
                    "reference_margin": prev["reference_advantage"],
                    "prev_margin": prev_m, "target_margin": target_m,
                    "prev_drop": prev_drop, "target_drop": target_drop,
                    "delta_drop": delta_drop, "delta_drop_per_step": normalized_change(delta_drop, steps),
                    "prev_drop_slope": (prev_drop - prevprev["advantage_drop"]) if prevprev else 0.0,
                    "slope_history_available": prevprev is not None,
                    **labels,
                    "new_negative_pid_count": 1 if exact_single_pair and prev_m > 0 and target_m <= 0 else (0 if exact_single_pair else ""),
                    "negative_pid_state": (
                        "new_error_pid" if exact_single_pair and prev_m > 0 and target_m <= 0 else
                        "preexisting_error_pid" if exact_single_pair and prev_m <= 0 and target_m <= 0 else
                        "recovered_pid" if exact_single_pair and prev_m <= 0 and target_m > 0 else
                        "correct_pid" if exact_single_pair else
                        "aggregate_not_decomposable"
                    ),
                    "relation_sha256": target["relation_sha256"],
                }
                # 风险分数只能使用 target 之前的信息。
                row["risk_prev_drop"] = prev_drop
                row["risk_low_prev_margin"] = -prev_m
                row["risk_recent_slope"] = row["prev_drop_slope"]
                seed_rows.append(row); all_rows.append(row)
        seed_fields = list(seed_rows[0]) if seed_rows else []
        with (args.output_dir / f"future_boundary_seed{seed}.csv").open("w", newline="", encoding="utf-8") as seed_fh:
            seed_writer = csv.DictWriter(seed_fh, fieldnames=seed_fields)
            seed_writer.writeheader(); seed_writer.writerows(seed_rows)
        interval_summary = []
        for interval in sorted({(r["feature_update"], r["target_update"]) for r in seed_rows}):
            ir = [r for r in seed_rows if (r["feature_update"], r["target_update"]) == interval]
            interval_summary.append({
                "feature_update": interval[0], "target_update": interval[1], "interval_steps": interval[1] - interval[0],
                "rows": len(ir),
                "persistent_rate": float(np.mean([r["y_persistent"] for r in ir])),
                "onset_rate": float(np.mean([r["y_onset"] for r in ir])),
                "increment_rate": float(np.mean([r["y_increment"] for r in ir])),
                "flip_rate": float(np.mean([r["y_flip"] for r in ir])),
                "margin_drop_rate": float(np.mean([r["y_margin_drop"] for r in ir])),
                "mean_delta_drop": float(np.mean([r["delta_drop"] for r in ir])),
                "mean_delta_drop_per_step": float(np.mean([r["delta_drop_per_step"] for r in ir])),
            })
        metrics = {
            "persistent_all": metric_block(seed_rows, "y_persistent", ["risk_prev_drop", "risk_low_prev_margin", "risk_recent_slope"]),
            "onset_prev_unactive_only": metric_block([r for r in seed_rows if r["prev_drop"] <= args.epsilon], "y_onset", ["risk_prev_drop", "risk_low_prev_margin", "risk_recent_slope"]),
            "increment_all": metric_block(seed_rows, "y_increment", ["risk_prev_drop", "risk_low_prev_margin", "risk_recent_slope"]),
            "increment_excluding_0_to_20": metric_block([r for r in seed_rows if r["feature_update"] > 0], "y_increment", ["risk_prev_drop", "risk_low_prev_margin", "risk_recent_slope"]),
            "flip_prev_correct_only": metric_block([r for r in seed_rows if r["prev_correct"]], "y_flip", ["risk_prev_drop", "risk_low_prev_margin", "risk_recent_slope"]),
            "margin_drop_all": metric_block(seed_rows, "y_margin_drop", ["risk_prev_drop", "risk_low_prev_margin", "risk_recent_slope"]),
        }
        summaries.append({
            "seed": seed, "rows": len(seed_rows), "anchor_count": len(by_anchor),
            "pid_count": len({r["pid"] for r in seed_rows}),
            "state_counts": {s: sum(r["state"] == s for r in seed_rows) for s in ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")},
            "intervals": interval_summary, "metrics": metrics,
            "groups_sha256": json.loads(group_path.read_text(encoding="utf-8"))["groups_sha256"],
            "metadata_source": str(meta_path),
        })
    fields = list(all_rows[0]) if all_rows else []
    with (args.output_dir / "future_boundary_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    (args.output_dir / "future_boundary_summary.json").write_text(json.dumps({
        "status": "PASS_WITH_LIMITED_COVERAGE",
        "epsilon": args.epsilon, "epsilon_increment": args.epsilon_increment,
        "definitions": {
            "persistent": "D_target>epsilon",
            "onset": "D_prev<=epsilon and D_target>epsilon",
            "increment": "D_target-D_prev>epsilon_increment",
            "flip": "M_prev>0 and M_target<=0",
            "margin_drop": "M_prev-M_target>epsilon",
        },
        "future_leakage_control": "risk columns use previous node; target node only supplies labels",
        "optimizer_step_executed": False, "official_query_gallery_used": False,
        "coverage_warning": "64 fixed groups per seed; not full 9576-image population",
        "rows": summaries,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_WITH_LIMITED_COVERAGE", "rows": len(all_rows), "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
