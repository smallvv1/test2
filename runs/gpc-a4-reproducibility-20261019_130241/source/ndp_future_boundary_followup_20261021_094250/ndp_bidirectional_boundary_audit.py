#!/usr/bin/env python3
"""固定 continuation 关系上的双侧局部身份边界审计。

只保留存在显式反向 image direction 的关系。A->B 与 B->A 是两个有向
margin，不在去重时合并；无向 image-pair/identity-pair 仅作为分组键。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("seed", "anchor", "feature_update", "target_update", "pid"):
                r[k] = int(r[k])
            for k in ("prev_margin", "target_margin", "delta_drop"):
                r[k] = float(r[k])
            r["y_flip"] = r["y_flip"].lower() == "true"
            out.append(r)
    return out


def reciprocal_pairs(groups: dict[int, dict]) -> tuple[list[tuple[int, int, dict, dict]], list[dict]]:
    """返回合法单负互反图像对；多负关系进入排除审计。"""
    pairs = []
    excluded = []
    seen = set()
    for anchor, group_a in groups.items():
        negatives_a = [int(x) for x in group_a.get("negatives", [])]
        for other in negatives_a:
            group_b = groups.get(other)
            if group_b is None or anchor not in {int(x) for x in group_b.get("negatives", [])}:
                continue
            key = tuple(sorted((anchor, other)))
            if key in seen:
                continue
            seen.add(key)
            negatives_b = [int(x) for x in group_b.get("negatives", [])]
            if len(negatives_a) != 1 or len(negatives_b) != 1 or negatives_a[0] != other or negatives_b[0] != anchor:
                excluded.append({
                    "anchor_a": anchor, "anchor_b": other,
                    "negative_count_a": len(negatives_a), "negative_count_b": len(negatives_b),
                    "reason": "非单负互反关系，聚合 margin 不能等同于该 image pair 的 margin",
                })
                continue
            if int(group_a["pid"]) == int(group_b["pid"]):
                excluded.append({"anchor_a": anchor, "anchor_b": other, "reason": "两端 PID 相同"})
                continue
            if not group_a.get("positives") or not group_b.get("positives"):
                excluded.append({"anchor_a": anchor, "anchor_b": other, "reason": "缺少合法同 PID 正样本"})
                continue
            pairs.append((anchor, other, group_a, group_b))
    return pairs, excluded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs=2, required=True)
    ap.add_argument("--groups", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    all_out = []; summaries = []; exclusions = []
    for row_path, group_path in zip(args.rows, args.groups):
        rows = read_rows(row_path); seed = rows[0]["seed"]
        groups = {int(g["anchor"]): g for g in json.loads(group_path.read_text(encoding="utf-8"))["groups"]}
        by_key = {(r["anchor"], r["feature_update"], r["target_update"]): r for r in rows}
        directed, seed_exclusions = reciprocal_pairs(groups)
        exclusions.extend([{"seed": seed, **x} for x in seed_exclusions])
        interval_set = sorted({(r["feature_update"], r["target_update"]) for r in rows})
        for anchor, neg, g_a, g_b in directed:
            pid_a, pid_b = int(g_a["pid"]), int(g_b["pid"])
            identity_pair = f"{min(pid_a,pid_b)}-{max(pid_a,pid_b)}"
            image_pair = f"{min(anchor,neg)}-{max(anchor,neg)}"
            for feature_update, target_update in interval_set:
                # 关键修复：a/b 是同一条 feature_update→target_update 记录。
                # 禁止从下一条 transition 读取 target_margin，避免 0→20 被错配成 M50-M20。
                a = by_key.get((anchor, feature_update, target_update))
                b = by_key.get((neg, feature_update, target_update))
                if not all((a, b)):
                    continue
                da = float(a["target_margin"] - a["prev_margin"])
                db = float(b["target_margin"] - b["prev_margin"])
                all_out.append({
                    "seed": seed, "feature_update": feature_update, "target_update": target_update,
                    "interval_steps": target_update - feature_update,
                    "identity_pair": identity_pair, "image_pair": image_pair,
                    "anchor_a": anchor, "anchor_b": neg, "pid_a": pid_a, "pid_b": pid_b,
                    "a_positive_indices": ";".join(map(str, g_a.get("positives", []))),
                    "b_positive_indices": ";".join(map(str, g_b.get("positives", []))),
                    "a_negative_indices": ";".join(map(str, g_a.get("negatives", []))),
                    "b_negative_indices": ";".join(map(str, g_b.get("negatives", []))),
                    "m_a_prev": a["prev_margin"], "m_a_target": a["target_margin"],
                    "m_b_prev": b["prev_margin"], "m_b_target": b["target_margin"],
                    "delta_a_to_b": da, "delta_b_to_a": db,
                    "a_flip": bool(a["prev_margin"] > 0 and a["target_margin"] <= 0),
                    "b_flip": bool(b["prev_margin"] > 0 and b["target_margin"] <= 0),
                    "a_improves_b_worsens": bool(da > 0 and db < 0),
                    "a_worsens_b_improves": bool(da < 0 and db > 0),
                    "both_improve": bool(da > 0 and db > 0),
                    "both_worsen": bool(da < 0 and db < 0),
                })
        seed_out = [r for r in all_out if r["seed"] == seed]
        seed_fields = list(seed_out[0]) if seed_out else []
        with (args.output_dir / f"bidirectional_boundary_seed{seed}.csv").open("w", newline="", encoding="utf-8") as seed_fh:
            seed_writer = csv.DictWriter(seed_fh, fieldnames=seed_fields)
            seed_writer.writeheader(); seed_writer.writerows(seed_out)
        summaries.append({
            "seed": seed, "directed_relation_pairs": len(directed), "transition_rows": len(seed_out),
            "identity_pair_count": len({r["identity_pair"] for r in seed_out}),
            "image_pair_count": len({r["image_pair"] for r in seed_out}),
            "one_side_a_improves_b_worsens": sum(r["a_improves_b_worsens"] for r in seed_out),
            "one_side_a_worsens_b_improves": sum(r["a_worsens_b_improves"] for r in seed_out),
            "both_improve": sum(r["both_improve"] for r in seed_out),
            "both_worsen": sum(r["both_worsen"] for r in seed_out),
            "a_flip": sum(r["a_flip"] for r in seed_out), "b_flip": sum(r["b_flip"] for r in seed_out),
        })
    fields = list(all_out[0]) if all_out else []
    with (args.output_dir / "bidirectional_boundary_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(all_out)
    (args.output_dir / "bidirectional_boundary_summary.json").write_text(json.dumps({
        "status": "PASS_WITH_LIMITED_COVERAGE", "rows": summaries,
        "deduplication": "directed image pair retained; identity_pair and image_pair are grouping keys",
        "time_alignment": "每一行直接使用同一 transition 的 prev_margin 与 target_margin",
        "excluded_relations": exclusions,
        "official_query_gallery_used": False, "optimizer_step_executed": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_WITH_LIMITED_COVERAGE", "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
