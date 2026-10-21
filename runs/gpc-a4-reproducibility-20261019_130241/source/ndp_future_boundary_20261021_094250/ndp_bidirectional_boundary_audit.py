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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs=2, required=True)
    ap.add_argument("--groups", type=Path, nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    all_out = []; summaries = []
    for row_path, group_path in zip(args.rows, args.groups):
        rows = read_rows(row_path); seed = rows[0]["seed"]
        groups = {int(g["anchor"]): g for g in json.loads(group_path.read_text(encoding="utf-8"))["groups"]}
        by_key = {(r["anchor"], r["feature_update"]): r for r in rows}
        directed = []
        seen = set()
        for anchor, g in groups.items():
            for neg in g.get("negatives", []):
                neg = int(neg)
                other = groups.get(neg)
                if other is None or anchor not in {int(x) for x in other.get("negatives", [])}:
                    continue
                key = tuple(sorted(((anchor, neg), (neg, anchor))))
                if key in seen:
                    continue
                seen.add(key)
                directed.append((anchor, neg, g, other))
        interval_set = sorted({(r["feature_update"], r["target_update"]) for r in rows})
        for anchor, neg, g_a, g_b in directed:
            pid_a, pid_b = int(g_a["pid"]), int(g_b["pid"])
            identity_pair = f"{min(pid_a,pid_b)}-{max(pid_a,pid_b)}"
            image_pair = f"{min(anchor,neg)}-{max(anchor,neg)}"
            for feature_update, target_update in interval_set:
                a = by_key.get((anchor, feature_update)); at = by_key.get((anchor, target_update))
                b = by_key.get((neg, feature_update)); bt = by_key.get((neg, target_update))
                if not all((a, at, b, bt)):
                    continue
                da = float(at["target_margin"] - a["target_margin"])
                db = float(bt["target_margin"] - b["target_margin"])
                all_out.append({
                    "seed": seed, "feature_update": feature_update, "target_update": target_update,
                    "interval_steps": target_update - feature_update,
                    "identity_pair": identity_pair, "image_pair": image_pair,
                    "anchor_a": anchor, "anchor_b": neg, "pid_a": pid_a, "pid_b": pid_b,
                    "m_a_prev": a["prev_margin"], "m_a_target": at["target_margin"],
                    "m_b_prev": b["prev_margin"], "m_b_target": bt["target_margin"],
                    "delta_a_to_b": da, "delta_b_to_a": db,
                    "a_flip": bool(a["prev_margin"] > 0 and at["target_margin"] <= 0),
                    "b_flip": bool(b["prev_margin"] > 0 and bt["target_margin"] <= 0),
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
        "official_query_gallery_used": False, "optimizer_step_executed": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_WITH_LIMITED_COVERAGE", "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
