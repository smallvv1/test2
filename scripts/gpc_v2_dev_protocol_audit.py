#!/usr/local/bin/python3.10
"""GPC-v2 LTCC 开发验证协议审计与扩展清单生成。

本脚本只读取 LTCC 原始 train split、冻结开发划分和已有 CSV，不读取模型特征，
也不启动训练。它严格沿用历史 LTCC loader 的文件名语义：
PID_second-field_cCAM，其中 second-field 是 clothing，cCAM 是 camera。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/DIFFER")
PATH_RE = re.compile(r"^(?P<pid>\d+)_(?P<clothes>\d+)_c(?P<cam>\d+)_")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_manifest(paths: list[str]) -> str:
    text = "\n".join(paths) + "\n"
    return sha256_bytes(text.encode("utf-8"))


def parse_path(path: str) -> dict[str, int]:
    match = PATH_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"无法解析 LTCC 文件名：{path}")
    return {
        "raw_pid": int(match.group("pid")),
        "clothes": int(match.group("clothes")),
        "camera": int(match.group("cam")),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row["raw_pid"]), int(row["clothes"]), int(row["camera"]), row["image_path"]))


def legal_positive_count(query: dict[str, Any], gallery: list[dict[str, Any]]) -> int:
    return sum(
        int(item["raw_pid"]) == int(query["raw_pid"])
        and int(item["clothes"]) != int(query["clothes"])
        and int(item["camera"]) != int(query["camera"])
        for item in gallery
    )


def make_protocol(frozen_split: Path, split_csv: Path, output: Path, run_time: str) -> dict[str, Any]:
    frozen = json.loads(frozen_split.read_text(encoding="utf-8"))
    rows = read_csv(split_csv)
    validation = []
    training = []
    frozen_query_paths = set(frozen["validation_query"])
    frozen_gallery_paths = set(frozen["validation_gallery"])
    for row in rows:
        meta = parse_path(row["image_path"])
        item = {
            "image_path": row["image_path"],
            "pid": int(row["pid"]),
            "raw_pid": meta["raw_pid"],
            "clothes": meta["clothes"],
            "camera": meta["camera"],
            "source_clothes_id": int(row["clothes_id"]),
            "source_camid": int(row["camid"]),
            "split": row["split"],
        }
        (validation if row["split"] == "dev_validation" else training).append(item)
    validation = sort_rows(validation)
    training = sort_rows(training)
    by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in validation:
        by_pid[int(item["raw_pid"])].append(item)

    gallery: list[dict[str, Any]] = []
    pid_summary: list[dict[str, Any]] = []
    all_min_clothes_queries: list[dict[str, Any]] = []
    eligible_min_clothes_queries: list[dict[str, Any]] = []
    dropped_no_positive: list[dict[str, Any]] = []
    for raw_pid in sorted(by_pid):
        items = by_pid[raw_pid]
        clothes = sorted({int(item["clothes"]) for item in items})
        min_clothes = clothes[0]
        pid_gallery = [item for item in items if int(item["clothes"]) != min_clothes]
        gallery.extend(pid_gallery)
        min_queries = [item for item in items if int(item["clothes"]) == min_clothes]
        all_min_clothes_queries.extend(min_queries)
        pid_eligible = len(clothes) >= 2
        eligible_count = 0
        dropped_count = 0
        if pid_eligible:
            for query in min_queries:
                positives = legal_positive_count(query, gallery=pid_gallery)
                query = {**query, "min_query_clothes": min_clothes, "legal_positive_count": positives}
                if positives > 0:
                    eligible_min_clothes_queries.append(query)
                    eligible_count += 1
                else:
                    dropped_no_positive.append(query)
                    dropped_count += 1
        pid_summary.append(
            {
                "raw_pid": raw_pid,
                "pid": int(items[0]["pid"]),
                "validation_image_count": len(items),
                "clothes": clothes,
                "min_query_clothes": min_clothes,
                "min_clothes_image_count": len(min_queries),
                "cross_clothes_pid": pid_eligible,
                "expanded_query_count": eligible_count,
                "dropped_no_legal_positive_count": dropped_count,
                "gallery_count": len(pid_gallery),
            }
        )

    gallery = sort_rows(gallery)
    eligible_min_clothes_queries = sort_rows(eligible_min_clothes_queries)
    all_min_clothes_queries = sort_rows(all_min_clothes_queries)
    frozen_old_query_items = [item for item in validation if item["image_path"] in frozen_query_paths]
    old_query_legal = [item for item in frozen_old_query_items if legal_positive_count(item, gallery) > 0]

    query_paths = [item["image_path"] for item in eligible_min_clothes_queries]
    gallery_paths = [item["image_path"] for item in gallery]
    train_paths = [item["image_path"] for item in training]
    validation_paths = [item["image_path"] for item in validation]
    query_set = set(query_paths)
    gallery_set = set(gallery_paths)
    train_set = set(train_paths)
    validation_pid_set = {int(item["raw_pid"]) for item in validation}
    train_pid_set = {int(item["raw_pid"]) for item in training}

    checks = {
        "frozen_split_file_present": frozen_split.is_file(),
        "query_gallery_path_disjoint": not bool(query_set & gallery_set),
        "query_not_in_train": not bool(query_set & train_set),
        "gallery_not_in_train": not bool(gallery_set & train_set),
        "train_validation_pid_disjoint": not bool(train_pid_set & validation_pid_set),
        "all_expanded_queries_have_legal_positive": all(int(item["legal_positive_count"]) > 0 for item in eligible_min_clothes_queries),
        "no_same_image_match": not bool(query_set & gallery_set),
        "gallery_uses_only_validation_pool": gallery_set.issubset(set(validation_paths)),
        "query_uses_only_validation_pool": query_set.issubset(set(validation_paths)),
        "official_query_gallery_not_used": not any("/query/" in path or "/test/" in path for path in query_paths + gallery_paths),
    }
    # The frozen JSON stores component manifest hashes rather than a split hash;
    # the exact file hash is recorded separately and is checked by the caller.
    def manifest_record(paths: list[str]) -> dict[str, Any]:
        return {"count": len(paths), "sha256": sha256_manifest(paths), "paths": paths}

    result = {
        "created_utc": now(),
        "run_time": run_time,
        "source": {
            "frozen_split": str(frozen_split),
            "frozen_split_sha256": sha256_bytes(frozen_split.read_bytes()),
            "development_split_csv": str(split_csv),
            "development_split_csv_sha256": sha256_bytes(split_csv.read_bytes()),
            "source_dataset": "LTCC 原始 train split only",
        },
        "field_semantics": {
            "filename_regex": r"(PID)_(CLOTHING)_c(CAMERA)_...",
            "clothing": "文件名第二字段；历史 loader 的 pattern2 = (\\w+)_c",
            "camera": "文件名 c 后字段；历史 loader 的 pattern1 第三字段",
            "metric": "LTCC CC：过滤同 PID 同 camera 与同 PID 同 clothing",
        },
        "frozen_partition": {
            "train_pid_count": len(train_pid_set),
            "validation_pid_count": len(validation_pid_set),
            "train_image_count": len(training),
            "validation_image_count": len(validation),
            "train_raw_pids": sorted(train_pid_set),
            "validation_raw_pids": sorted(validation_pid_set),
        },
        "old_protocol": {
            "frozen_query_count": len(frozen_query_paths),
            "old_legal_query_count": len(old_query_legal),
            "old_invalid_query_count": len(frozen_old_query_items) - len(old_query_legal),
            "old_gallery_count": len(gallery_paths),
            "invalid_old_query_raw_pids": sorted({int(item["raw_pid"]) for item in frozen_old_query_items if item not in old_query_legal}),
            "old_query_rule": frozen.get("rule"),
            "historical_evaluation_observed": {"query_count": 11, "gallery_count": 1256},
        },
        "expanded_protocol": {
            "query_rule": "固定验证 PID 内，取每个有至少两种衣物状态的 PID 的最小衣物状态全部图像；剔除在 LTCC 摄像头规则下无合法跨衣正样本的 Query。",
            "gallery_rule": "固定验证 PID 内每个 PID 的非最小衣物状态全部图像；不含 Query。",
            "query_count_before_legal_filter": len([item for item in all_min_clothes_queries if len({int(x["clothes"]) for x in by_pid[int(item["raw_pid"])]}) >= 2]),
            "query_count": len(query_paths),
            "gallery_count": len(gallery_paths),
            "effective_pid_count": len({int(item["raw_pid"]) for item in eligible_min_clothes_queries}),
            "dropped_no_legal_positive_count": len(dropped_no_positive),
            "dropped_no_legal_positive_paths": [item["image_path"] for item in dropped_no_positive],
            "pid_summary": pid_summary,
            "query_manifest": manifest_record(query_paths),
            "gallery_manifest": manifest_record(gallery_paths),
            "train_manifest": manifest_record(train_paths),
            "validation_manifest": manifest_record(validation_paths),
        },
        "checks": checks,
        "status": "PASS" if all(bool(value) for value in checks.values()) else "FAIL",
    }

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "expanded_query.csv", eligible_min_clothes_queries, ["image_path", "pid", "raw_pid", "clothes", "camera", "legal_positive_count"])
    write_csv(output / "expanded_gallery.csv", gallery, ["image_path", "pid", "raw_pid", "clothes", "camera"])
    write_csv(output / "validation_pid_summary.csv", pid_summary, list(pid_summary[0]))
    write_csv(output / "dropped_no_legal_positive.csv", dropped_no_positive, ["image_path", "pid", "raw_pid", "clothes", "camera", "legal_positive_count"])
    (output / "expanded_dev_protocol.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--frozen-split", default=str(ROOT / "gpc_v2/ltcc_dev_split_20261015_145338.json"))
    parser.add_argument("--split-csv", default=str(ROOT / "outputs/gpc_v2_ltcc_20261015_155822/development_split.csv"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = make_protocol(Path(args.frozen_split), Path(args.split_csv), Path(args.output), args.run_time)
    print(json.dumps({"status": result["status"], "output": args.output, "old": result["old_protocol"], "expanded": {key: value for key, value in result["expanded_protocol"].items() if key in ("query_count_before_legal_filter", "query_count", "gallery_count", "effective_pid_count", "dropped_no_legal_positive_count")}}, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
