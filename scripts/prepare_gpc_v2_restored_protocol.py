#!/usr/local/bin/python3
"""为恢复运行生成固定开发划分的只读派生清单。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_name(path: str) -> tuple[int, str, int]:
    name = Path(path).name
    match = re.search(r"(\d+)_(\d+)_c(\d+)", name)
    if not match:
        raise RuntimeError(f"无法解析 LTCC 文件名：{path}")
    pid, _unused, camera = map(int, match.groups())
    cloth = re.search(r"(\w+)_c", name).group(1)
    return pid, cloth, camera - 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-split", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.frozen_split.read_text(encoding="utf-8"))
    train_images = list(frozen["train_images"])
    validation_images = set(frozen["validation_images"])
    if set(train_images) & validation_images:
        raise RuntimeError("冻结开发划分存在训练/验证路径重叠")
    if len(train_images) != frozen["counts"]["dev_train"] or len(validation_images) != frozen["counts"]["dev_val_pool"]:
        raise RuntimeError("冻结开发划分计数不一致")

    all_paths = train_images + sorted(validation_images)
    cloth_tokens = sorted({parse_name(path)[1] for path in all_paths})
    cloth_to_label = {value: index for index, value in enumerate(cloth_tokens)}
    raw_pid_values = sorted({parse_name(path)[0] for path in all_paths})
    raw_pid_to_label = {raw_pid: label for label, raw_pid in enumerate(raw_pid_values)}
    validation_raw_pids = set(map(int, frozen["validation_pids"]))
    rows = []
    for original_index, path in enumerate(all_paths):
        raw_pid, cloth, camera = parse_name(path)
        pid = raw_pid_to_label[raw_pid]
        split = "dev_validation" if path in validation_images else "development_train"
        expected = "dev_validation" if raw_pid in validation_raw_pids else "development_train"
        if split != expected:
            raise RuntimeError(f"PID 与冻结划分不一致：{path}")
        rows.append({
            "original_train_index": original_index,
            "split": split,
            "pid": pid,
            "raw_pid": raw_pid,
            "clothes_id": cloth_to_label[cloth],
            "camid": camera,
            "image_path": path,
        })
    rows.sort(key=lambda row: row["image_path"])
    csv_path = output / "development_split.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    frozen_copy = output / f"frozen_dev_split_{frozen['run_time']}.json"
    shutil.copy2(args.frozen_split, frozen_copy)
    train_rows = [row for row in rows if row["split"] == "development_train"]
    val_rows = [row for row in rows if row["split"] == "dev_validation"]
    manifest = {
        "phase": "DIFFER+GPC-v2 LTCC restored run",
        "run_time": args.run_time,
        "source_dataset": "LTCC original train split only",
        "frozen_split_source": str(args.frozen_split),
        "frozen_split_sha256": sha256_file(args.frozen_split),
        "frozen_split_run_time": frozen["run_time"],
        "frozen_split_seed": frozen["seed"],
        "raw_pid_to_historical_loader_label": raw_pid_to_label,
        "train_raw_pids": sorted({parse_name(row["image_path"])[0] for row in train_rows}),
        "validation_raw_pids": sorted({parse_name(row["image_path"])[0] for row in val_rows}),
        "train_pids": sorted({row["pid"] for row in train_rows}),
        "validation_pids": sorted({row["pid"] for row in val_rows}),
        "train_pid_count": len({row["pid"] for row in train_rows}),
        "validation_pid_count": len({row["pid"] for row in val_rows}),
        "train_image_count": len(train_rows),
        "validation_image_count": len(val_rows),
        "train_validation_pid_disjoint": not (set(row["pid"] for row in train_rows) & set(row["pid"] for row in val_rows)),
        "train_validation_path_disjoint": not (set(row["image_path"] for row in train_rows) & set(row["image_path"] for row in val_rows)),
        "query_gallery_used_for_training": False,
        "query_gallery_used_for_split": False,
        "official_query_gallery_used": False,
        "development_split_csv": str(csv_path),
        "development_split_csv_sha256": sha256_file(csv_path),
        "frozen_manifest_sha256": frozen["sha256"],
        "counts": frozen["counts"],
        "validation_rule": frozen["rule"],
        "clothing_label_source": "historical LTCC loader pattern2 sorted labels; labels are for validation metadata only",
        "pid_label_source": "historical LTCC loader pattern1 sorted raw PID labels; raw PID retained in raw_pid",
    }
    (output / "development_split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    record = {
        "run_time": args.run_time,
        "status": "PASS",
        "frozen_split": str(args.frozen_split),
        "frozen_split_sha256": sha256_file(args.frozen_split),
        "derived_manifest": str(output / "development_split_manifest.json"),
        "derived_csv": str(csv_path),
        "derived_csv_sha256": sha256_file(csv_path),
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "train_pids": len(manifest["train_pids"]),
        "validation_pids": len(manifest["validation_pids"]),
        "pid_disjoint": manifest["train_validation_pid_disjoint"],
        "path_disjoint": manifest["train_validation_path_disjoint"],
        "official_query_gallery_used": False,
    }
    (output / "restored_protocol_manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
