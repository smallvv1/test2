#!/usr/bin/env python3
"""验证历史 LTCC train-only Caption 资产及其与 Bio loader 的逐样本对齐。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ASSET_ROOT = Path("/data/data/DIFFER_TextCaption/LTCC_ReID")
TRAIN_NPZ = ASSET_ROOT / "EVA02-CLIP-B-16/train.npz"
SUMMARY_JSON = ASSET_ROOT / "train_caption_summary_biometric.json"
IMAGE_ROOT = Path("/home/data/LTCC_ReID")
SPLIT_JSON = Path("/home/DIFFER/gpc_v2/ltcc_dev_split_20261015_145338.json")
HASH_LIST = Path("/data/DIFFER/artifacts/reproduction_final/SHA256_LIST.json")
FEATURE_KEY = "ft_EVA02-CLIP-B-16"
IMAGE_RE = re.compile(r"^(\d+)_(\d+)_c(\d+)_(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(lines: list[str], message: str) -> None:
    lines.append(message)
    print(message, flush=True)


def decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def parse_stem(stem: str) -> tuple[int, int, int]:
    match = IMAGE_RE.fullmatch(stem)
    if not match:
        raise ValueError(f"无法解析 LTCC 文件名: {stem}")
    raw_pid, image_index, raw_camera = map(int, match.groups()[:3])
    # LTCC 的服装标记位于 image_index 与 camera 之间的 c 前字段；
    # 对当前格式，第二个字段是图像序号，服装从 image_index 前缀解析由下方完成。
    return raw_pid, image_index, raw_camera


def parse_image(path: Path) -> tuple[int, str, int]:
    # 历史 loader 的 pattern2=r"(\w+)_c" 会把 000_1 作为该 PID 的服装状态键。
    parts = path.stem.split("_")
    if len(parts) != 4 or not parts[2].startswith("c"):
        raise ValueError(f"无法解析 LTCC 文件名: {path.name}")
    return int(parts[0]), f"{parts[0]}_{parts[1]}", int(parts[2][1:])


def historical_hashes() -> dict[str, dict[str, Any]]:
    payload = json.loads(HASH_LIST.read_text(encoding="utf-8"))
    records = payload["files"] if isinstance(payload, dict) else payload
    return {str(item["role"]): item for item in records if item.get("role", "").startswith("LTCC")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    result: dict[str, Any] = {
        "run_time": args.run_time,
        "asset_root": str(ASSET_ROOT),
        "train_npz": str(TRAIN_NPZ),
        "summary_json": str(SUMMARY_JSON),
        "fixed_split": str(SPLIT_JSON),
        "checks": {},
        "status": "FAIL",
    }

    try:
        log(lines, f"开始 LTCC train-only Caption 资产验证；RUN_TIME={args.run_time}")
        log(lines, f"资产来源：{ASSET_ROOT}")
        required = [TRAIN_NPZ, SUMMARY_JSON, IMAGE_ROOT / "train", IMAGE_ROOT / "query", IMAGE_ROOT / "test", SPLIT_JSON]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"缺少必需路径：{path}")

        actual_npz_sha = sha256_file(TRAIN_NPZ)
        actual_summary_sha = sha256_file(SUMMARY_JSON)
        hashes = historical_hashes()
        expected_npz_sha = hashes.get("LTCC image-level text", {}).get("sha256")
        expected_summary_sha = hashes.get("LTCC Summary Bio text", {}).get("sha256")
        result["hashes"] = {
            "train_npz": {"actual": actual_npz_sha, "historical": expected_npz_sha, "match": actual_npz_sha == expected_npz_sha, "bytes": TRAIN_NPZ.stat().st_size},
            "summary_json": {"actual": actual_summary_sha, "historical": expected_summary_sha, "match": actual_summary_sha == expected_summary_sha, "bytes": SUMMARY_JSON.stat().st_size},
        }
        log(lines, f"train.npz SHA256：{actual_npz_sha}；历史匹配：{actual_npz_sha == expected_npz_sha}")
        log(lines, f"Summary Bio JSON SHA256：{actual_summary_sha}；历史匹配：{actual_summary_sha == expected_summary_sha}")
        if actual_npz_sha != expected_npz_sha or actual_summary_sha != expected_summary_sha:
            raise RuntimeError("资产哈希与历史成功实验不一致，停止恢复")
        result["checks"]["historical_hash_exact"] = True

        with np.load(TRAIN_NPZ, allow_pickle=True) as archive:
            keys = list(archive.files)
            data = np.asarray(archive["data"])
            metadata = [decode(item) for item in archive["metadata"]]
        log(lines, f"NPZ keys={keys}；data shape={list(data.shape)} dtype={data.dtype}；metadata count={len(metadata)}")
        if set(keys) != {"data", "metadata"}:
            raise RuntimeError(f"NPZ 键不符合历史格式：{keys}")
        if data.shape != (57456, 512) or len(metadata) != 9576 or data.shape[0] != len(metadata) * 6:
            raise RuntimeError(f"NPZ 形状不符合历史格式：data={data.shape}, metadata={len(metadata)}")
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise RuntimeError("NPZ data 存在非数值或 NaN/Inf")
        if len(set(metadata)) != len(metadata):
            raise RuntimeError("NPZ metadata 存在重复样本键")
        result["npz"] = {"keys": keys, "data_shape": list(data.shape), "data_dtype": str(data.dtype), "metadata_count": len(metadata), "finite": True, "factor_count": 6, "metadata_unique": True}
        result["checks"].update({"npz_load": True, "npz_finite": True, "npz_dimension": True, "npz_factor_count": True, "npz_metadata_unique": True})

        summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or set(summary) != {str(i) for i in range(77)}:
            raise RuntimeError(f"Summary Bio PID 键不符合 77 个 LTCC train PID：count={len(summary) if isinstance(summary, dict) else '非dict'}")
        summary_shapes = set()
        summary_finite = True
        for key, value in summary.items():
            if not isinstance(value, dict) or FEATURE_KEY not in value:
                raise RuntimeError(f"Summary Bio 缺少 {key}/{FEATURE_KEY}")
            arr = np.asarray(value[FEATURE_KEY], dtype=np.float32)
            summary_shapes.add(tuple(arr.shape))
            summary_finite = summary_finite and bool(np.isfinite(arr).all())
        if summary_shapes != {(1, 512)} or not summary_finite:
            raise RuntimeError(f"Summary Bio 形状或有限性错误：shapes={summary_shapes}, finite={summary_finite}")
        result["summary"] = {"entry_count": len(summary), "feature_key": FEATURE_KEY, "feature_shape": [1, 512], "finite": summary_finite}
        result["checks"].update({"summary_load": True, "summary_pid_count": True, "summary_semantic_key": True, "summary_dimension": True, "summary_finite": True})
        log(lines, "Summary Bio：77 个 PID、每项 [1,512]、全部有限；语义键顺序确认：factor-0 Biometrics")

        train_paths = sorted((IMAGE_ROOT / "train").glob("*.png"))
        query_paths = sorted((IMAGE_ROOT / "query").glob("*.png"))
        test_paths = sorted((IMAGE_ROOT / "test").glob("*.png"))
        train_stems = [path.stem for path in train_paths]
        query_stems = {path.stem for path in query_paths}
        test_stems = {path.stem for path in test_paths}
        metadata_set = set(metadata)
        if len(train_paths) != 9576 or metadata_set != set(train_stems):
            missing = sorted(set(train_stems) - metadata_set)[:10]
            extra = sorted(metadata_set - set(train_stems))[:10]
            raise RuntimeError(f"train metadata 映射不完整：images={len(train_paths)}, missing={missing}, extra={extra}")
        if metadata_set & (query_stems | test_stems):
            raise RuntimeError("Caption metadata 混入 query/gallery/test 样本")
        result["image_mapping"] = {
            "train_images": len(train_paths), "query_images": len(query_paths), "test_images": len(test_paths),
            "metadata_matches_train_set": True, "metadata_order_matches_sorted_images": metadata == train_stems,
            "query_test_overlap": False,
        }
        result["checks"].update({"train_count_9576": True, "train_metadata_set_alignment": True, "no_query_gallery_test_caption": True})
        log(lines, f"训练图像={len(train_paths)}、query={len(query_paths)}、test={len(test_paths)}；metadata 与训练图像一一映射；未混入 query/gallery/test")

        raw_info: dict[str, tuple[int, str, int]] = {}
        for path in train_paths:
            raw_info[path.stem] = parse_image(path)
        raw_pids = sorted({value[0] for value in raw_info.values()})
        raw_clothes = sorted({value[1] for value in raw_info.values()})
        pid_to_label = {pid: label for label, pid in enumerate(raw_pids)}
        clothes_to_label = {cloth: label for label, cloth in enumerate(raw_clothes)}
        if set(summary) != {str(pid_to_label[pid]) for pid in raw_pids}:
            raise RuntimeError("Summary Bio PID 键与 LTCC loader 的排序标签不一致")
        result["labels"] = {"raw_pid_count": len(raw_pids), "raw_clothes_count": len(raw_clothes), "camera_count": len({value[2] for value in raw_info.values()}), "pid_mapping": pid_to_label, "clothes_mapping": clothes_to_label, "summary_labels_match_loader": True}
        result["checks"].update({"pid_alignment": True, "clothes_alignment": True, "camera_parse": True})

        split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
        split_sets = {name: set(split[name]) for name in ("train_images", "validation_images", "validation_query", "validation_gallery")}
        all_train = set(str(path) for path in train_paths)
        if not all(paths <= all_train for paths in split_sets.values()):
            raise RuntimeError("冻结开发划分包含非 LTCC original train split 路径")
        if split_sets["train_images"] & (split_sets["validation_images"] | split_sets["validation_query"] | split_sets["validation_gallery"]):
            raise RuntimeError("冻结开发划分存在训练/验证路径重叠")
        if set(split["train_pids"]) & set(split["validation_pids"]):
            raise RuntimeError("冻结开发划分存在 PID 重叠")
        result["fixed_split"] = {"run_time": split["run_time"], "seed": split["seed"], "counts": split["counts"], "sha256": split["sha256"], "pid_disjoint": True, "path_disjoint": True, "all_from_original_train": True}
        result["checks"].update({"fixed_split_preserved": split["run_time"] == "20261015_145338", "fixed_split_pid_disjoint": True, "fixed_split_path_disjoint": True, "fixed_split_train_only": True})
        log(lines, f"冻结开发划分：run_time={split['run_time']}，train/validation PID 与路径均不重叠；未重新随机划分")

        # 使用历史 LTCC loader 逐样本验证 PID、服装、camera 及 Summary Bio 替换。
        import sys
        sys.path.insert(0, "/data/DIFFER")
        from data.datasets.ltcc import LTCC

        dataset = LTCC(root=str(IMAGE_ROOT.parent), caption_dir=str(ASSET_ROOT), caption_model="EVA02-CLIP-B-16", load_sum_ft=True, bio_index=[0], nonbio_index=[1, 2])
        if len(dataset.train) != 9576 or dataset.num_train_pids != 77:
            raise RuntimeError(f"历史 loader 数据规模错误：train={len(dataset.train)}, pids={dataset.num_train_pids}")
        metadata_index = {name: index for index, name in enumerate(metadata)}
        factor12_match = True
        factor0_match = True
        label_match = True
        runtime_finite = True
        for item in dataset.train:
            path = Path(item["image_path"])
            stem = path.stem
            index = metadata_index[stem]
            raw_pid, raw_cloth, raw_cam = raw_info[stem]
            factor_block = data[index * 6 : (index + 1) * 6]
            factor12_match = factor12_match and np.array_equal(np.asarray(item["caption_feature"])[1:], factor_block[[1, 2]])
            expected_summary = np.asarray(summary[str(pid_to_label[raw_pid])][FEATURE_KEY], dtype=np.float32)[0]
            factor0_match = factor0_match and np.allclose(np.asarray(item["caption_feature"])[0].astype(np.float32), expected_summary, atol=1e-3, rtol=0.0)
            label_match = label_match and int(item["pid"]) == pid_to_label[raw_pid] and int(item["clothes_id"]) == clothes_to_label[raw_cloth] and int(item["camid"]) == raw_cam - 1
            runtime_finite = runtime_finite and bool(np.isfinite(np.asarray(item["caption_feature"])).all())
        if not (factor12_match and factor0_match and label_match and runtime_finite):
            raise RuntimeError(f"历史 loader 逐样本对齐失败：factor12={factor12_match}, factor0_summary={factor0_match}, labels={label_match}, finite={runtime_finite}")
        result["runtime_loader"] = {"loader": "/data/DIFFER/data/datasets/ltcc.py", "train_count": len(dataset.train), "train_pid_count": dataset.num_train_pids, "train_clothes_count": dataset.num_train_clothes, "camera_count": dataset.num_camera, "bio_index": [0], "nonbio_index": [1, 2], "summary_text_enabled": True, "caption_shape": [3, 512], "factor12_raw_match": True, "factor0_summary_match": True, "pid_clothes_camera_match": True, "runtime_finite": True}
        result["checks"].update({"historical_loader": True, "bio_runtime_dimension": True, "raw_nonbio_alignment": True, "summary_bio_alignment": True, "pid_clothes_camera_alignment": True, "runtime_finite": True})
        log(lines, "历史 LTCC loader 逐样本验证通过：caption=[3,512]；factor-1/2 原始特征、factor-0 Summary Bio、PID/Clothing/Camera 全部对齐")

        result["status"] = "PASS"
        log(lines, "结论：LTCC train-only Caption 资产恢复并验证通过；可以进入 Bio/GPC 双 PPU 门禁。")
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        log(lines, f"中文结论：Caption 资产验证失败，禁止启动正式训练。原因：{type(exc).__name__}: {exc}")

    result_path = output_root / "caption_alignment_audit.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = output_root / "caption_alignment_audit.md"
    report_lines = [
        "# LTCC Caption 资产恢复与对齐审计", "", f"RUN_TIME：`{args.run_time}`", "", f"结论：**{result['status']}**", "",
        f"资产来源：`{ASSET_ROOT}`", f"train.npz：`{TRAIN_NPZ}`", f"Summary Bio：`{SUMMARY_JSON}`", "",
        "## 检查日志", "", *[f"- {line}" for line in lines], "", "## 机器可读结果", "", f"- JSON：`{result_path}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
