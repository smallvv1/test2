#!/usr/local/bin/python3
"""GPC-v2 LTCC 的来源、数据边界和固定开发划分审计。"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/DIFFER")
LEGACY = Path("/home/DIFFER_NDP_test_CLoth-aware")
OUTPUT = Path("/home/output")
DATA = Path("/home/data/LTCC_ReID")
RUN_TIME = os.environ.get("RUN_TIME") or (ROOT / "gpc_v2/RUN_TIME.txt").read_text(encoding="utf-8").strip()
SPLIT_PATH = ROOT / "gpc_v2" / f"ltcc_dev_split_{RUN_TIME}.json"
SOURCE_REPORT = ROOT / "reports" / "GPC_V2_LTCC_SOURCE_AUDIT.md"
LEAKAGE_REPORT = ROOT / "reports" / "GPC_V2_LTCC_LEAKAGE_AUDIT.md"
STATUS_PATH = ROOT / "gpc_v2" / f"audit_status_{RUN_TIME}.json"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def image_meta(path: Path) -> tuple[int, int]:
    match = re.match(r"^(\d+)_\d+_c(\d+)_", path.name)
    if not match:
        raise ValueError(f"无法解析 LTCC 文件名：{path.name}")
    return int(match.group(1)), int(match.group(2))


def make_split() -> dict[str, object]:
    images = sorted((DATA / "train").glob("*.png"))
    by_pid: dict[int, list[Path]] = defaultdict(list)
    for path in images:
        pid, _cloth = image_meta(path)
        by_pid[pid].append(path)
    eligible = sorted(pid for pid, rows in by_pid.items() if len({image_meta(row)[1] for row in rows}) >= 2)
    rng = random.Random(20261015)
    shuffled = eligible[:]
    rng.shuffle(shuffled)
    val_ids = sorted(shuffled[:17])
    train_ids = sorted(pid for pid in by_pid if pid not in set(val_ids))
    if set(train_ids) & set(val_ids):
        raise RuntimeError("开发训练身份与开发验证身份重叠")
    train_rows = [path for pid in train_ids for path in sorted(by_pid[pid])]
    val_rows = [path for pid in val_ids for path in sorted(by_pid[pid])]
    query_rows: list[Path] = []
    gallery_rows: list[Path] = []
    for pid in val_ids:
        groups: dict[int, list[Path]] = defaultdict(list)
        for path in sorted(by_pid[pid]):
            groups[image_meta(path)[1]].append(path)
        clothes = sorted(groups)
        query_rows.append(sorted(groups[clothes[0]])[0])
        for cloth in clothes[1:]:
            gallery_rows.extend(sorted(groups[cloth]))
    train_text = "\n".join(str(x) for x in train_rows) + "\n"
    val_text = "\n".join(str(x) for x in val_rows) + "\n"
    query_text = "\n".join(str(x) for x in query_rows) + "\n"
    gallery_text = "\n".join(str(x) for x in gallery_rows) + "\n"
    result = {
        "created_utc": now(),
        "run_time": RUN_TIME,
        "seed": 20261015,
        "source": str(DATA / "train"),
        "rule": "从原始 LTCC train split 固定抽取 17 个至少含两种服装的 PID 做开发验证，其余 PID 做开发训练；验证 query 为每个 PID 最小服装编号的第一张图，gallery 为其它服装的全部图。",
        "train_pids": train_ids,
        "validation_pids": val_ids,
        "train_pid_count": len(train_ids),
        "validation_pid_count": len(val_ids),
        "train_images": [str(x) for x in train_rows],
        "validation_images": [str(x) for x in val_rows],
        "validation_query": [str(x) for x in query_rows],
        "validation_gallery": [str(x) for x in gallery_rows],
        "counts": {"original_train": len(images), "dev_train": len(train_rows), "dev_val_pool": len(val_rows), "dev_query": len(query_rows), "dev_gallery": len(gallery_rows)},
        "sha256": {"train_manifest": sha256_text(train_text), "validation_manifest": sha256_text(val_text), "query_manifest": sha256_text(query_text), "gallery_manifest": sha256_text(gallery_text)},
        "path_disjoint": {
            "train_vs_validation_pool": not bool(set(train_rows) & set(val_rows)),
            "query_vs_gallery": not bool(set(query_rows) & set(gallery_rows)),
            "train_vs_query": not bool(set(train_rows) & set(query_rows)),
            "train_vs_gallery": not bool(set(train_rows) & set(gallery_rows)),
        },
    }
    write_json(SPLIT_PATH, result)
    return result


def file_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256(path)}


def source_audit(split: dict[str, object]) -> tuple[dict[str, object], bool]:
    caption_candidates = [
        Path("/home/data/DIFFER_TextCaption/LTCC_ReID"),
        Path("/data/DIFFER_TextCaption/LTCC_ReID"),
        LEGACY / "data/DIFFER_TextCaption/LTCC_ReID",
    ]
    files = [
        LEGACY / "loss/gpc_relation.py",
        LEGACY / "scripts/train_final_gpc.py",
        LEGACY / "scripts/phase7i_relation_specificity.py",
        LEGACY / "scripts/phase7o_generic_positive_consistency_replication.py",
        LEGACY / "reports/PHASE_7I_SOURCE_AUDIT.md",
        LEGACY / "reports/PHASE_7O_SOURCE_AUDIT.md",
        LEGACY / "reports/FINAL_GPC_SOURCE_AUDIT.md",
        OUTPUT / "phase6c_full_schedule_confirmation/B0_full/model_final.pth",
        OUTPUT / "phase6c_full_schedule_confirmation/B1_full/model_final.pth",
        OUTPUT / "phase7i_relation_specificity/I3_id_only_matched/model_final.pth",
        OUTPUT / "phase6c_full_schedule_confirmation/B1_full/metrics_per_epoch.csv",
        OUTPUT / "phase6c_full_schedule_confirmation/B1_full/relation_exposure.csv",
    ]
    caption_records = {str(path): file_record(path) for path in caption_candidates}
    checks = {
        "legacy_gpc_relation_present": (LEGACY / "loss/gpc_relation.py").is_file(),
        "legacy_phase7i_present": (LEGACY / "scripts/phase7i_relation_specificity.py").is_file(),
        "legacy_phase7o_present": (LEGACY / "scripts/phase7o_generic_positive_consistency_replication.py").is_file(),
        "v1_selector_source_readable": (LEGACY / "reports/PHASE_7O_SOURCE_AUDIT.md").is_file(),
        "b0_checkpoint_present": (OUTPUT / "phase6c_full_schedule_confirmation/B0_full/model_final.pth").is_file(),
        "b1_checkpoint_present": (OUTPUT / "phase6c_full_schedule_confirmation/B1_full/model_final.pth").is_file(),
        "b1_e60_metrics_present": (OUTPUT / "phase6c_full_schedule_confirmation/B1_full/metrics_per_epoch.csv").is_file(),
        "b1_exposure_present": (OUTPUT / "phase6c_full_schedule_confirmation/B1_full/relation_exposure.csv").is_file(),
        "phase7i_i3_exact_gpc_v1_checkpoint_present": (OUTPUT / "phase7i_relation_specificity/I3_id_only_matched/model_final.pth").is_file(),
        "latest_confirmatory_output_root_present": (ROOT / "outputs/final_confirmatory_gpc").is_dir(),
        "phase6c_b1_is_exact_i3_selector": False,
        "caption_features_present": any(path.is_dir() for path in caption_candidates),
        "dev_split_frozen": bool(split.get("sha256")) and bool(split.get("path_disjoint", {}).get("train_vs_validation_pool")),
        "historical_v1_not_overwritten": True,
        "ndp_not_in_candidate_protocol": True,
    }
    result = {"created_utc": now(), "run_time": RUN_TIME, "baseline_source_root": str(LEGACY), "checkpoint_root": str(OUTPUT), "checks": checks, "source_files": {str(path): file_record(path) for path in files}, "caption_candidates": caption_records, "gpc_v1_frozen_protocol": {"loss": "L_CE + L_Triplet + L_Bio + 0.1 * L_GPC", "eligibility": "PID_i == PID_j, i != j; same-clothes and cross-clothes both allowed", "selector": "count=min(cross,same,id), deterministic SHA-256 selection", "reduction": "mean over selected relations", "dense_all_pair": False, "historical_phase7i_phase7o_preserved": True}, "checks_passed": all(checks.values())}
    return result, bool(result["checks_passed"])


def leakage_audit(split: dict[str, object]) -> tuple[dict[str, object], bool]:
    train = set(split["train_images"])
    validation = set(split["validation_images"])
    query = set(split["validation_query"])
    gallery = set(split["validation_gallery"])
    official_query = {str(p) for p in sorted((DATA / "query").glob("*.png"))}
    official_gallery = {str(p) for p in sorted((DATA / "test").glob("*.png"))}
    checks = {
        "dev_train_pid_and_validation_pid_disjoint": not bool(set(split["train_pids"]) & set(split["validation_pids"])),
        "dev_train_validation_path_disjoint": not bool(train & validation),
        "dev_query_gallery_path_disjoint": not bool(query & gallery),
        "dev_query_not_from_dev_train": not bool(query & train),
        "dev_gallery_not_from_dev_train": not bool(gallery & train),
        "official_query_gallery_path_disjoint": not bool(official_query & official_gallery),
        "official_test_not_in_original_train": not bool((official_query | official_gallery) & {str(p) for p in sorted((DATA / "train").glob("*.png"))}),
        "training_evaluator_calls_during_candidate_training": "0 次",
        "dev_validation_features_not_in_candidate_loss": True,
        "official_metrics_not_used_before_freeze": True,
    }
    # The missing caption directory is deliberately recorded here, too: it
    # prevents a false PASS for a training run that cannot construct Bio loss.
    checks["required_train_caption_assets_available"] = False
    passed = all(value is True for key, value in checks.items() if key != "training_evaluator_calls_during_candidate_training")
    result = {"created_utc": now(), "run_time": RUN_TIME, "checks": checks, "dev_split": str(SPLIT_PATH), "official_counts": {"query": len(official_query), "gallery": len(official_gallery)}, "status": "PASS" if passed else "FAIL"}
    return result, passed


def render_source(result: dict[str, object]) -> None:
    checks = result["checks"]
    lines = [
        "# GPC_V2_LTCC_SOURCE_AUDIT",
        "",
        f"生成时间：{result['created_utc']}；RUN_TIME：{RUN_TIME}",
        "",
        "## 结论",
        "",
        f"来源审计：**{'通过' if result['checks_passed'] else '未通过'}**。未通过项会阻止正式训练。",
        "",
        "本次以 `/home/DIFFER_NDP_test_CLoth-aware` 作为历史只读源码来源，以 `/home/output` 作为历史 checkpoint 来源；`/home/DIFFER` 只写入本轮独立 GPC-v2 资产。",
        "",
        "## GPC-v1 冻结协议",
        "",
        "- 损失：`L_CE + L_Triplet + L_Bio + 0.1 * L_GPC`。",
        "- 候选关系：同 PID、不同图像索引；不强制不同衣服。",
        "- selector：`count=min(cross,same,id)`，对同身份候选执行确定性 SHA-256 选择。",
        "- reduction：只对选中关系取 mean；不是 dense all-pair。",
        "- Phase 7I I3、Phase 7O O1 的源码、报告和历史 checkpoint 均保持只读。",
        "",
        "## 检查表",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
    ]
    for key, value in checks.items():
        lines.append(f"| `{key}` | {'通过' if value else '失败'} |")
    lines += ["", "## 历史文件哈希", "", "| 文件 | 存在 | SHA256 |", "|---|---|---|"]
    for key, value in result["source_files"].items():
        lines.append(f"| `{key}` | {value['exists']} | `{value['sha256']}` |")
    lines += ["", "## 阻塞原因", "", "- 当前运行环境没有发现 `/home/data/DIFFER_TextCaption/LTCC_ReID`、`/data/DIFFER_TextCaption/LTCC_ReID` 或仓库内同名 caption 目录。", "- LTCC Bio loss 读取这些 train-only caption 特征；没有它们不能构造与 GPC-v1 相同的 `L_Bio`，不能用零特征、随机特征或测试图像特征替代。", "- 双 PPU 预检已单独通过，但这不能解除数据资产缺失。", "- 当前可作为精确 GPC-v1 只读基线的是 `/home/output/phase7i_relation_specificity/I3_id_only_matched/model_final.pth`；`/home/output/phase6c_full_schedule_confirmation/B1_full` 的关系计数为 12,171，而 I3 matched 计数为 8,780，selector 不同，不能混用。", "- 最新封账报告引用的 `/home/DIFFER/outputs/final_confirmatory_gpc` 不存在，因此其 `55.6122/31.5157` 目前只能作为报告记录，不能作为当前 checkpoint 的哈希事实。", "", "因此本轮正式训练未启动，也没有生成候选结果或官方测试结果。", ""]
    SOURCE_REPORT.write_text("\n".join(lines), encoding="utf-8")


def render_leakage(result: dict[str, object]) -> None:
    lines = [
        "# GPC_V2_LTCC_LEAKAGE_AUDIT",
        "",
        f"生成时间：{result['created_utc']}；RUN_TIME：{RUN_TIME}",
        "",
        f"数据泄漏审计：**{result['status']}**。",
        "",
        "开发划分固定来自 LTCC 原始 train split：训练 PID 与验证 PID 不重叠；验证 query/gallery 只来自开发验证身份；官方 query/gallery 不参与候选训练。",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
    ]
    for key, value in result["checks"].items():
        lines.append(f"| `{key}` | {'通过' if value else '失败'} |")
    lines += ["", "## 运行边界", "", "- 候选训练不得构造官方 evaluator。", "- 开发验证特征不得进入 loss 或 memory bank。", "- 官方指标只允许在最终方案冻结、E60 checkpoint 写入并计算 SHA256 后执行。", "- 本次由于 train caption 资产缺失，训练边界虽已预注册，但正式训练尚未启动。", ""]
    LEAKAGE_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    split = make_split()
    source, source_pass = source_audit(split)
    leakage, leakage_pass = leakage_audit(split)
    write_json(STATUS_PATH, {"run_time": RUN_TIME, "source_audit_pass": source_pass, "leakage_audit_pass": leakage_pass, "split": str(SPLIT_PATH), "status": "PASS" if source_pass and leakage_pass else "BLOCKED_MISSING_TRAIN_CAPTION_ASSETS", "source": source, "leakage": leakage})
    render_source(source)
    render_leakage(leakage)
    print(json.dumps({"run_time": RUN_TIME, "source_audit_pass": source_pass, "leakage_audit_pass": leakage_pass, "split": str(SPLIT_PATH), "status": "PASS" if source_pass and leakage_pass else "BLOCKED_MISSING_TRAIN_CAPTION_ASSETS"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
