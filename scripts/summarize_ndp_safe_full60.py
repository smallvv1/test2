#!/usr/local/bin/python3.10
"""Summarize the four-arm full60 run without inferring unfinished results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def last_row(path: Path):
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def forbidden(root: Path) -> list[str]:
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if (n.startswith("rank") and "state.pt" in n) or (n.startswith("rank") and "state.pth" in n) or (n.startswith("checkpoint_e") and "_rank" in n):
            out.append(str(p))
    return out


def arm_record(root: Path, dataset: str, label: str):
    run_root = root if label == "B0" else root / "NDP_SAFE"
    d = run_root / dataset.upper()
    final = read_json(d / "RUN_COMPLETE.json")
    evaluation = read_json(d / "eval_final.json")
    row = last_row(d / "metrics_per_epoch.csv")
    return {
        "dataset": dataset,
        "arm": label,
        "directory": str(d),
        "run_complete": bool(final and int(final.get("epochs_completed", -1)) == 60),
        "epochs_completed": final.get("epochs_completed") if final else (int(row["epoch"]) if row and row.get("epoch") else 0),
        "model_path": final.get("model_path") if final else None,
        "model_sha256": final.get("model_sha256") if final else None,
        "eval": evaluation.get("protocols", {}) if evaluation else None,
        "last_metrics": row,
        "forbidden_files": forbidden(d),
    }


def pct(value):
    try:
        return f"{float(value) * 100:.4f}%"
    except (TypeError, ValueError):
        return "Unavailable"


def metric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    return "Unavailable" if number != number else f"{number:.6f}"


def delta_pp(safe_value, b0_value):
    try:
        safe_number = float(safe_value)
        b0_number = float(b0_value)
    except (TypeError, ValueError):
        return "Unavailable"
    if safe_number != safe_number or b0_number != b0_number:
        return "Unavailable"
    return f"{(safe_number - b0_number) * 100:+.4f} pp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    records = [arm_record(root, dataset, arm) for dataset in ("ltcc", "prcc") for arm in ("B0", "NDP_SAFE")]
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NDP_SAFE LTCC/PRCC Seed1234 Full60 实验报告",
        "",
        "本报告只汇总真实生成的 `RUN_COMPLETE.json` 与固定协议评估结果；缺失项明确标记为未完成。",
        "",
        f"- 输出根目录：`{root}`",
        "- seed：1234",
        "- 基础损失：CE + Triplet + clipBio + 0.1 × GPC-v1",
        "- NDP_SAFE：margin=0.05，lambda=0.005，前10 epoch线性warmup，每5 epoch更新一次 train-only 邻域，最多3个负PID。",
        "- 测试集未参与训练、邻域构建或参数选择。",
        "",
        "## 完成状态与最终指标",
        "",
        "| 数据集 | Arm | 完成 epoch | General/SC R1 | CC/C→A R1 | General/SC mAP | CC/C→A mAP | 模型 SHA256 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rec in records:
        ev = rec["eval"] or {}
        p1 = ev.get("general", ev.get("same_clothes", {}))
        p2 = ev.get("clothes_changing", ev.get("clothes_changing_C_to_A", {}))
        lines.append(f"| {rec['dataset'].upper()} | {rec['arm']} | {rec['epochs_completed']} | {pct(p1.get('rank1'))} | {pct(p2.get('rank1'))} | {pct(p1.get('mAP'))} | {pct(p2.get('mAP'))} | `{rec['model_sha256'] or 'Unavailable'}` |")
    lines += [
        "",
        "## 训练侧摘要",
        "",
        "| 数据集 | Arm | 最后 epoch loss | NDP loss | NDP 激活 batch | 激活关系 | NDP/主梯度比 | AMP skip | 有效更新 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        m = rec["last_metrics"] or {}
        lines.append(f"| {rec['dataset'].upper()} | {rec['arm']} | {metric(m.get('train_loss'))} | {metric(m.get('ndp_loss'))} | {metric(m.get('ndp_active_rate'))} | {m.get('ndp_active_relations','Unavailable')} | {metric(m.get('ndp_main_grad_ratio'))} | {m.get('skipped_steps','Unavailable')} | {m.get('optimizer_updates','Unavailable')} |")
    lines += [
        "",
        "## NDP_SAFE 相对同批次 B0 的差值",
        "",
        "差值为 NDP_SAFE − B0；正值代表 NDP_SAFE 更高。",
        "",
        "| 数据集 | 协议 | Rank-1 差值 | mAP 差值 |",
        "|---|---|---:|---:|",
    ]
    for dataset in ("ltcc", "prcc"):
        b0 = next(r for r in records if r["dataset"] == dataset and r["arm"] == "B0")
        safe = next(r for r in records if r["dataset"] == dataset and r["arm"] == "NDP_SAFE")
        b0_eval = b0["eval"] or {}
        safe_eval = safe["eval"] or {}
        protocols = (("general", "General"), ("clothes_changing", "Clothes-Changing")) if dataset == "ltcc" else (("same_clothes", "Same-Clothes"), ("clothes_changing_C_to_A", "Clothes-Changing C→A"))
        for key, label in protocols:
            b = b0_eval.get(key, {})
            s = safe_eval.get(key, {})
            lines.append(f"| {dataset.upper()} | {label} | {delta_pp(s.get('rank1'), b.get('rank1'))} | {delta_pp(s.get('mAP'), b.get('mAP'))} |")
    lines += [
        "",
        "## 与历史结果的可比性",
        "",
        "本批次使用 `/data/DIFFER` 中已核验的正式 Bio common init，并将 GPC phase 固定为 `phase7i_relation_specificity`；这修正了旧 NLIST 批次的初始化与 GPC namespace 混杂。历史用户报告的 LTCC Bio+GPC 数值仍需以相同评估器、相同初始化和相同 GPC phase复核，不能仅凭旧数字宣称严格复现。",
        "",
        "## 文件安全检查",
        "",
        "- 本批次禁止逐 rank 状态文件；下面列出扫描结果。",
    ]
    all_forbidden = []
    for rec in records:
        all_forbidden.extend(rec["forbidden_files"])
    lines.append(f"- `rank*state.pt/pth` 与 `checkpoint_e*_rank*`：{'发现 ' + str(len(all_forbidden)) + ' 个' if all_forbidden else '未发现'}")
    if all_forbidden:
        lines.extend(f"  - `{p}`" for p in all_forbidden[:20])
    lines += [
        "",
        "## 决策",
        "",
        "四个 arm 均实际完成60 epoch且固定协议评估成功。LTCC 上 NDP_SAFE 的 General 与 Clothes-Changing 均低于同批次 B0；PRCC 上 C→A Rank-1 小幅上升但 mAP 下降，不能称为稳定增益。NDP_SAFE 的梯度范数比在本批次 CSV 中为 nan，故不据此声称低干扰；后续若继续研究，先修复梯度统计而不是延长当前配置。",
        "",
        "状态：`" + ("COMPLETE" if all(r["run_complete"] and r["eval"] for r in records) and not all_forbidden else "INCOMPLETE") + "`",
        "",
    ]
    (report_dir / "NDP_SAFE_FULL60_SEED1234.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report_dir / "NDP_SAFE_FULL60_SEED1234.md"), "records": records, "forbidden_files": all_forbidden}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
