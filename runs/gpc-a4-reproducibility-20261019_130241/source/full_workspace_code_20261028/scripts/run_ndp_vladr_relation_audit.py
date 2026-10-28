#!/usr/local/bin/python3.10
"""VLADR/NDP 关系研究的可复现实验入口；仅 train-only 离线统计和 CPU 单测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-time", required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"运行时间：{args.run_time}",
        "本轮范围：VLADR关系实现核对、独立NDP原型单测、LTCC train-only静态诊断。",
        "明确禁止：修改GPC-v1、正式processor、PRCC、NDP历史结果、官方query/gallery训练选择。",
    ]
    source_paths = [
        Path(__file__).resolve(),
        Path("/home/DIFFER/scripts/ndp_vladr_relation.py"),
        Path("/home/DIFFER/scripts/test_ndp_vladr_relation.py"),
        Path("/home/DIFFER/scripts/ndp_vladr_static_diagnostic.py"),
        Path("/data/DIFFER/scripts/gpc_v2_train.py"),
        Path("/data/DIFFER/scripts/phase7i_relation_specificity.py"),
        Path("/data/DIFFER/loss/make_loss.py"),
        Path("/data/DIFFER/loss/triplet_loss.py"),
        Path("/data/DIFFER/model/build.py"),
        Path("/data/DIFFER/model/eva_meta_bio.py"),
        Path("/home/DIFFER/references/CVPR2026-VLADR/reid/trainer_stage2.py"),
    ]
    source_hashes = {str(p): sha256(p) for p in source_paths if p.exists()}
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for p in source_paths:
        if p.exists():
            shutil.copy2(p, snapshot / (p.name + "." + sha256(p)[:12]))
    test_cmd = [sys.executable, "/home/DIFFER/scripts/test_ndp_vladr_relation.py"]
    test = subprocess.run(test_cmd, text=True, capture_output=True, check=False)
    (out / "unit_test.stdout.log").write_text(test.stdout + test.stderr, encoding="utf-8")
    unit_pass = test.returncode == 0 and "NDP_VLADR_RELATION_UNIT_TEST_PASS" in test.stdout
    log_lines.append(f"独立张量单测：{'PASS' if unit_pass else 'FAIL'}，returncode={test.returncode}。")
    diagnostics = {}
    for seed in (1234, 4321):
        path = out / f"static_diagnostic_seed{seed}.json"
        cmd = [sys.executable, "/home/DIFFER/scripts/ndp_vladr_static_diagnostic.py", "--seed", str(seed), "--output", str(path)]
        run = subprocess.run(cmd, text=True, capture_output=True, check=False)
        (out / f"static_diagnostic_seed{seed}.stdout.log").write_text(run.stdout + run.stderr, encoding="utf-8")
        if run.returncode != 0:
            log_lines.append(f"seed{seed} train-only静态诊断：FAIL，returncode={run.returncode}。")
            diagnostics[str(seed)] = {"status": "FAIL", "returncode": run.returncode, "stderr": run.stderr[-2000:]}
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            diagnostics[str(seed)] = data
            log_lines.append(
                f"seed{seed} train-only静态诊断：PASS；有效局部anchor={data['valid_local_anchor_count']}/"
                f"{data['sample_anchor_count']}；参考可靠率={data['local_candidate']['reference_reliable_rate']:.4f}。"
            )
    manifest = {
        "run_time": args.run_time,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "VLADR relation audit and independent NDP prototype; train-only offline",
        "official_reference": {
            "repository": "https://github.com/zhoujiahuan1991/CVPR2026-VLADR",
            "local_commit": subprocess.run(["git", "-C", "/home/DIFFER/references/CVPR2026-VLADR", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip(),
            "trainer": "/home/DIFFER/references/CVPR2026-VLADR/reid/trainer_stage2.py",
            "implementation_summary": "cosine affinity -> row softmax temperature 0.005, diagonal included, KL(new||old), old target detached",
        },
        "unit_test_pass": unit_pass,
        "official_query_gallery_used": False,
        "model_forward_or_optimizer_step": False,
        "source_hashes": source_hashes,
        "diagnostics": diagnostics,
        "historical_gate": {
            "real_trajectory_natural_ndp_activation": "UNVERIFIED",
            "ppu_aa_comparability": "UNRESOLVED",
            "formal_training_started": False,
        },
    }
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "audit.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "unit_test_pass": unit_pass, "source_count": len(source_hashes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
