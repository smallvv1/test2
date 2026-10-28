#!/usr/local/bin/python3.10
"""NDP 夜间阶段监督器。

只允许在已有真实 PPU 探针通过后，恢复两个 B0/GPC-v1 E60 完整状态，执行
train-only continuation 轨迹采集。若恢复状态、PPU、AMP 或 rank 间更新不可比，
立即停止，不自动凑训练时长。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/DIFFER")
OLD_OUTPUT = ROOT / "outputs/gpc_v2_ltcc_20261015_155822"
PROBE_OUTPUT = ROOT / "outputs/ndp_live_ppu_gradient_probe_20261020_191922"
PROBE_SCRIPT = ROOT / "scripts/ndp_live_ppu_gradient_probe.py"
CONT_SCRIPT = ROOT / "scripts/ndp_continuation_trajectory.py"
WORLD_SIZE = 2
MAX_SECONDS = 8 * 3600


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_status(run_dir: Path, stage: str, status: str, **extra) -> None:
    payload = {"stage": stage, "status": status, "time": now(), **extra}
    write_json(run_dir / "stage_status.json", payload)
    write_json(run_dir / "heartbeat.json", payload)


def ppu_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PPU_SDK": "/usr/local/PPU_SDK",
        "PPU_HOME": "/usr/local/PPU_SDK",
        "PPU_PATH": "/usr/local/PPU_SDK",
        "LD_LIBRARY_PATH": "/usr/local/PPU_SDK/lib:/usr/local/PPU_SDK/sailSHMEM/lib:" + env.get("LD_LIBRARY_PATH", ""),
        "PYTHONPATH": "/data/DIFFER/scripts:/data/DIFFER:" + env.get("PYTHONPATH", ""),
    })
    return env


def stage0_audit() -> dict:
    manifest = PROBE_OUTPUT / "manifest.json"
    summary = PROBE_OUTPUT / "ndp_live_ppu_gradient_probe.json"
    if not manifest.is_file() or not summary.is_file():
        raise RuntimeError("阶段0缺少既有 PPU 探针 manifest/summary")
    m = json.loads(manifest.read_text())
    s = json.loads(summary.read_text())
    expected = {
        "PYTHON_ENV": "PYTHON_ENV_PASS",
        "NEIGHBOR_GRAPH_AUDIT": "NEIGHBOR_GRAPH_AUDIT_PASS",
        "LIVE_PPU_FORWARD": "LIVE_PPU_FORWARD_PASS",
        "LIVE_PPU_BACKWARD": "LIVE_PPU_BACKWARD_PASS",
        "NDP_TENSOR_ACTIVATION": "NDP_TENSOR_ACTIVATION_PASS",
        "REAL_TRAJECTORY_ACTIVATION": "REAL_TRAJECTORY_ACTIVATION_UNVERIFIED",
        "GRADIENT_INCREMENTALITY": "GRADIENT_INCREMENTALITY_UNRESOLVED",
        "SHORT_PILOT": "SHORT_PILOT_NOT_READY",
    }
    if s.get("decisions") != expected:
        raise RuntimeError(f"阶段0既有状态不符合预期: {s.get('decisions')}")
    if s.get("optimizer_step_executed") is not False or s.get("official_query_gallery_used") is not False:
        raise RuntimeError("阶段0既有探针边界被污染")
    if not PROBE_SCRIPT.is_file() or sha256_file(PROBE_SCRIPT) != m.get("source_script_sha256"):
        raise RuntimeError("阶段0源码 SHA256 与 manifest 不一致")
    return {"manifest": str(manifest), "summary": str(summary), "source_sha256": sha256_file(PROBE_SCRIPT), "decisions": s["decisions"], "passed": True}


def stage1_audit() -> dict:
    split = OLD_OUTPUT / "development_split_manifest.json"
    if not split.is_file():
        raise RuntimeError("缺少冻结开发训练划分")
    split_data = json.loads(split.read_text())
    if split_data.get("official_query_gallery_used") or split_data.get("query_gallery_used_for_training"):
        raise RuntimeError("训练划分 manifest 声明使用了官方 query/gallery")
    result = {"split_sha256": sha256_file(split), "split": str(split), "seeds": {}, "passed": True}
    for seed in (1234, 4321):
        run = OLD_OUTPUT / "training" / "two_seed" / f"B0_seed{seed}"
        manifest = run / "run_manifest.json"
        arm_manifest = run / "v2_arm_manifest.json"
        state_paths = [run / "rank0_state_epoch060.pt", run / "rank1_state_epoch060.pt"]
        if not manifest.is_file() or not arm_manifest.is_file() or not all(path.is_file() for path in state_paths):
            raise RuntimeError(f"seed{seed} 缺少完整 rank 状态")
        data = json.loads(manifest.read_text())
        arm_data = json.loads(arm_manifest.read_text())
        required = {
            "seed": seed,
            "arm": "I3",
            "lambda_relation": 0.1,
            "pair_budget": "per-batch minimum feasible count across I1/I2/I3; deterministic hash selection",
        }
        for key, value in required.items():
            if data.get(key) != value:
                raise RuntimeError(f"seed{seed} manifest {key} 不匹配: {data.get(key)!r} != {value!r}")
        if arm_data.get("selector") != "Phase7I I3 deterministic SHA-256 ID-only selector" or arm_data.get("gpc_v1_selector_preserved") is not True:
            raise RuntimeError(f"seed{seed} B0 arm manifest 未确认 GPC-v1 selector")
        result["seeds"][str(seed)] = {
            "run_dir": str(run),
            "manifest_sha256": sha256_file(manifest),
            "arm_manifest_sha256": sha256_file(arm_manifest),
            "rank_state_sha256": {f"rank{i}": sha256_file(state_paths[i]) for i in range(2)},
            "epoch": 60,
            "global_step": 35760,
            "complete_state": True,
        }
    return result


def run_seed(run_dir: Path, seed: int, resume_dir: Path, deadline: float) -> dict:
    seed_dir = run_dir / f"trajectory_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / f"NDP基础轨迹_seed{seed}_{run_dir.name.rsplit('_', 1)[-1]}.log"
    if log_path.exists():
        log_path = log_path.with_name(log_path.stem + "_recovery1.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    port = str(29680 + (seed % 100))
    command = [
        "/usr/local/bin/python3.10", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", f"--master_port={port}",
        str(CONT_SCRIPT), "--seed", str(seed), "--resume-dir", str(resume_dir), "--run-dir", str(seed_dir), "--max-effective-updates", "400",
    ]
    update_status(run_dir, f"stage2_seed{seed}", "RUNNING", seed=seed, pid=None, log=str(log_path), resume_dir=str(resume_dir))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=ppu_env(), start_new_session=True)
        update_status(run_dir, f"stage2_seed{seed}", "RUNNING", seed=seed, pid=proc.pid, log=str(log_path), resume_dir=str(resume_dir))
        while proc.poll() is None:
            if time.time() > deadline:
                proc.terminate()
                time.sleep(5)
                if proc.poll() is None:
                    proc.kill()
                update_status(run_dir, f"stage2_seed{seed}", "STOPPED_TIME_BUDGET", seed=seed, pid=proc.pid, log=str(log_path))
                return {"seed": seed, "returncode": proc.returncode, "status": "STOPPED_TIME_BUDGET", "log": str(log_path)}
            summary = seed_dir / "trajectory_summary.json"
            progress = {}
            if summary.is_file():
                try: progress = json.loads(summary.read_text())
                except Exception: progress = {}
            update_status(run_dir, f"stage2_seed{seed}", "RUNNING", seed=seed, pid=proc.pid, log=str(log_path), progress=progress)
            time.sleep(60)
        summary = seed_dir / "trajectory_summary.json"
        progress = json.loads(summary.read_text()) if summary.is_file() else {}
        status = "PASS" if proc.returncode == 0 and not progress.get("failure") else "FAIL"
        update_status(run_dir, f"stage2_seed{seed}", status, seed=seed, pid=proc.pid, returncode=proc.returncode, progress=progress, log=str(log_path))
        return {"seed": seed, "returncode": proc.returncode, "status": status, "summary": progress, "log": str(log_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-time", required=True)
    ap.add_argument("--resume-existing", action="store_true")
    args = ap.parse_args()
    run_dir = ROOT / "outputs" / f"ndp_overnight_{args.run_time}"
    if run_dir.exists() and not args.resume_existing:
        raise RuntimeError(f"运行目录已存在，必须显式 --resume-existing: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    start = time.time()
    manifest = {
        "run_time": args.run_time,
        "started_utc": now(),
        "max_seconds": MAX_SECONDS,
        "project": str(ROOT),
        "git_reference": "b7464735c3cd85b71a60b6a195aaf644d5c4a5b2",
        "probe_source": str(PROBE_SCRIPT),
        "continuation_source": str(CONT_SCRIPT),
        "official_query_gallery_used": False,
        "cpu_model_fallback": False,
        "gpc_v1_modified": False,
        "ndp_optimizer_step_allowed_only_after_gates": True,
        "stages": ["stage0_probe_audit", "stage1_state_audit", "stage2_base_continuation", "stage3_activation_audit", "stage4_gradient_probe", "stage5_short_intervention"],
    }
    if not args.resume_existing:
        write_json(run_dir / "RUN_MANIFEST.json", manifest)
        update_status(run_dir, "stage0_probe_audit", "RUNNING")
    try:
        if args.resume_existing:
            s0 = json.loads((run_dir / "stage0_probe_audit.json").read_text())
            s1 = json.loads((run_dir / "stage1_state_audit.json").read_text())
            if not s0.get("passed") or not s1.get("passed"):
                raise RuntimeError("恢复运行缺少已通过的阶段0/1门禁")
        else:
            s0 = stage0_audit()
            write_json(run_dir / "stage0_probe_audit.json", s0)
            update_status(run_dir, "stage0_probe_audit", "PASS", artifact=str(run_dir / "stage0_probe_audit.json"))
            s1 = stage1_audit()
            write_json(run_dir / "stage1_state_audit.json", s1)
            update_status(run_dir, "stage1_state_audit", "PASS", artifact=str(run_dir / "stage1_state_audit.json"))
    except Exception as exc:
        failure = {"stage": "stage0_or_stage1", "type": type(exc).__name__, "message": str(exc), "time": now()}
        write_json(run_dir / "failure_events.json", [failure])
        update_status(run_dir, "stage0_or_stage1", "BLOCKED", failure=failure)
        write_json(run_dir / "FINAL_DECISION.json", {"decision": "NO_GO", "reason": str(exc), "elapsed_seconds": time.time() - start})
        return

    deadline = start + MAX_SECONDS - 300
    seed_results = []
    for seed in (1234, 4321):
        result = run_seed(run_dir, seed, OLD_OUTPUT / "training" / "two_seed" / f"B0_seed{seed}", deadline)
        seed_results.append(result)
        write_json(run_dir / "seed_results.json", seed_results)
        if result.get("status") != "PASS":
            write_json(run_dir / "FINAL_DECISION.json", {"decision": "NO_GO", "reason": f"stage2 seed{seed} failed", "seed_results": seed_results, "elapsed_seconds": time.time() - start})
            return

    write_json(run_dir / "FINAL_DECISION.json", {"decision": "STAGE2_COMPLETE_NEXT_AUDIT_REQUIRED", "seed_results": seed_results, "elapsed_seconds": time.time() - start})
    update_status(run_dir, "stage2_base_continuation", "PASS", seed_results=seed_results)


if __name__ == "__main__":
    main()
