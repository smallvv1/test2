#!/usr/local/bin/python3.10
"""顺序执行两个 seed 的真实 continuation NDP 激活审计，不执行 optimizer.step。"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/DIFFER")


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def env():
    e = os.environ.copy()
    e.update({"PPU_SDK": "/usr/local/PPU_SDK", "PPU_HOME": "/usr/local/PPU_SDK", "PPU_PATH": "/usr/local/PPU_SDK", "LD_LIBRARY_PATH": "/usr/local/PPU_SDK/lib:/usr/local/PPU_SDK/sailSHMEM/lib:" + e.get("LD_LIBRARY_PATH", "")})
    return e


def main():
    run_time = os.environ["NDP_RUN_TIME"]
    run_dir = ROOT / "outputs" / f"ndp_overnight_{run_time}"
    results = []
    start = time.time()
    for seed in (1234, 4321):
        trajectory = run_dir / f"trajectory_seed{seed}"
        log = run_dir / "logs" / f"NDP真实轨迹激活_seed{seed}_{run_time}.log"
        recovery_index = 1
        while log.exists():
            log = run_dir / "logs" / f"NDP真实轨迹激活_seed{seed}_{run_time}_恢复{recovery_index}.log"
            recovery_index += 1
        command = ["/usr/local/bin/python3.10", str(ROOT / "scripts/ndp_real_trajectory_activation.py"), "--seed", str(seed), "--trajectory-dir", str(trajectory), "--resume-dir", str(ROOT / "outputs/gpc_v2_ltcc_20261015_155822/training/two_seed" / f"B0_seed{seed}")]
        write(run_dir / "stage3_status.json", {"stage": f"stage3_seed{seed}", "status": "RUNNING", "time": now(), "seed": seed, "log": str(log)})
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env())
        summary = trajectory / "activation_audit" / "activation_summary.json"
        result = {"seed": seed, "returncode": proc.returncode, "log": str(log), "summary": str(summary), "status": "PASS" if proc.returncode == 0 and summary.is_file() else "FAIL"}
        results.append(result)
        write(run_dir / "stage3_results.json", results)
        if result["status"] != "PASS":
            write(run_dir / "stage3_status.json", {"stage": f"stage3_seed{seed}", "status": "FAIL", "time": now(), "result": result})
            return
    write(run_dir / "stage3_status.json", {"stage": "stage3_activation_audit", "status": "PASS", "time": now(), "results": results, "elapsed_seconds": time.time() - start})


if __name__ == "__main__":
    main()
