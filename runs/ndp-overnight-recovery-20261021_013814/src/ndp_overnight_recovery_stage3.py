#!/usr/local/bin/python3.10
"""恢复昨晚 NDP 阶段3：只做真实轨迹激活审计，不执行 optimizer.step。"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/DIFFER')
OLD_RUN = ROOT / 'outputs/ndp_overnight_20261020_201841'
ACTIVATION = ROOT / 'scripts/ndp_real_trajectory_activation.py'


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def emit(stream, message, **fields):
    payload = {'time': now(), 'message': message, **fields}
    stream.write(json.dumps(payload, ensure_ascii=False) + '\n')
    stream.flush()


def ppu_env():
    env = os.environ.copy()
    env.update({
        'PPU_SDK': '/usr/local/PPU_SDK',
        'PPU_HOME': '/usr/local/PPU_SDK',
        'PPU_PATH': '/usr/local/PPU_SDK',
        'LD_LIBRARY_PATH': '/usr/local/PPU_SDK/lib:/usr/local/PPU_SDK/sailSHMEM/lib:' + env.get('LD_LIBRARY_PATH', ''),
        'PYTHONPATH': '/data/DIFFER/scripts:/data/DIFFER:' + env.get('PYTHONPATH', ''),
        'PYTHONUNBUFFERED': '1',
    })
    return env


def link_nodes(old_seed_dir: Path, new_seed_dir: Path):
    new_seed_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for old_node in sorted(old_seed_dir.glob('node_update*/model.pth')):
        node = new_seed_dir / old_node.parent.name
        node.mkdir(parents=True, exist_ok=True)
        target = node / 'model.pth'
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(old_node)
        links.append({'path': str(target), 'target': str(old_node), 'target_sha256': sha256(old_node)})
    if not links:
        raise RuntimeError(f'没有找到既有 continuation 模型节点: {old_seed_dir}')
    return links


def main():
    if len(sys.argv) != 2:
        raise SystemExit('用法：ndp_overnight_recovery_stage3.py RUN_TIME')
    run_time = sys.argv[1]
    run_dir = ROOT / 'outputs' / f'ndp_overnight_recovery_{run_time}'
    if run_dir.exists():
        raise RuntimeError(f'恢复目录已存在，拒绝覆盖: {run_dir}')
    run_dir.mkdir(parents=True)
    (run_dir / 'logs').mkdir()
    main_log = ROOT / 'logs' / f'NDP夜间恢复阶段3_{run_time}.log'
    start = time.time()
    manifest = {
        'run_time': run_time,
        'started_utc': now(),
        'source_run': str(OLD_RUN),
        'stage': 'stage3_activation_audit_only',
        'source_script': str(ACTIVATION),
        'source_script_sha256': sha256(ACTIVATION),
        'python': '/usr/local/bin/python3.10',
        'python_unbuffered': True,
        'model_computation': 'PPU-ZW810E only',
        'optimizer_step_executed': False,
        'official_query_gallery_used': False,
        'cpu_model_fallback': False,
        'old_stage0': str(OLD_RUN / 'stage0_probe_audit.json'),
        'old_stage1': str(OLD_RUN / 'stage1_state_audit.json'),
    }
    atomic_json(run_dir / 'RUN_MANIFEST.json', manifest)
    status = {'stage': 'stage3_activation_audit', 'status': 'RUNNING', 'time': now(), 'run_time': run_time}
    atomic_json(run_dir / 'stage_status.json', status)
    atomic_json(run_dir / 'heartbeat.json', status)
    with main_log.open('w', encoding='utf-8', buffering=1) as outer:
        emit(outer, '恢复阶段3开始', run_time=run_time, source_run=str(OLD_RUN), activation_sha256=manifest['source_script_sha256'])
        results = []
        try:
            for seed in (1234, 4321):
                old_seed = OLD_RUN / f'trajectory_seed{seed}'
                new_seed = run_dir / f'trajectory_seed{seed}'
                links = link_nodes(old_seed, new_seed)
                atomic_json(run_dir / 'heartbeat.json', {'stage': f'stage3_seed{seed}', 'status': 'RUNNING', 'time': now(), 'seed': seed, 'linked_nodes': len(links)})
                child_log = run_dir / 'logs' / f'NDP真实轨迹激活_seed{seed}_{run_time}.log'
                command = ['/usr/local/bin/python3.10', '-u', str(ACTIVATION), '--seed', str(seed), '--trajectory-dir', str(new_seed), '--resume-dir', str(ROOT / 'outputs/gpc_v2_ltcc_20261015_155822/training/two_seed' / f'B0_seed{seed}')]
                emit(outer, '启动seed阶段3', seed=seed, command=command, child_log=str(child_log), linked_nodes=len(links))
                with child_log.open('w', encoding='utf-8', buffering=1) as child:
                    proc = subprocess.run(command, stdout=child, stderr=subprocess.STDOUT, env=ppu_env(), check=False)
                summary = new_seed / 'activation_audit' / 'activation_summary.json'
                result = {'seed': seed, 'returncode': proc.returncode, 'child_log': str(child_log), 'summary': str(summary), 'summary_exists': summary.is_file(), 'status': 'COMPLETED' if proc.returncode == 0 and summary.is_file() else 'FAILED'}
                results.append(result)
                atomic_json(run_dir / 'stage3_results.json', results)
                emit(outer, 'seed阶段3结束', **result)
                if result['status'] != 'COMPLETED':
                    raise RuntimeError(f'seed{seed} 阶段3失败，停止后续 seed；returncode={proc.returncode}')
            final = {'stage': 'stage3_activation_audit', 'status': 'COMPLETED', 'time': now(), 'run_time': run_time, 'results': results, 'elapsed_seconds': time.time() - start}
            atomic_json(run_dir / 'stage_status.json', final)
            atomic_json(run_dir / 'heartbeat.json', final)
            atomic_json(run_dir / 'FINAL_DECISION.json', {'run_time': run_time, 'stage3': 'COMPLETED', 'stage4': 'NOT_RUN', 'stage5': 'NOT_RUN', 'next_stage': 'PENDING_STAGE3_ANALYSIS'})
            emit(outer, '恢复阶段3全部完成', elapsed_seconds=time.time() - start)
        except Exception as exc:
            failure = {'stage': 'stage3_activation_audit', 'status': 'FAILED', 'time': now(), 'type': type(exc).__name__, 'message': str(exc), 'results': results}
            atomic_json(run_dir / 'failure_events.json', {'events': [failure]})
            atomic_json(run_dir / 'stage_status.json', failure)
            atomic_json(run_dir / 'heartbeat.json', failure)
            atomic_json(run_dir / 'FINAL_DECISION.json', {'run_time': run_time, 'stage3': 'FAILED', 'stage4': 'NOT_RUN', 'stage5': 'NOT_RUN', 'next_stage': 'NO_GO', 'reason': str(exc)})
            emit(outer, '恢复阶段3失败', **failure)
            raise


if __name__ == '__main__':
    main()
