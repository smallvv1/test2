#!/usr/local/bin/python3.10
"""封存本次 NDP 夜间运行；只读取结构化日志，不执行模型计算。"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/DIFFER')
RUN_TIME = '20261020_201841'
RUN_DIR = ROOT / 'outputs' / f'ndp_overnight_{RUN_TIME}'
REPORT_DIR = ROOT / 'reports'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def stats(seed):
    path = RUN_DIR / f'trajectory_seed{seed}' / 'trajectory_steps.csv'
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    def count(fn):
        return sum(1 for row in rows if fn(row))
    return {
        'seed': seed,
        'path': str(path),
        'sha256': sha256(path),
        'batch_rows': len(rows),
        'effective_updates': count(lambda r: r['step_executed'] == 'True'),
        'found_inf_steps': count(lambda r: r['found_inf'] not in ('0', 'False', 'false')),
        'active_gpc_batches': count(lambda r: r['active'] == 'True'),
        'active_gpc_rate': count(lambda r: r['active'] == 'True') / len(rows),
        'selected_relation_sum_rank0_log': sum(int(r['selected_count']) for r in rows),
        'nonzero_gpc_loss_batches': count(lambda r: float(r['gpc']) != 0.0),
        'all_parameters_finite': all(r['parameter_finite'] == 'True' for r in rows),
        'lr_values': sorted({r['lr'] for r in rows}),
        'scale_values': sorted({r['scale_before'] for r in rows}),
    }

def main():
    seed_stats = [stats(1234), stats(4321)]
    failure_events = {
        'run_time': RUN_TIME,
        'events': [
            {'stage': 'stage2_seed1234_initial', 'status': 'RECOVERED', 'returncode': 1, 'cause': '训练脚本使用错误的 common_init 模块路径；phase5b_id_damage_ds 没有 load_common_init，随后改用 v2.b6.load_common_init。', 'log': str(RUN_DIR / 'logs/NDP基础轨迹_seed1234_201841.log')},
            {'stage': 'stage3_seed1234_initial', 'status': 'FAILED', 'returncode': 1, 'cause': '固定采样组构造触发 /data/DIFFER/scripts/gpc_v2_train.py:218 的 KeyError: 9，未进入 EVA02 前向。', 'log': str(RUN_DIR / 'logs/NDP真实轨迹激活_seed1234_20261020_201841.log')},
            {'stage': 'stage3_seed1234_recovery1', 'status': 'FAILED', 'returncode': 1, 'cause': 'active_state 修复后仍触发 KeyError: 9；根因为 choose_groups 再次调用 install_protocol，导致 loader_v2 嵌套并对 covered_pids 二次 remap。达到两次恢复上限，停止阶段3。', 'log': str(RUN_DIR / 'logs/NDP真实轨迹激活_seed1234_20261020_201841_恢复1.log')},
        ],
    }
    write_json(RUN_DIR / 'failure_events.json', failure_events)
    write_json(RUN_DIR / 'stage_status.json', {
        'run_time': RUN_TIME, 'stage': 'stage3_activation_audit', 'status': 'BLOCKED_TECHNICAL_AFTER_TWO_RECOVERIES',
        'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'stage0': 'PASS', 'stage1': 'PASS',
        'stage2': 'PASS_BASE_CONTINUATION', 'stage3': 'BLOCKED', 'stage4': 'NOT_RUN', 'stage5': 'NOT_RUN',
        'effective_updates': {str(row['seed']): row['effective_updates'] for row in seed_stats},
        'reason': '激活审计 harness 的双重 install_protocol/loader_v2 remap 缺陷未修复；按止损规则不再重试。',
        'latest_artifact': str(RUN_DIR / 'failure_events.json'),
    })
    write_json(RUN_DIR / 'heartbeat.json', {'time': dt.datetime.now(dt.timezone.utc).isoformat(), 'stage': 'finalized', 'status': 'BLOCKED_TECHNICAL_AFTER_TWO_RECOVERIES', 'pid': None})
    write_json(RUN_DIR / 'FINAL_DECISION.json', {
        'run_time': RUN_TIME, 'TRAJECTORY_RECOVERY': 'PASS_FOR_E60_TO_NEW_CONTINUATION', 'REAL_NDP_ACTIVATION': 'NOT_VERIFIED',
        'LIVE_GRADIENT': 'PASS_FOR_PRIOR_NO_STEP_PROBE', 'PPU_AA_COMPARABILITY': 'FAIL_OR_UNRESOLVED',
        'SHORT_INTERVENTION': 'NOT_RUN', 'LOCAL_STRUCTURE_PROTECTION': 'INCONCLUSIVE', 'NEXT_STAGE': 'NO_GO',
        'scientific_reason': '阶段3未进入模型前向，不能从未生成的 activation_summary 推断 NDP 自然激活；阶段5受到真实激活、AMP可比性和训练协议门禁阻止。',
    })
    write_json(RUN_DIR / 'trajectory_summary_index.json', {'run_time': RUN_TIME, 'seed_stats': seed_stats, 'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat()})

if __name__ == '__main__':
    main()
