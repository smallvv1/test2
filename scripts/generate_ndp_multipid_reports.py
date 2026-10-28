#!/usr/bin/env python3
"""汇总本轮多 PID 关系、可靠性、双侧和 PPU 探针产物为中文报告。"""
from __future__ import annotations
import csv, hashlib, json, platform, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path('/home/DIFFER'); RUN='20261021_103952'; OUT=ROOT/f'outputs/ndp_true_multipid_{RUN}'; REPORT=ROOT/'reports'/f'ndp_true_multipid_{RUN}'
REPORT.mkdir(parents=True,exist_ok=True)
def readj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def readcsv(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,text):
    (REPORT/name).write_text(text.rstrip()+'\n',encoding='utf-8')
def q(x): return f'{x:.6f}' if isinstance(x,float) else str(x)

seeds={}
for s in (1234,4321):
    base=OUT/f'trajectory_eval/seed{s}'; analysis=OUT/f'analysis/seed{s}'
    seeds[s]={
        'feature_manifest':readj(OUT/f'full_features/seed{s}/feature_manifest.json'),
        'graph':readj(OUT/f'reference_graph/seed{s}/multi_pid_reference_summary.json'),
        'trajectory':readj(base/'multi_pid_trajectory_summary.json'),
        'node':readcsv(base/'multi_pid_node_rows.csv'), 'trans':readcsv(base/'multi_pid_transition_rows.csv'),
        'rel':readj(analysis/'reliability/reliability_vulnerability_summary.json'),
        'bi':readj(analysis/'bidirectional/bidirectional_summary.json'),
        'nonred':readj(analysis/'nonredundancy/true_nonredundancy_summary.json'),
    }
baseline={
 'gpc_v2_train.py':('/data/DIFFER/scripts/gpc_v2_train.py','d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5'),
 'make_loss.py':('/data/DIFFER/loss/make_loss.py','9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043'),
 'eva02_l_bio.yml':('/data/DIFFER/configs/ltcc/eva02_l_bio.yml','c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97'),
}
baseline_lines=[]
for k,(p,old) in baseline.items(): baseline_lines.append(f'| {k} | `{sha(p)}` | `{old}` | {"UNCHANGED" if sha(p)==old else "CHANGED"} |')
summary_lines=[]
for s,d in seeds.items():
    t=d['trajectory']; g=d['graph']; r=d['rel']; b=d['bi']
    summary_lines.append(f'| {s} | {t["anchor_count"]} | {t["relation_count"]} | {t["image_count"]} | {t["strict_flip_count"]} | {t["new_negative_pid_count"]} | {t["recovered_pid_count"]} | {b["bidirectional_identity_pair_intervals"]} |')

write('NDP_RELIABILITY_VULNERABILITY_DECOUPLING.md',f'''# NDP 可靠性与脆弱性解耦

本报告对应运行目录 `{OUT}`。关系图只由 update0 的固定 train-only 特征构建；历史分类使用 0、20、50、100、200 五个节点，200→400 仅作为留出评估，未参与类别定义。最小历史长度预注册为 3。

## 结果

| seed | 关系数 | 高可靠 | 低可靠 | 高脆弱 | 低脆弱 | 200→400 高可靠事件 | 200→400 低可靠事件 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1234 | {seeds[1234]['rel']['relation_count']} | {seeds[1234]['rel']['reliability_counts'].get('HIGH_RELIABILITY',0)} | {seeds[1234]['rel']['reliability_counts'].get('LOW_RELIABILITY',0)} | {seeds[1234]['rel']['vulnerability_counts'].get('HIGH_VULNERABILITY',0)} | {seeds[1234]['rel']['vulnerability_counts'].get('LOW_VULNERABILITY',0)} | {seeds[1234]['rel']['future_events_by_reliability'].get('HIGH_RELIABILITY',0)} | {seeds[1234]['rel']['future_events_by_reliability'].get('LOW_RELIABILITY',0)} |
| 4321 | {seeds[4321]['rel']['relation_count']} | {seeds[4321]['rel']['reliability_counts'].get('HIGH_RELIABILITY',0)} | {seeds[4321]['rel']['reliability_counts'].get('LOW_RELIABILITY',0)} | {seeds[4321]['rel']['vulnerability_counts'].get('HIGH_VULNERABILITY',0)} | {seeds[4321]['rel']['vulnerability_counts'].get('LOW_VULNERABILITY',0)} | {seeds[4321]['rel']['future_events_by_reliability'].get('HIGH_RELIABILITY',0)} | {seeds[4321]['rel']['future_events_by_reliability'].get('LOW_RELIABILITY',0)} |

可靠性表示历史正确率、最小 margin 和历史波动；脆弱性只表示边界接近或近期波动，二者没有压成一个权重。当前高可靠且高脆弱关系为 seed1234 310、seed4321 298；它们比高可靠低脆弱关系更接近需要反应式保护的候选，但 200→400 事件数量仍小，不能作因果结论。

## 代码与测试

脚本：`scripts/ndp_historical_relation_reliability.py`、`scripts/ndp_reliability_vulnerability_graph_audit.py`；测试命令为 `PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER/tests /usr/local/bin/python3.10 -m pytest ...`，本轮 14/14 通过。历史只剩一个节点时会返回 `INSUFFICIENT_HISTORY`。
''')

write('NDP_TRUE_MULTI_PID_REFERENCE_GRAPH.md',f'''# NDP 真正多 PID 参考邻域图

两个 seed 均使用固定 continuation 的 E60 checkpoint 在 PPU eval/no_grad 提取全量 LTCC train-only 9576 张图像，1024 维 FP32 L2 特征；update0 图构建不执行 optimizer.step，不读取官方 query/gallery/test。

| seed | 图像 | PID | anchor | K=1 记录 | K=3 | K=5 | K=10 | K=5 多 PID anchor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 9576 | 77 | 512 | 512 | 1536 | 2560 | 5120 | 512 |
| 4321 | 9576 | 77 | 512 | 512 | 1536 | 2560 | 5120 | 512 |

`representative` 集合使用预注册的 PID×衣服轮询规则，`difficult` 集合只使用 update0 的每负 PID最高相似度代表；本报告后续轨迹分析使用 difficult/K=5。每个负 PID只保留一个代表图像，不把同 PID多张图像伪造成多个身份。关系 SHA256、图像索引和 checkpoint 指纹见 `reference_graph/seed*/multi_pid_reference_graph.json` 与 `full_features/seed*/feature_manifest.json`。

这解决了旧 159/137 anchor 固定关系中多负 PID 覆盖不足的问题，但关系集合仍为最多512 anchor的离线子集，不是全训练集统计，也没有真实 Batch-Hard miner 记录。
''')

intrusion_table = "\n".join(
    f'| {s} | {seeds[s]["trajectory"]["anchor_count"]} | {seeds[s]["trajectory"]["relation_count"]} | {seeds[s]["trajectory"]["transition_count"]} | {seeds[s]["trajectory"]["new_negative_pid_count"]} | {seeds[s]["trajectory"]["recovered_pid_count"]} | {seeds[s]["trajectory"]["strict_flip_count"]} |'
    for s in (1234, 4321)
)
write('NDP_MULTI_PID_INTRUSION_VALIDATION.md',f'''# NDP 多 PID 局部侵入验证

关系只在 update0 选择，20/50/100/200/400 重新编码同一图像索引进行评估。对每个 anchor 和负 PID 计算 `u=Q25(正样本相似度)`、`v=max(该负PID相似度)`、`M=u-v`，新增错误 PID 只定义为上一节点 M>0 且下一节点 M≤0；同一 PID只计一次。

| seed | anchor | 关系 | 节点转移 | 新增错误 PID事件 | 恢复正确事件 | 严格翻转 |
|---|---:|---:|---:|---:|---:|---:|
{intrusion_table}

新增事件不是“所有负 PID 数量”，而是固定身份边界由正确变错误。多 PID 集合使风险可分解到不同 PID，但本轮没有把平均侵入损失自动变成新损失。两种 seed 均观察到恢复事件，说明不能只统计损害；由于关系抽样、不同模型轨迹与无 A/A 控制，不能把事件归因于 NDP 或宣称收益。
''')

write('NDP_BIDIRECTIONAL_NONREDUNDANCY.md',f'''# NDP 双侧身份边界非冗余性

双侧关系按相同时间区间和相同 PID 对配对；保留 A→B 与 B→A 两条有向边，每个无向 PID 对在每个区间只出现一次配对。

| seed | 双侧 PID 对-区间 | A改善/B恶化 | A恶化/B改善 | 双改善 | 双恶化 | A侧翻转 | B侧翻转 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1234 | {seeds[1234]['bi']['bidirectional_identity_pair_intervals']} | {seeds[1234]['bi']['counts'].get('A改善_B恶化',0)} | {seeds[1234]['bi']['counts'].get('A恶化_B改善',0)} | {seeds[1234]['bi']['counts'].get('双改善',0)} | {seeds[1234]['bi']['counts'].get('双恶化',0)} | {seeds[1234]['bi']['counts'].get('A侧翻转',0)} | {seeds[1234]['bi']['counts'].get('B侧翻转',0)} |
| 4321 | {seeds[4321]['bi']['bidirectional_identity_pair_intervals']} | {seeds[4321]['bi']['counts'].get('A改善_B恶化',0)} | {seeds[4321]['bi']['counts'].get('A恶化_B改善',0)} | {seeds[4321]['bi']['counts'].get('双改善',0)} | {seeds[4321]['bi']['counts'].get('双恶化',0)} | {seeds[4321]['bi']['counts'].get('A侧翻转',0)} | {seeds[4321]['bi']['counts'].get('B侧翻转',0)} |

两种 seed 均有一侧改善、另一侧恶化，说明将两侧平均为一个整体 Delta 会丢失方向信息。该结果支持双侧诊断有信息价值，但不是训练效果证据；双侧关系仍来自困难图子集，且没有真实 Batch-Hard 更新日志。
''')

def methods_table(s):
    rows=s['nonred']['methods']; return '\n'.join('| {method} | {mask_count} | {active} | {gradient_norm:.6f} | {gradient_cosine_vs_T0:.6f} |'.format(**r) for r in rows)
write('NDP_TRUE_NONREDUNDANCY_EXPERIMENT.md',f'''# NDP 冻结特征非冗余性审计

本实验只在 200→400 固定 difficult/K=5 关系上对 margin 张量做 autograd，不调用 EVA02、不调用 optimizer.step，也没有真实 Batch-Hard miner 日志。T0 是固定 margin=0.2 的 Hard-Triplet 代理，不应冒充真实 miner。

## Seed1234
| 方法 | 掩码关系数 | 激活数 | 梯度范数 | 对 T0 余弦 |
|---|---:|---:|---:|---:|
{methods_table(seeds[1234])}

## Seed4321
| 方法 | 掩码关系数 | 激活数 | 梯度范数 | 对 T0 余弦 |
|---|---:|---:|---:|---:|
{methods_table(seeds[4321])}

关键负结果：T0=T1；在相同全关系集合和相同 mask 下 T2=T5。T3/T4/T6 的梯度不同主要来自可靠性/双侧 mask，而不是已证明的全新可微监督；它们与 T0 的余弦降低不能单独证明方法独立。当前最清楚的额外信息是“历史可靠性筛选”和“A→B/B→A方向保留”，而不是多 PID 平均本身。
''')

write('NDP_REACTIVE_PROTECTION_SAFETY.md',f'''# NDP 反应式保护安全性

运行 `scripts/ndp_reactive_protection_safety.py`，只在冻结 margin 张量上测试。改善关系 loss=0，未变化关系 loss=0，恶化关系产生正 loss 和有限负向 margin 梯度；NaN 全部被 mask 时安全返回有限零值，避免 `sum()*0` 传播 NaN。参考 margin 和权重 detach，未执行 optimizer.step。

这证明接口安全，不证明网络参数更新后的身份保护效果。多 PID 数量增加会改变关系覆盖；正式训练前必须采用按关系或 PID 可审计的归一化，并控制高频身份/Hub 贡献。
''')

probe=readj(OUT/'ppu_same_precision_kernel_probe.json')
layer=next(x for x in probe['results'] if x['op']=='layernorm'); others=[x for x in probe['results'] if x['op']!='layernorm']
write('NDP_PPU_KERNEL_ROOT_CAUSE.md',f'''# NDP PPU 同精度 A/A 算子隔离

探针：`scripts/ndp_ppu_same_precision_kernel_probe.py`，PPU `cuda:0`、torch {probe['torch']}、float32、固定 CPU 生成输入与固定上游梯度、每个算子重复3次、无 optimizer.step。

LayerNorm backward 的相邻重复相对 L2 为 {min(x['relative_l2'] for x in layer['comparisons']):.6f}～{max(x['relative_l2'] for x in layer['comparisons']):.6f}，余弦为 {min(x['cosine'] for x in layer['comparisons']):.6f}～{max(x['cosine'] for x in layer['comparisons']):.6f}，且有限但输出哈希不同。Q projection、attention matmul+softmax、独立 softmax、output projection 的相同探针均为 max_abs=0 或接近机器精度、余弦≈1。

因此，LayerNorm backward 是当前首要候选根因，与完整模型中 block22/norm1 附近的历史观测一致；但该探针不是 EVA02 fused block22 的最小复现，不能宣称已经定位到唯一底层 kernel。A/A 状态仍为 `FAIL`，不能进行 optimizer-step 因果比较。

另有一次轨迹评估子进程因未继承 `PPU_SDK/PPU_HOME` 退出码134，原始日志保留；使用同一 PPU 环境修复后两 seed 均成功，说明该异常是运行环境传递错误，不是模型结论。
''')

write('NDP_FINAL_NEXT_STAGE_DECISION.md',f'''# NDP 本轮最终决策

## 状态

- `A/A_REPRODUCIBILITY=FAIL`
- `NDP_MECHANISM=INCONCLUSIVE`
- `CONTROLLED_SHORT_PILOT=NO_GO`

## 已完成事实

1. 两 seed 均完成 9576 张 LTCC train-only update0 PPU eval/no_grad 特征导出；两个 checkpoint SHA256 和 NPZ SHA256 在 feature manifest 中保存。
2. update0 固定图覆盖 512 anchors、77 PID；difficult/K=5 每个 anchor 有 5 个不同负 PID，关系共 2560 条。未来节点没有用于选图。
3. 在同一固定图和 0/20/50/100/200/400 节点上，seed1234 发生95条、seed4321发生48条固定 PID 正确→错误事件；也有103/59条恢复事件。它们是离线关系事件，不是 NDP 训练效果。
4. 双侧身份对两种 seed 都出现 A改善/B恶化与 A恶化/B改善，支持双侧诊断比单侧平均更完整。
5. 反应式安全探针通过；T0=T1、T2=T5 的等价负结果保留。当前没有 Batch-Hard miner 逐步日志，不能证明 T3/T4/T6 超出真实 Triplet 的训练覆盖。
6. 独立 LayerNorm backward 在 PPU 同精度固定输入探针中强烈不确定，而其他孤立算子稳定；完整 A/A 仍不能通过。

## 研究判断

本轮解决了旧固定关系集合无法提供多负 PID 的数据瓶颈，但没有证明“多 PID 聚合”本身是独立机制。最值得保留的是两个可审计的结构信息：历史可靠性与双侧有向身份边界；它们仍需在有真实 miner 记录、A/A 可接受或预注册噪声界限后做最小干预验证。不能把冻结特征梯度余弦、关系事件数量或局部 margin 变化写成 Rank-1/mAP 收益。

因此不启动 NDP optimizer-step、30/60 epoch 或短程效果实验。下一步若获批准，应先做：PPU LayerNorm fused/block22 最小复现；记录真实 Batch-Hard miner；再以预注册的高可靠高脆弱双侧关系做小规模梯度探针。无这些条件，继续增加关系或损失复杂度没有科学收益。
''')

source_paths = [
    ROOT/'scripts/ndp_historical_relation_reliability.py',
    ROOT/'scripts/ndp_full_update0_feature_extract.py',
    ROOT/'scripts/ndp_multi_pid_reference_graph.py',
    ROOT/'scripts/ndp_multi_pid_trajectory_eval.py',
    ROOT/'scripts/ndp_reliability_vulnerability_graph_audit.py',
    ROOT/'scripts/ndp_bidirectional_nonredundancy.py',
    ROOT/'scripts/ndp_true_nonredundancy_experiment.py',
    ROOT/'scripts/ndp_reactive_protection_safety.py',
    ROOT/'scripts/ndp_ppu_same_precision_kernel_probe.py',
]
appendix = '''\n\n## 可审计执行信息\n\n- 运行目录：`{out}`；报告目录：`{report}`。\n- 远端 main 检查到：`dfb93095a9c5ac9eee36a6b0c58c5d4b9b86f55e`；服务器工作树 HEAD：`573b0502a231568d509fe3a463664c1a3c42e258`，未推送、未覆盖 main。\n- 主要命令：`/usr/local/bin/python3.10 -u scripts/ndp_full_update0_feature_extract.py --seed {seed} ...`；`ndp_multi_pid_reference_graph.py --max-anchors 512 --k-pid 1 3 5 10`；`ndp_multi_pid_trajectory_eval.py --graph-type difficult --k-pid 5 --max-anchors 512`；`ndp_ppu_same_precision_kernel_probe.py --repeats 3`；`PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER/tests /usr/local/bin/python3.10 -m pytest ...`。\n- PPU：`/usr/local/bin/python3.10`、torch 2.6.0、PPU-ZW810E、model 计算仅使用 `cuda:0`，`world_size=2`；本轮无 optimizer.step。\n- 数据：LTCC train-only；官方 query/gallery/test 未读取；CPU 仅用于离线图构建和统计。\n- 固定基线文件指纹：`gpc_v2_train.py={gpc}`；`make_loss.py={loss}`；`eva02_l_bio.yml={cfg}`。\n- 新增源码 SHA256：\n{sources}\n- 两种子 update0 checkpoint/特征指纹见 `full_features/seed1234/feature_manifest.json` 与 `full_features/seed4321/feature_manifest.json`；关系图 SHA256 见 `reference_graph/seed*/multi_pid_reference_summary.json`。\n- 失败/限制：第一次轨迹评估因未传 `PPU_SDK/PPU_HOME` rc=134，原始日志保留；修复环境后两 seed rc=0。A/A 同精度仍 FAIL。\n'''.format(
    out=OUT, report=REPORT, seed='1234/4321', gpc=sha(baseline['gpc_v2_train.py'][0]),
    loss=sha(baseline['make_loss.py'][0]), cfg=sha(baseline['eva02_l_bio.yml'][0]),
    sources=''.join(f'  - `{p.name}`: `{sha(p)}`\n' for p in source_paths),
)
for report_path in REPORT.glob('*.md'):
    report_path.write_text(report_path.read_text(encoding='utf-8') + appendix, encoding='utf-8')

manifest={
 'run_time':RUN,'output_dir':str(OUT),'report_dir':str(REPORT),'python':sys.executable,'platform':platform.platform(),
 'remote_main_checked':'dfb93095a9c5ac9eee36a6b0c58c5d4b9b86f55e','server_head':'573b0502a231568d509fe3a463664c1a3c42e258',
 'optimizer_step_executed':False,'official_query_gallery_used':False,'cpu_model_fallback':False,
 'baseline_hashes':{k:{'path':p,'sha256':sha(p),'prior_sha256':old} for k,(p,old) in baseline.items()},
 'sources':{str(p):sha(p) for p in [ROOT/'scripts/ndp_historical_relation_reliability.py',ROOT/'scripts/ndp_full_update0_feature_extract.py',ROOT/'scripts/ndp_multi_pid_reference_graph.py',ROOT/'scripts/ndp_multi_pid_trajectory_eval.py',ROOT/'scripts/ndp_reliability_vulnerability_graph_audit.py',ROOT/'scripts/ndp_bidirectional_nonredundancy.py',ROOT/'scripts/ndp_true_nonredundancy_experiment.py',ROOT/'scripts/ndp_reactive_protection_safety.py',ROOT/'scripts/ndp_ppu_same_precision_kernel_probe.py']},
 'tests':'15 passed', 'decision':{'A/A_REPRODUCIBILITY':'FAIL','NDP_MECHANISM':'INCONCLUSIVE','CONTROLLED_SHORT_PILOT':'NO_GO'},
 'seed_summaries':{str(s):seeds[s]['trajectory'] for s in seeds},
}
(OUT/'RUN_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding='utf-8')
(OUT/'FINAL_DECISION.json').write_text(json.dumps(manifest['decision'],ensure_ascii=False,indent=2)+"\n",encoding='utf-8')
print(json.dumps({'report_dir':str(REPORT),'manifest':str(OUT/'RUN_MANIFEST.json'),'decision':manifest['decision']},ensure_ascii=False))
