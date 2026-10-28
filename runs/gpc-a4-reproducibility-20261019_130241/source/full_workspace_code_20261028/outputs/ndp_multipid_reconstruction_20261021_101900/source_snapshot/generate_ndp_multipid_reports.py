#!/usr/bin/env python3
"""从本轮结构化产物生成中文审计报告；不读取官方 query/gallery/test。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


RUN = "20261021_101900"
ROOT = Path("/home/DIFFER")
OUT = ROOT / f"outputs/ndp_multipid_reconstruction_{RUN}"
REPORT = ROOT / "reports"
REPORT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def source_hashes():
    names = [
        "scripts/ndp_trajectory_feature_export.py",
        "scripts/ndp_pid_level_intrusion_audit.py",
        "scripts/ndp_historical_relation_reliability.py",
        "scripts/ndp_reactive_nonredundancy_audit.py",
        "scripts/ndp_relation_coverage_bias_audit.py",
        "scripts/ndp_reactive_prototype.py",
        "scripts/ndp_aa_operator_probe.py",
        "scripts/ndp_aa_operator_analysis.py",
        "tests/test_ndp_reactive_prototype.py",
        "tests/test_ndp_aa_operator_analysis.py",
    ]
    return {n: sha(ROOT / n) for n in names if (ROOT / n).exists()}


def baseline_hashes():
    paths = {
        "/data/DIFFER/scripts/gpc_v2_train.py": Path("/data/DIFFER/scripts/gpc_v2_train.py"),
        "/data/DIFFER/loss/make_loss.py": Path("/data/DIFFER/loss/make_loss.py"),
        "/data/DIFFER/configs/ltcc/eva02_l_bio.yml": Path("/data/DIFFER/configs/ltcc/eva02_l_bio.yml"),
    }
    return {name: sha(path) for name, path in paths.items() if path.exists()}


def md(name: str, text: str):
    (REPORT / f"{name}_{RUN}.md").write_text(text.rstrip() + "\n", encoding="utf-8")


def main():
    fe = {s: read_json(OUT / f"feature_export/seed{s}/feature_export_manifest.json") for s in (1234, 4321)}
    rep = {s: read_json(OUT / f"feature_export/seed{s}/advantage_reproduction_summary.json") for s in (1234, 4321)}
    pid = {s: read_json(OUT / f"pid_intrusion/seed{s}/pid_level_summary_seed{s}.json") for s in (1234, 4321)}
    hist = {s: read_json(OUT / f"historical_reliability/seed{s}/historical_relation_summary.json") for s in (1234, 4321)}
    react = {s: read_json(OUT / f"reactive_nonredundancy/seed{s}/reactive_nonredundancy_summary.json") for s in (1234, 4321)}
    cov = {s: read_json(OUT / f"coverage_bias/seed{s}/relation_coverage_summary.json") for s in (1234, 4321)}

    common = f"""\n\n本报告属于 RUN_TIME `{RUN}`，所有模型计算仅使用 PPU-ZW810E 的 `eval()+no_grad` 或无 optimizer.step 反向探针；未读取官方 query/gallery/test，未修改 GPC-v1。\n\n实际远端 main：`ef18089486f340e11dfc52949e5bc8b6bc8d58d2`；服务器 `/data/DIFFER` 本地 HEAD：`573b0502a231568d509fe3a463664c1a3c42e258`，且工作树原有 dirty，故本轮新增代码只在 `/home/DIFFER` 保存，未将 dirty 工作树误当作远端版本。\n\n固定基线文件与新增代码 SHA256：\n\n```json\n{json.dumps({'baseline': baseline_hashes(), 'new_code': source_hashes()}, ensure_ascii=False, indent=2)}\n```\n"""

    md("NDP_TRAJECTORY_FEATURE_EXPORT_AUDIT", f"""# NDP 同轨迹逐节点特征导出审计

## 结论

两个 seed 的固定诊断集合均已从同一条 Bio+GPC continuation 的 update 0/20/50/100/200/400 节点导出逐图像 FP32 归一化 `feat[0]`。这是“扩大固定集合”的真实逐样本特征，不是全量 9576 张图像特征库。

| seed | 固定 anchor | 唯一图像 | 节点 | NPZ | 旧 advantage 行数 | 最大/均值/P95复算误差 |
|---:|---:|---:|---|---|---:|---:|
|1234|{fe[1234]['groups']}|{fe[1234]['image_count']}|{fe[1234]['updates']}|`{fe[1234]['npz']}`|{rep[1234]['rows_matched']}|{rep[1234]['max_abs_error']}/{rep[1234]['mean_abs_error']}/{rep[1234]['p95_abs_error']}|
|4321|{fe[4321]['groups']}|{fe[4321]['image_count']}|{fe[4321]['updates']}|`{fe[4321]['npz']}`|{rep[4321]['rows_matched']}|{rep[4321]['max_abs_error']}/{rep[4321]['mean_abs_error']}/{rep[4321]['p95_abs_error']}|

两组复算均 `rows_unmatched=0`，误差全为 0，说明旧的 LogMeanExp advantage 公式和新导出特征可逐条对齐。update0 重复编码误差也为 0。首轮 seed1234 曾在复算函数漏调用 `act.advantage` 时退出码 1，真实 traceback 保存在 `log_NDP/ndp_multipid_reconstruction_{RUN}/feature_export_seed1234_retry0_traceback.log`；修复后只重跑该导出任务。

## 资产与边界

节点参数 SHA256、groups SHA256、图像索引 SHA256 写入各 seed 的 `feature_export_manifest.json`。来源 continuation manifest 标记 `gpc_v1_preserved=true`、`optimizer_step_count=400`、`cpu_model_fallback=false`。文件名 `gpc_v2_train.py` 只是真实训练引擎文件名，不能单独作为方法版本判断；本轮沿用 B0/GPC-v1 continuation 的 manifest 证据。

没有导出全量训练集，因此不能把本结果外推为全量局部邻域覆盖；但已满足本轮固定 159/137 anchor 集合的逐节点分解要求。""" + common)

    interval_lines = []
    for s in (1234, 4321):
        for k, v in sorted(pid[s]["intervals"].items()):
            interval_lines.append(f"|{s}|{k}|{v['rows']}|{v['new_error_pid']}|{v['recovered_pid']}|")
    md("NDP_PID_LEVEL_INTRUSION_AUDIT", f"""# NDP 多 PID 局部身份侵入审计

## 结果

脚本 `ndp_pid_level_intrusion_audit.py` 固定使用启动前 groups；同一负 PID 的多张图片先按 `max similarity` 得到一个 `v_ic`，不会把多张图片重复计成多个身份。`M_ic=Q25(P_i)-max_c(sim(i,N_c))`，新增错误 PID 严格定义为 `M_ic(t)>0` 且 `M_ic(t+1)<=0`。

| seed | anchor | 固定负 PID关系 | 关系行 | 正确→错误 | 错误→正确 | 正确→正确 | 错误→错误 |
|---:|---:|---:|---:|---:|---:|---:|---:|
|1234|{pid[1234]['anchor_count']}|{pid[1234]['negative_pid_count']}|{pid[1234]['pid_level_rows']}|{pid[1234]['state_counts']['correct_to_wrong']}|{pid[1234]['state_counts']['wrong_to_correct']}|{pid[1234]['state_counts']['correct_to_correct']}|{pid[1234]['state_counts']['wrong_to_wrong']}|
|4321|{pid[4321]['anchor_count']}|{pid[4321]['negative_pid_count']}|{pid[4321]['pid_level_rows']}|{pid[4321]['state_counts']['correct_to_wrong']}|{pid[4321]['state_counts']['wrong_to_correct']}|{pid[4321]['state_counts']['correct_to_correct']}|{pid[4321]['state_counts']['wrong_to_wrong']}|

| seed | 区间 | 关系数 | 新增错误 PID | 恢复正确 PID |
|---:|---|---:|---:|---:|
{chr(10).join(interval_lines)}

seed1234 的 159 anchor 实际每个只有 1 个负 PID，不能声称该 seed 已完成多身份邻域风险的充分检验；seed4321 只有 1 个 anchor 含 2 个负 PID，其余也基本是单负 PID。因此当前结果能证明“按 PID 去重的固定关系统计实现正确”，不能证明多 PID 软侵入在大覆盖上成立。

跨衣/摄像头字段写入逐关系 CSV：`positive_cross_clothes`、`negative_cross_clothes_count`、`positive_cross_camera_count`、`negative_cross_camera_count`。本轮不把这些分组差异解释为因果训练效果。""" + common)

    hist_lines = []
    for s in (1234, 4321):
        h = hist[s]
        hist_lines.append(f"|{s}|{h['classes']['A_STABLE_CORRECT']}|{h['classes']['B_CORRECT_NEAR_OR_VARIABLE']}|{h['classes']['C_HISTORY_NOT_RELIABLE']}|{h['event_by_update_and_class']}|")
    md("NDP_HISTORICAL_RELATION_RELIABILITY", f"""# NDP 历史关系可靠性审计

可靠性分层只使用节点 `0..t` 的历史前缀，未来节点只用于离线标签。预注册阈值为：A 类所有历史 margin>0、最小 margin≥0.05 且历史标准差≤0.05；B 类历史均正确但接近边界或波动较大；C 类历史已出现错误。该分层不修改 GPC。

| seed | A 稳定正确行 | B 正确但近/波动行 | C 历史不可靠行 |
|---:|---:|---:|---:|
|1234|{hist[1234]['classes']['A_STABLE_CORRECT']}|{hist[1234]['classes']['B_CORRECT_NEAR_OR_VARIABLE']}|{hist[1234]['classes']['C_HISTORY_NOT_RELIABLE']}|
|4321|{hist[4321]['classes']['A_STABLE_CORRECT']}|{hist[4321]['classes']['B_CORRECT_NEAR_OR_VARIABLE']}|{hist[4321]['classes']['C_HISTORY_NOT_RELIABLE']}|

在 200→400 的新增错误事件中，seed1234 的 A/B/C 分别为 0/0/5；seed4321 分别为 0/2/1。这个结果支持“单次当前正确”不等于稳定可靠，也没有支持 A 类稳定关系在本小样本中发生后续新增侵入。它只是观察性分层，样本仍是固定困难集合，不足以成为训练门控。

逐行结果：`historical_relation_rows.csv`；摘要：`historical_relation_summary.json`。""" + common)

    rlines = []
    for s in (1234, 4321):
        for r in react[s]["results"]:
            rlines.append(f"|{s}|{r['method']}|{r['n_relation']}|{r['active_relations']}|{r['loss']:.6g}|{r['gradient_cosine_vs_T0']}|")
    md("NDP_REACTIVE_NONREDUNDANCY_AUDIT", f"""# NDP 反应式候选非冗余性审计

本审计使用固定特征边界的 200→400 转移，不执行网络 optimizer.step。T0/T1 是固定关系 proxy：没有逐步真实 Batch-Hard miner 记录，不能把它们写成完整 Batch-Hard 训练梯度。

| seed | 候选 | 关系数 | 激活数 | loss | 与T0梯度余弦 |
|---:|---|---:|---:|---:|---:|
{chr(10).join(rlines)}

T0 与 T1 在固定关系集合上完全相同，证明固定-margin hinge 与该离线 proxy 不提供额外信息。T2 是参考 margin 自适应 hinge，T3 是历史 A 类关系的单侧版本；两者相对 T0 的梯度余弦约为 seed1234 的 0.714/0.495、seed4321 的 0.669/0.482，但这只是冻结边界张量梯度，不是 EVA02 参数更新因果证据。当前实现的 T4 在该数据中与 T3 关系集合相同，T5 仅因纳入所有固定负 PID关系而扩大集合；不能据此宣布双侧或多 PID的独立算法增益。

`reactive_boundary_loss` 已修复全 NaN/Inf 行通过 `sum()*0` 产生 NaN 的问题；单元测试验证参考 detach、改善不惩罚、恶化有负方向梯度、空关系和非有限输入均安全。""" + common)

    clines = []
    for s in (1234, 4321):
        c = cov[s]
        clines.append(f"|{s}|{c['anchor_count']}|{c['pid_count']}|{c['image_count']}|{c['negative_pid_mean_per_anchor']:.3f}|{c['negative_pid_max_per_anchor']}|{c['multi_negative_pid_anchor_count']}|{c['cross_clothes_positive_anchor_count']}|{c['cross_camera_positive_anchor_count']}|")
    md("NDP_RELATION_COVERAGE_AND_BIAS_AUDIT", f"""# NDP 关系覆盖与抽样偏差审计

| seed | anchor | PID | image | 平均负PID/anchor | 最大负PID | 多负PID anchor | 有跨衣正样本 | 有跨相机正样本 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(clines)}

seed1234 的固定集合每个 anchor 只有 1 个负 PID；seed4321 只有 1 个 anchor 有 2 个负 PID。因此本轮虽然已正确保存逐样本特征并可按 PID 分解，但“多身份”覆盖仍严重不足。当前集合是历史困难/固定关系集合，代表性按 PID×衣服×相机的独立分层集合不存在；结果不能外推全量 LTCC，也不能把困难集合的事件率当作总体事件率。

下一次如继续，必须在不使用未来节点的前提下预注册一个代表性集合，并与困难集合分开报告。""" + common)

    aa_rows = []
    for s in (1234, 4321):
        p = OUT / f"ppu_same_precision_aa/seed{s}_comparison.csv"
        with p.open(encoding="utf-8", newline="") as fh:
            aa_rows.extend(list(csv.DictReader(fh)))
    aa_amp = [r for r in aa_rows if r["comparison"] == "amp" and r["group"] == "unscaled_gradients"]
    aa_summary = []
    for s in (1234, 4321):
        rr = [r for r in aa_amp if int(r["seed"]) == s]
        worst = max(rr, key=lambda r: float(r["relative_l2_to_a"]))
        aa_summary.append(f"|{s}|{worst['name']}|{float(worst['relative_l2_to_a']):.6g}|{float(worst['cosine']):.12f}|{float(worst['max_abs']):.6g}|")
    md("NDP_PPU_SAME_PRECISION_AA_AUDIT", f"""# PPU 同精度 A/A 无 optimizer.step 审计

同一物化 batch、同一关系 hash、同一 checkpoint、AMP、同一 PPU 环境下，每个 seed 独立进程重复两次 forward/backward；四次探针均 `PASS`、`found_inf=0`、`optimizer_step_executed=false`，loss 和输入/关系 hash一致。

| seed | 最差抽样参数 | 重复反向相对L2差 | 余弦（float64） | 最大绝对差 |
|---:|---|---:|---:|---:|
{chr(10).join(aa_summary)}

差异不是 AMP skip 造成：同一 AMP 精度、无 step、found_inf=0 仍有梯度张量哈希差异。差异最大出现在本探针保存的早期 `blocks.0.attn.proj.weight`/`q_proj` 和 patch embedding，block22 q_proj 差异较小但仍非零。由于只保存了主要模块和输出梯度，没有对每个底层 kernel 建立同上游梯度的最小复现，因此根因仍不能定位到某一个 norm、Q/K/V、matmul 或 softmax kernel。

余弦实现已改为 float64 点积/范数并同时保存 `cosine_raw` 与有界 `cosine`；本轮超界问题不再由未经检查的 float32 统计产生。该结果仍不构成 A/A PASS：同精度反向不完全重复，尚未建立可接受的预注册噪声界。

旧的 block22 观察与本轮全模型重复探针并不矛盾：前者是另一套更窄的算子/模块观察；本轮证据只支持“全模型反向存在可重复的微小非确定性”，不支持精确根因定位。""" + common)

    md("NDP_NEXT_STAGE_RESEARCH_DECISION", f"""# NDP 下一阶段研究决策

## 状态

```text
FEATURE_EXPORT = PASS_FIXED_SUBSET_ONLY
PID_LEVEL_DECOMPOSITION = PASS_WITH_SEVERE_MULTI_PID_COVERAGE_LIMIT
HISTORICAL_RELIABILITY = PASS_PREFIX_ONLY
REACTIVE_NONREDUNDANCY = UNIT_TEST_AND_FROZEN_BOUNDARY_ONLY
PPU_SAME_PRECISION_AA = FAIL
NDP_MECHANISM = INCONCLUSIVE
CONTROLLED_SHORT_PILOT = NO_GO
```

## 已验证事实

1. 两个 seed 的 0/20/50/100/200/400 节点均可对固定 382/340 图像集合导出逐样本特征，且旧 LogMeanExp advantage 954/822 行逐条零误差复现。
2. 固定集合中 seed1234 没有真正的多负 PID anchor，seed4321 只有 1 个；因此本轮不能把多 PID 侵入机制当作已验证。
3. 历史前缀可靠性可以无未来泄漏地分层；200→400 的新增错误事件并未出现在 A 稳定类中，但样本量和抽样偏差不足以推广。
4. 反应式 NDP 与固定关系 margin hinge 的差异主要来自历史可靠性掩码、参考 margin 和关系集合，不是已证明的独立网络监督；T0/T1 完全相同，T4/T3 在当前集合也相同。
5. 同精度 PPU 反向重复仍有相对 L2 约 0.3%–0.47% 的主要参数梯度差异，虽余弦接近 1，但足以维持 A/A FAIL。没有进行 optimizer-step。

## 决策

当前不申请 NDP 短程训练，不启动 30/60 epoch。最小可保留方向是“历史前缀可靠性 + 固定多 PID/双侧关系的反应式诊断”，但它还没有超出 Hard Triplet 的因果独立价值证据。下一步优先级不是再加 Loss，而是：扩充预注册代表性多 PID 关系、获取逐步真实 Batch-Hard miner 覆盖记录、并用同上游梯度的 PPU 最小算子复现定位 A/A。

不要把本轮冻结特征梯度、自然轨迹关系退化或固定集合统计解释为 Rank-1/mAP收益。""" + common)

    manifest = {
        "run_time": RUN, "status": "COMPLETED_OFFLINE_AND_NO_STEP_PPU_PROBES",
        "reports": [f"{n}_{RUN}.md" for n in ["NDP_TRAJECTORY_FEATURE_EXPORT_AUDIT", "NDP_PID_LEVEL_INTRUSION_AUDIT", "NDP_HISTORICAL_RELATION_RELIABILITY", "NDP_REACTIVE_NONREDUNDANCY_AUDIT", "NDP_RELATION_COVERAGE_AND_BIAS_AUDIT", "NDP_PPU_SAME_PRECISION_AA_AUDIT", "NDP_NEXT_STAGE_RESEARCH_DECISION"]],
        "feature_export": fe, "pid_intrusion": pid, "historical": hist, "reactive": react, "coverage": cov,
        "aa_same_precision": {s: str(OUT / f"ppu_same_precision_aa/seed{s}_comparison.json") for s in (1234, 4321)},
        "source_hashes": source_hashes(), "baseline_hashes": baseline_hashes(), "official_query_gallery_used": False,
        "optimizer_step_executed": False, "gpc_modified": False, "controlled_short_pilot": "NO_GO",
    }
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_REPORTS_GENERATED", "run_time": RUN, "reports": len(manifest["reports"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
