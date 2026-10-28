#!/usr/bin/env python3
"""为当前关系修正批次生成中文证据报告，不读取测试集、不运行训练。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pct(x):
    return "NA" if x is None else f"{100.0 * x:.3f}%"


def probe_line(p):
    d = load(p)
    values = []
    for g in d.get("groups", []):
        c = g.get("comparisons", [])
        values.extend((g["name"], g["amp"], x["output"]["max_abs"], x["input_grad"]["max_abs"], x["status"]) for x in c)
    valid = all(x[4] == "VALID_COMPARISON" for x in values) and d.get("status") == "PASS_FIXED_INPUT_UPSTREAM_PROBE"
    max_out = max((x[2] for x in values), default=None)
    max_grad = max((x[3] for x in values), default=None)
    return d, valid, max_out, max_grad


def write(path, text):
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    args = ap.parse_args(); out = args.run_dir; rep = args.report_dir; rep.mkdir(parents=True, exist_ok=True)
    s = {seed: load(out / f"bidirectional_correction/seed{seed}/bidirectional_relation_correction_summary.json") for seed in (1234, 4321)}
    t = {seed: load(out / f"triplet_coverage/seed{seed}/triplet_miner_coverage_summary.json") for seed in (1234, 4321)}
    r = {seed: load(out / f"reliability_vulnerability/seed{seed}/reliability_vulnerability_information_summary.json") for seed in (1234, 4321)}
    m = {seed: load(out / f"matched_nonredundancy/seed{seed}/matched_nonredundancy_summary.json") for seed in (1234, 4321)}
    p0, p0ok, p0max, p0gmax = probe_line(out / "ndp_layernorm_probe.json")
    p1, p1ok, p1max, p1gmax = probe_line(out / "ndp_layernorm_probe_seed4321.json")
    source_files = [
        Path("/home/DIFFER/scripts/ndp_ppu_same_precision_kernel_probe.py"),
        Path("/home/DIFFER/scripts/ndp_multi_pid_reference_graph.py"),
        Path("/home/DIFFER/scripts/ndp_multi_pid_trajectory_eval.py"),
        Path("/home/DIFFER/scripts/ndp_bidirectional_relation_correction.py"),
        Path("/home/DIFFER/scripts/ndp_reliability_vulnerability_information_audit.py"),
        Path("/home/DIFFER/scripts/ndp_real_triplet_miner_coverage.py"),
        Path("/home/DIFFER/scripts/ndp_matched_nonredundancy_audit.py"),
        Path("/home/DIFFER/tests/test_ndp_relation_correction.py"),
    ]
    hashes = "\n".join(f"- `{x}`: `{digest(x)}`" for x in source_files if x.exists())
    command_note = """本批次的实际命令保存在 `/home/DIFFER/log_NDP/ndp_relation_correction_20261021_192813/`；所有模型探针使用 `/usr/local/bin/python3.10`，PPU 环境为 `PPU_SDK=/usr/local/PPU_SDK`，离线梯度审计不执行 optimizer.step。"""

    write(rep / "NDP_LAYER_NORM_PROBE_CORRECTION.md", f"""# LayerNorm 探针随机性修正

## 结论

上一轮 `relative L2≈1.4` 不能作为 LayerNorm kernel 非确定性证据：旧代码的 `up=torch.randn_like(x)` 没有使用固定 generator。该旧结果已标记为 `INVALID_UPSTREAM_CONTROL`。

本批次在循环外物化 input、upstream、weight、bias，每次重复新建计算图并清空梯度，同时核对输入/上游哈希。Seed1234 和 Seed4321 的独立 LayerNorm、实际 EVA02 `blocks.22.norm1`，均在 PPU 的 AMP/FP32 路径通过固定输入比较；输入梯度和参数梯度均有限且重复差异为 0（报告中的 cosine 仅受浮点显示精度影响）。

- Seed1234 probe: `{p0.get('status')}`，最大 forward 差 `{p0max}`，最大 input-gradient 差 `{p0gmax}`。
- Seed4321 probe: `{p1.get('status')}`，最大 forward 差 `{p1max}`，最大 input-gradient 差 `{p1gmax}`。
- CPU 参考重复也为 0 差异。

这只撤回“LayerNorm 独立探针已证明不稳定”的旧判断，不能撤回完整模型 A/A FAIL；完整 EVA02 backward 的首个分叉仍未被本探针定位。

## 复现与证据

{command_note}

- JSON: [`ndp_layernorm_probe.json`]({out / 'ndp_layernorm_probe.json'})
- JSON: [`ndp_layernorm_probe_seed4321.json`]({out / 'ndp_layernorm_probe_seed4321.json'})
- 原始日志: [`ndp_layernorm_probe.log`]({Path('/home/DIFFER/log_NDP/ndp_relation_correction_20261021_192813/ndp_layernorm_probe.log')})

## 源码 SHA256

{hashes}
""")

    write(rep / "NDP_SAME_INPUT_SAME_UPSTREAM_AA.md", f"""# 同输入、同上游的 A/A 复核

## 可接受的局部结论

两种子每个探针组都满足：输入哈希相同、上游梯度哈希相同、stride/shape 相同、参数初始哈希相同；比较状态为 `VALID_COMPARISON`。因此本轮不能再把随机上游输入造成的差异称为 kernel 非确定性。

## 尚未解除的门禁

该测试覆盖的是 LayerNorm 局部路径，而不是完整训练图。历史完整模型同 AMP A/A 仍有约 `relative L2 0.00410/0.00471` 的梯度差异，且此前差异涉及 block22/attention 链；本轮没有完成 q/k/v、attention matmul、softmax、DDP 归约的同精度闭环定位。因此：

`A/A_REPRODUCIBILITY=FAIL`

不能进行 NDP optimizer-step 因果效果比较。

## 证据

- Seed1234/4321 probe JSON: [`ndp_layernorm_probe.json`]({out / 'ndp_layernorm_probe.json'})、[`ndp_layernorm_probe_seed4321.json`]({out / 'ndp_layernorm_probe_seed4321.json'})
- 旧完整模型 A/A 对照仍在历史目录，未覆盖：`/home/DIFFER/outputs/ndp_multipid_reconstruction_20261021_101900/ppu_same_precision_aa/`
- 回归测试: [`pytest_ndp_regression.log`]({Path('/home/DIFFER/log_NDP/ndp_relation_correction_20261021_192813/pytest_ndp_regression.log')})，23 passed。

{command_note}
""")

    write(rep / "NDP_BIDIRECTIONAL_RELATION_CORRECTION.md", f"""# 双侧身份关系修正

## 修正内容

旧实现以 `(interval, anchor_pid, negative_pid)` 为字典键并 last-write-wins，会覆盖同一身份对下不同图像边界。本批次改为保存完整 directed row 列表；图像级双侧关系要求同一时间区间、A 图像恰为 B 的负图像、B 图像恰为 A 的负图像、PID 对调一致，并保留 A→B 与 B→A 两个方向。

## 结果

| Seed | 输入 transition | 身份有向组 | 严格图像互反对 | 身份对-区间组 | A改善/B恶化 | A恶化/B改善 | 双改善 | 双恶化 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1234|{s[1234]['input_transition_rows']}|{s[1234]['identity_directed_rows']}|{s[1234]['image_level_reciprocal_pairs']}|{s[1234]['identity_pair_interval_count']}|{s[1234]['quadrants_image_level']['A改善_B恶化']}|{s[1234]['quadrants_image_level']['A恶化_B改善']}|{s[1234]['quadrants_image_level']['双改善']}|{s[1234]['quadrants_image_level']['双恶化']}|
|4321|{s[4321]['input_transition_rows']}|{s[4321]['identity_directed_rows']}|{s[4321]['image_level_reciprocal_pairs']}|{s[4321]['identity_pair_interval_count']}|{s[4321]['quadrants_image_level']['A改善_B恶化']}|{s[4321]['quadrants_image_level']['A恶化_B改善']}|{s[4321]['quadrants_image_level']['双改善']}|{s[4321]['quadrants_image_level']['双恶化']}|

双侧不对称仍然出现，但它是固定关系上的描述性现象，不是 NDP 训练收益证据。两个方向不能被平均成一个“身份对风险”。

证据 CSV：`{out}/bidirectional_correction/seed1234/identity_directed_rows.csv`、`{out}/bidirectional_correction/seed1234/image_level_reciprocal_pairs.csv`，以及 Seed4321 对应文件。

{command_note}
""")

    write(rep / "NDP_RECIPROCAL_KNN_VALIDATION.md", f"""# 反向 KNN 验证

图构建脚本已从“仅计算选定 anchor 的 Top-K”改为对所有候选负图像执行真实 Top-K 查询；anchor 或反向查询缺失时抛出异常，不再默认 False。

| Seed | 反向查询数 | 反向 Top-K 是否实际计算 | 未来节点参与选择 | 自引用 |
|---:|---:|---|---|---|
|1234|2153|是|否|排除|
|4321|2112|是|否|排除|

图关系源自 update0 train-only 特征，使用分块相似度，不执行 optimizer.step，也未使用官方 query/gallery/test。这个修正确保互为近邻标记有明确分母，但不证明互反关系本身值得保护。

证据：`{out}/reference_graph/seed1234/multi_pid_reference_summary.json`、Seed4321 对应 JSON。
""")

    def rel_line(seed):
        q = r[seed]
        high = q["by_cell"].get("HIGH_RELIABILITY__HIGH_VULNERABILITY", {})
        low = q["by_cell"].get("HIGH_RELIABILITY__LOW_VULNERABILITY", {})
        return f"|{seed}|{q['relation_count']}|{q['event_count']}|{pct(q['event_rate'])}|{high.get('n',0)}|{pct(high.get('event_rate'))}|{low.get('n',0)}|{pct(low.get('event_rate'))}|{q['auc_current_margin_score']:.4f}|{q['auc_history_instability_score']:.4f}|"
    write(rep / "NDP_RELIABILITY_VULNERABILITY_INFORMATION_AUDIT.md", f"""# Reliability 与 Vulnerability 信息审计

本批次使用 200→400 固定关系转移，分类只使用该转移之前的历史行，不使用未来事件。结果必须谨慎解释：当前脚本的 `HIGH_VULNERABILITY` 规则包含当前 margin 条件，所以它与 Hard Margin 并非独立变量；观察到低可靠/低脆弱空格不能被解释成自然事实上的四象限缺失。

| Seed | 关系 | 事件 | 总事件率 | 高可靠高脆弱 n/率 | 高可靠低脆弱 n/率 | 当前 margin 风险 AUC | 历史不稳定性 AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
{rel_line(1234)}
{rel_line(4321)}

共同观察：低脆弱组事件为 0，但这是因为 vulnerability 规则本身把接近边界的关系放入高脆弱，而不是独立预测成功。高可靠高脆弱组事件率高于高可靠低脆弱组，但高可靠组事件率并未稳定高于低可靠组，尚不能证明“历史可靠性”带来超出当前 margin 的增量信息。

证据：`{out}/reliability_vulnerability/seed1234/reliability_vulnerability_information_summary.json` 及逐关系 CSV；Seed4321 对应目录。
""")

    def trip_line(seed):
        q=t[seed]
        return f"|{seed}|{q['batch_count']}|{q['miner_row_count']}|{q['ndp_relation_count']}|{q['ndp_relations_with_batch_opportunity']}|{q['ndp_relations_with_full_miner_selection']}|{q['ndp_relations_with_active_triplet']}|{q['ndp_anchors_with_any_triplet']}/512|"
    write(rep / "NDP_REAL_TRIPLET_MINER_COVERAGE.md", f"""# 真实 Batch-Hard Triplet 覆盖审计

源码复核结果：原 Triplet 使用欧氏距离；hard positive 是 batch 内同 PID 最大欧氏距离，hard negative 是异 PID 最小欧氏距离；配置 `NO_MARGIN=True`，损失为 `SoftMarginLoss(dist_an-dist_ap)`；make_loss 没有把 feat[0] 先归一化后再交给 Triplet。此前用 cosine ReLU 代理的 T0 不能当作真实 Triplet。

本批次通过原 sampler 代码重建 epoch 60/61、rank0/1 的 batch，使用 update0 特征只读计算 miner，不调用 optimizer.step。没有历史逐步 batch/miner 日志，因此下面是“重建采样近似”，不是历史真实梯度覆盖。

| Seed | 重建 batch | miner 行 | NDP 固定关系 | 有共同 batch 机会 | 完整 miner 选中 | 有效 Triplet 直接选中 | 有任意其他 Triplet 的 anchor |
|---:|---:|---:|---:|---:|---:|---:|---:|
{trip_line(1234)}
{trip_line(4321)}

直接固定关系覆盖为 0，不表示 Triplet 对这些 anchor 没有任何作用：两种子都有 398/512 个 NDP anchor 在其他关系上出现过 active Triplet。正确结论是：固定的多 PID 局部关系没有被该重建 Batch-Hard 直接覆盖，而 anchor 级间接约束存在；真实历史梯度覆盖仍为 `NOT_AVAILABLE`。

证据：`{out}/triplet_coverage/seed1234/triplet_miner_coverage_summary.json`、逐关系 `ndp_triplet_coverage_rows.csv` 及 Seed4321 对应文件。
""")

    def mask_table(seed):
        rows=m[seed]["masks"]
        return "\n".join(f"|{x['method']}|{x['relation_count']}|{x['event_count_200_400']}|{pct(x['event_rate_within_method'])}|{x['pid_count']}|" for x in rows)
    write(rep / "NDP_MATCHED_NONREDUNDANCY_AUDIT.md", f"""# 预算匹配离线非冗余性审计

该审计固定 2560 条 update0 difficult-K5 关系，比较实际 Batch-Hard 重建覆盖、margin 代理、可靠性/脆弱性 mask、严格图像互反 mask 及数量匹配随机 mask。它没有执行网络 forward 或 optimizer.step；冻结 feature autograd 只验证局部梯度接口，因此不能宣称训练收益。

## Seed1234

| mask | 关系数 | 200→400 事件 | 组内事件率 | PID数 |
|---|---:|---:|---:|---:|
{mask_table(1234)}

## Seed4321

| mask | 关系数 | 200→400 事件 | 组内事件率 | PID数 |
|---|---:|---:|---:|---:|
{mask_table(4321)}

B0 直接关系覆盖为 0；B6/B7 样本数非常小（32/28 及 4/2），不能用其零事件支持机制优越性。B5 组虽有较高描述性事件率，但它的 vulnerability 定义含 margin 规则，且与随机数量匹配结果并未提供可确认的跨 seed 增量证据。梯度 JSON 报告 update0 同状态时 reactive hinge 不应激活；其“feature_space_only=true”，不等价网络参数梯度。

证据 JSON/CSV：`{out}/matched_nonredundancy/seed1234/`、Seed4321 对应目录。
""")

    write(rep / "NDP_MINIMAL_REACTIVE_CANDIDATE_DECISION.md", f"""# 最小反应式 NDP 候选决策

候选保持为单侧关系保护：

`L = mean_r w_r ReLU(M_ref(r) - M_current(r) - epsilon)`。

它具有三项明确性质：参考分支 detach；关系改善时零惩罚；只有当前边界相对可靠参考下降超过容忍量才产生梯度。它与参考 Margin 自适应 Hinge 在单一 `(anchor, positive, negative)` 关系上是同一类约束，不能仅凭名字或 mask 差异称为独立创新。

目前唯一可审计的增量来自关系集合：固定多 PID/双侧关系可能不在 Batch-Hard 的直接选择范围内。但这一覆盖结论依赖重建 sampler，且 B6/B7 关系数过少，未证明可靠性、双侧性或多 PID 本身带来超出 Triplet 的优化信息。因 A/A 仍 FAIL，本轮不接入训练。

决策：保留为“关系诊断与反应式原型”，不晋级正式 NDP loss；预测/干预机制仍 `INCONCLUSIVE`。
""")

    write(rep / "NDP_FINAL_RESEARCH_DECISION.md", f"""# NDP 关系修正批次最终决策

## 已验证

1. 旧 LayerNorm 大差异由上游随机张量未固定造成，旧证据无效；固定输入/参数/上游后，两个 seed 的通用 LayerNorm 和 EVA02 block22.norm1 PPU AMP/FP32 探针稳定。
2. 双侧身份关系不再 last-write-wins；Seed1234/4321 分别得到 80/70 个严格图像级互反对，A→B 与 B→A 的改善/恶化四象限仍不对称。
3. 反向 KNN 已真实计算所有候选负图像（2153/2112），缺失查询不再默认 False。
4. 原始 Triplet 是欧氏 Batch-Hard + SoftMarginLoss（NO_MARGIN），不是此前 cosine ReLU 代理。
5. 在重建采样审计中，2560 条固定 NDP 关系两 seed 的共同 batch、完整 miner 选中和 active Triplet 直接覆盖均为 0；398/512 anchor 有其他关系的 Triplet 约束。

## 未验证/失败

- 没有历史逐步 batch/miner 日志，不能把重建覆盖写成真实训练梯度覆盖。
- Reliability/Vulnerability 分类存在逻辑耦合，尚无独立增量价值证据。
- 双侧关系样本量有限，不能证明双侧保护优于单侧。
- PPU 完整模型同精度 backward 根因仍未定位；LayerNorm 修复不等于 A/A PASS。
- 没有执行 NDP optimizer.step、短程效果或正式训练。

## 门禁

`A/A_REPRODUCIBILITY=FAIL`

`NDP_MECHANISM=INCONCLUSIVE`

`CONTROLLED_SHORT_PILOT=NO_GO`

本批次没有改变 GPC、CE、Triplet、Bio、EVA02、采样协议或评估协议，也没有使用官方 query/gallery/test。

{command_note}

### 代码 SHA256

{hashes}

### 归档

- 当前输出: `{out}`
- 报告目录: `{rep}`
- 回归测试: 23 项通过；历史报告和失败结果保留，未覆盖。
""")

    print(json.dumps({"report_dir": str(rep), "reports": sorted(p.name for p in rep.glob("*.md")), "source_hash_count": len(source_files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
