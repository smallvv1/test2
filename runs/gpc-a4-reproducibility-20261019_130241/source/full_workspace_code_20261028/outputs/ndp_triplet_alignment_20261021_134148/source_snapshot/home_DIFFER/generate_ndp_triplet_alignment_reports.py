#!/usr/bin/env python3
"""生成 NDP/真实 Triplet 对照批次的中文审计报告与源码清单。

只读取既有 JSON/CSV/源码和本轮只读探针结果，不执行模型更新。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/home/DIFFER")
RUN = ROOT / "outputs/ndp_triplet_alignment_20261021_134148"
REPORT = ROOT / "reports/ndp_triplet_alignment_20261021_134148"
SNAPSHOT = RUN / "source_snapshot"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def j(path: Path):
    return read_json(path)


def pct(x):
    return "N/A" if x is None else f"{100.0 * float(x):.4f}%"


def num(x, digits=6):
    return "N/A" if x is None else f"{float(x):.{digits}g}"


def write(name: str, text: str):
    for root in (REPORT, RUN / "reports"):
        root.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def command_block(commands):
    return "\n".join(f"`{x}`" for x in commands)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    formula = j(RUN / "triplet_formula_audit.json")
    raw = {s: j(RUN / f"seed{s}/triplet_raw_feature_alignment_seed{s}.json") for s in (1234, 4321)}
    miner = {s: j(RUN / f"miner_coverage/seed{s}/triplet_miner_coverage_summary.json") for s in (1234, 4321)}
    rel = {s: j(RUN / f"reliability_incremental/seed{s}/reliability_incremental_summary.json") for s in (1234, 4321)}
    bidi = {s: j(RUN / f"bidirectional_additional/seed{s}/bidirectional_additional_summary.json") for s in (1234, 4321)}
    reactive = {s: j(RUN / f"reactive_feature_gradient/seed{s}/reactive_feature_gradient_summary.json") for s in (1234, 4321)}
    aa = j(RUN / "ppu_same_precision_kernel_probe.json")

    try:
        server_head = subprocess.check_output(["git", "-C", "/data/DIFFER", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        server_head = "UNAVAILABLE"
    remote_head = "0423469770dc90d85d6354667d4931d768c3235b"
    baseline = {
        "/data/DIFFER/scripts/gpc_v2_train.py": "d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5",
        "/data/DIFFER/loss/make_loss.py": "9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043",
        "/data/DIFFER/configs/ltcc/eva02_l_bio.yml": "c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97",
    }
    sources = [
        ROOT / "scripts/ndp_real_triplet_miner_coverage.py",
        ROOT / "scripts/ndp_triplet_formula_audit.py",
        ROOT / "scripts/ndp_triplet_raw_feature_alignment.py",
        ROOT / "scripts/ndp_reliability_incremental_audit.py",
        ROOT / "scripts/ndp_bidirectional_additional_audit.py",
        ROOT / "scripts/ndp_reactive_feature_gradient_audit.py",
        ROOT / "scripts/ndp_ppu_same_precision_kernel_probe.py",
        ROOT / "tests/test_ndp_triplet_formula.py",
        ROOT / "tests/test_ndp_relation_correction.py",
        ROOT / "tests/test_ndp_triplet_alignment_audit.py",
        ROOT / "scripts/generate_ndp_triplet_alignment_reports.py",
        Path("/data/DIFFER/loss/triplet_loss.py"),
    ]
    source_hashes = {}
    for src in sources:
        if not src.is_file():
            continue
        source_hashes[str(src)] = sha(src)
        dst = SNAPSHOT / ("data_DIFFER" if str(src).startswith("/data/DIFFER") else "home_DIFFER") / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (SNAPSHOT / "SHA256SUMS.json").write_text(json.dumps(source_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    commands = [
        "/usr/local/bin/python3.10 -u scripts/ndp_triplet_formula_audit.py --output outputs/ndp_triplet_alignment_20261021_134148/triplet_formula_audit.json",
        "/usr/local/bin/python3.10 -u scripts/ndp_triplet_raw_feature_alignment.py --seed 1234 --output-dir outputs/ndp_triplet_alignment_20261021_134148/seed1234 --epoch 60 --rank 0",
        "/usr/local/bin/python3.10 -u scripts/ndp_triplet_raw_feature_alignment.py --seed 4321 --output-dir outputs/ndp_triplet_alignment_20261021_134148/seed4321 --epoch 60 --rank 0",
        "/usr/local/bin/python3.10 -u scripts/ndp_real_triplet_miner_coverage.py --seed 1234 --features outputs/ndp_true_multipid_20261021_103952/full_features/seed1234/update0_full_features_seed1234.npz --graph outputs/ndp_true_multipid_20261021_103952/reference_graph/seed1234/multi_pid_reference_graph.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/miner_coverage/seed1234 --epochs 60 61",
        "/usr/local/bin/python3.10 -u scripts/ndp_real_triplet_miner_coverage.py --seed 4321 --features outputs/ndp_true_multipid_20261021_103952/full_features/seed4321/update0_full_features_seed4321.npz --graph outputs/ndp_true_multipid_20261021_103952/reference_graph/seed4321/multi_pid_reference_graph.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/miner_coverage/seed4321 --epochs 60 61",
        "/usr/local/bin/python3.10 -u scripts/ndp_reliability_incremental_audit.py --seed 1234 --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed1234/multi_pid_transition_rows.csv --reliability-rows outputs/ndp_true_multipid_20261021_103952/analysis/seed1234/reliability/reliability_vulnerability_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/reliability_incremental/seed1234",
        "/usr/local/bin/python3.10 -u scripts/ndp_reliability_incremental_audit.py --seed 4321 --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed4321/multi_pid_transition_rows.csv --reliability-rows outputs/ndp_true_multipid_20261021_103952/analysis/seed4321/reliability/reliability_vulnerability_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/reliability_incremental/seed4321",
        "/usr/local/bin/python3.10 -u scripts/ndp_bidirectional_additional_audit.py --seed 1234 --pairs outputs/ndp_true_multipid_20261021_103952/analysis/seed1234/bidirectional/bidirectional_pairs.csv --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed1234/multi_pid_transition_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/bidirectional_additional/seed1234",
        "/usr/local/bin/python3.10 -u scripts/ndp_bidirectional_additional_audit.py --seed 4321 --pairs outputs/ndp_true_multipid_20261021_103952/analysis/seed4321/bidirectional/bidirectional_pairs.csv --transition-rows outputs/ndp_true_multipid_20261021_103952/trajectory_eval/seed4321/multi_pid_transition_rows.csv --output-dir outputs/ndp_triplet_alignment_20261021_134148/bidirectional_additional/seed4321",
        "/usr/local/bin/python3.10 -u scripts/ndp_reactive_feature_gradient_audit.py --seed 1234 --features outputs/ndp_multipid_reconstruction_20261021_101900/feature_export/seed1234/trajectory_features_seed1234.npz --groups outputs/ndp_future_boundary_20261021_094250/coverage_expansion/seed1234_retry1/fixed_diagnostic_groups.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/reactive_feature_gradient/seed1234",
        "/usr/local/bin/python3.10 -u scripts/ndp_reactive_feature_gradient_audit.py --seed 4321 --features outputs/ndp_multipid_reconstruction_20261021_101900/feature_export/seed4321/trajectory_features_seed4321.npz --groups outputs/ndp_future_boundary_20261021_094250/coverage_expansion/seed4321/fixed_diagnostic_groups.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/reactive_feature_gradient/seed4321",
        "/usr/local/bin/python3.10 -u scripts/ndp_ppu_same_precision_kernel_probe.py --output outputs/ndp_triplet_alignment_20261021_134148/ppu_same_precision_kernel_probe.json --repeats 3 --include-eva-norm",
        "PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER /usr/local/bin/python3.10 -m pytest -q tests/test_ndp_*.py",
    ]

    formula_report = f"""# NDP 真实 Triplet 公式纠错

## 结论

原重建脚本把 `SoftMarginLoss(dist_an-dist_ap, target=+1)` 错写成 `softplus(dist_an-dist_ap)`，会把正确分离关系的损失方向反过来。已修正为：

\[
 d = d_{{an}}-d_{{ap}},\qquad L=\log(1+\exp(-d)),\qquad \frac{{\partial L}}{{\partial d}}=-\sigma(-d).
\]

逐元素对照 `torch.nn.SoftMarginLoss(reduction='none')` 的结果：

- loss 最大绝对误差：`{formula['loss_max_abs_error']:.3g}`；
- 梯度最大绝对误差：`{formula['gradient_max_abs_error']:.3g}`；
- `d_an` 增大时 loss 降低：`{formula['loss_direction_dan_increase']}`；
- 方向正确：`{formula['correct_derivative_negative']}`。

`gradient_above_1e-8` 只是连续 SoftMargin 梯度的报告阈值，不能解释为 Hinge 的 active/inactive 机制。B0 的 `hard_factor=0`，欧氏距离下限为 `1e-12`；默认 Triplet 输入不在 `TripletLoss` 内归一化。

## 原始实现核对

`loss/triplet_loss.py` 使用欧氏距离；同 PID 中取最大正距离，不同 PID 中取最小负距离；`margin=None` 时调用 SoftMarginLoss；`make_loss.py` 以 `TripletLoss(feat[0], target, normalize_feature=False)` 调用。自身对角线不构成最大正样本，只要 batch 中存在另一张同 PID 图像。

## 证据与边界

修正前的 miner loss/梯度解释标记为 `SUPERSEDED`，历史文件保留。本报告没有执行 optimizer.step，也没有修改 GPC、CE、Triplet 或 Bio。

## 可审计命令

{command_block(commands[:1])}
"""
    write("NDP_REAL_TRIPLET_FORMULA_CORRECTION.md", formula_report)

    raw_lines = []
    for s in (1234, 4321):
        x = raw[s]
        raw_lines.append(f"| {s} | {num(x['raw_norm_min'],6)}–{num(x['raw_norm_max'],6)} | {num(x['triplet_raw']['loss'],8)} | {num(x['triplet_normalized']['loss'],8)} | {x['raw_vs_norm_hard_positive_same']} | {x['raw_vs_norm_hard_negative_same']} |")
    raw_report = f"""# NDP Triplet 原始特征对齐审计

## 结论

已在 PPU `cuda:0` 对两个 B0 checkpoint 做只读 `eval/no_grad` 前向，并用相同 batch 的 `feat[0]` 叶张量调用原始 `TripletLoss`；没有请求模型参数梯度，也没有 optimizer.step。原训练契约是未归一化的 1024 维 `feat[0]`，而 update0 全量检索库是独立的 L2 归一化特征，不能混用。

| seed | raw 特征范数范围 | raw loss | normalized-bank loss | hard positive 相同 | hard negative 相同 |
|---:|---:|---:|---:|:---:|:---:|
{chr(10).join(raw_lines)}

seed4321 的 hard negative 在 raw 与 normalized 路径不同；因此旧的归一化 update0 特征只能用于离线邻域覆盖近似，不能声称复现真实训练 Batch-Hard。seed1234 本批恰好索引相同，但 loss 与梯度规模仍显著不同。

checkpoint：seed1234 `{raw[1234]['checkpoint_sha256']}`；seed4321 `{raw[4321]['checkpoint_sha256']}`。模型输出为 `torch.float32`，shape `[8,1024]`，均有限。

## 数据边界

这是从 checkpoint 重建的 sampler batch，不是保存下来的历史 batch 特征；只用于输入契约和 miner 对齐。正式历史 Batch 逐步记录仍不可用。数据仅为 LTCC train，未读取官方 query/gallery/test。

## 可审计命令

{command_block(commands[1:3])}
"""
    write("NDP_TRIPLET_RAW_FEATURE_ALIGNMENT.md", raw_report)

    cov_lines = []
    for s in (1234, 4321):
        x = miner[s]
        ctrl = x["random_pid_matched_control"]
        cov_lines.append(f"| {s} | {x['batch_count']} | {x['miner_row_count']} | {x['ndp_relation_count']} | {pct(x['coverage_negative_pid_opportunity'])} | {pct(x['coverage_hard_negative_pid'])} | {pct(x['coverage_fixed_image_opportunity'])} | {x['ndp_relations_with_full_miner_selection']} | {x['ndp_anchors_with_any_triplet']}/512 | {pct(ctrl['coverage_hard_negative_pid'])} |")
    cov_report = f"""# NDP 多层级真实 Triplet 覆盖审计

## 结论

当前没有历史逐步 miner 日志，因此本报告是按固定 DIFFER sampler 重建 epoch 60/61、两个 rank 的只读近似；不能标记为 observed historical coverage，也不能推出真实梯度覆盖率。

| seed | batches | miner rows | 固定关系 | B 身份共同出现 | D hard-negative PID | C 固定图像共同出现 | F 完整关系选中 | 有任意 Triplet 的 anchor | 随机 PID 匹配 hard-PID |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(cov_lines)}

层级定义：A 是 anchor 曝光；B 是任意该负 PID 同 batch；C 是固定负图像同 batch；D 是 hard negative PID 等于目标 PID；E 是 hard negative 图像等于固定图像；F 要求 hard positive 与 hard negative 同时落入固定候选。两 seed 的 B/D 约为 8.40%/7.70% 与 8.59%/7.81%，但 C/E/F 为 5/2560、0/2560 和 0。398/512 anchor 通过其他关系获得 Triplet 约束，说明图像精确覆盖为零不等于 anchor 完全没有 Triplet 训练。

随机 PID 数量匹配控制的 D 覆盖为 seed1234 `{pct(miner[1234]['random_pid_matched_control']['coverage_hard_negative_pid'])}`、seed4321 `{pct(miner[4321]['random_pid_matched_control']['coverage_hard_negative_pid'])}`；固定困难关系分别高出约 `{100*(miner[1234]['coverage_hard_negative_pid']-miner[1234]['random_pid_matched_control']['coverage_hard_negative_pid']):.2f}` 和 `{100*(miner[4321]['coverage_hard_negative_pid']-miner[4321]['random_pid_matched_control']['coverage_hard_negative_pid']):.2f}` 个百分点，但这是重建采样的描述性差异，不是历史因果证据。

## 重要限制

脚本当前使用 update0 归一化检索特征做近似 miner；P1 已证明这与真实 raw Triplet 输入不总相同。因此本报告可支持“关系访问机会口径”，不能支持“真实训练中 Triplet 实际覆盖比例”。

## 可审计命令

{command_block(commands[3:5])}
"""
    write("NDP_MULTILEVEL_TRIPLET_COVERAGE.md", cov_report)

    def mstr(z):
        return f"{z['mean']:.4f} [{z['ci95_low']:.4f},{z['ci95_high']:.4f}]"

    rel_lines = []
    for s in (1234, 4321):
        x = rel[s]
        m = x["metrics"]
        pre = m["pre200_history_min_margin_risk"]
        rel_lines.append(f"| {s} | {x['eligible_count']} | {x['strict_flip_count']} ({pct(x['strict_flip_rate'])}) | {x['robust_counts']['delta_1e-3']} | {x['robust_counts']['delta_1e-2']} | {m['current_margin']['auc']:.4f}/{m['current_margin']['average_precision']:.4f} | {m['pre200_history_instability']['auc']:.4f}/{m['pre200_history_instability']['average_precision']:.4f} | {m['pre200_history_min_margin_risk']['auc']:.4f}/{m['pre200_history_min_margin_risk']['average_precision']:.4f} | {m['pre200_history_decline_slope']['auc']:.4f}/{m['pre200_history_decline_slope']['average_precision']:.4f} |")
    rel_report = f"""# NDP Reliability/Vulnerability 增量信息审计

## 正确分母

只使用 `M_200 > 0` 的关系作为候选，标签为 `M_400 <= 0`。已在 update200 错误的关系不进入首次翻转分母。update400 只产生标签，不参与分数、关系选择或阈值。

| seed | eligible | 严格翻转 | 稳健 1e-3 | 稳健 1e-2 | 当前 Margin AUC/AP | pre-200 波动 AUC/AP | pre-200 最小 Margin AUC/AP | pre-200 斜率 AUC/AP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rel_lines)}

当前边界距离是最强的描述变量，但这有明显的“靠近零定义效应”：两 seed 的事件分别为 31/1020 与 9/977，且事件主要集中于 `M_200<=0.05`（31/356 与 9/320）。pre-200 历史最小 Margin 仍然很高，但它本质上也是历史边界接近度的代理，不能直接称为对未来更新方向的独立预测；历史下降斜率 AUC 仅 0.3244/0.4014，方向不稳定且 AP 很低。pre-200 波动 AUC 0.6375/0.7233，但 AP 0.0539/0.0197，事件稀少，不能作为跨 seed 已确认的动态机制。

做了 200 次按当前 Margin 最近邻的事件/非事件描述性匹配。匹配后的历史不稳定性 AUC 均值为 seed1234 `{mstr(rel[1234]['matched_current_margin_control']['history_auc']['history_instability'])}`、seed4321 `{mstr(rel[4321]['matched_current_margin_control']['history_auc']['history_instability'])}`；历史斜率均值为 `{mstr(rel[1234]['matched_current_margin_control']['history_auc']['history_decline_slope'])}` / `{mstr(rel[4321]['matched_current_margin_control']['history_auc']['history_decline_slope'])}`。小事件数和匹配近似使其仅为探索性控制，不能升级为预测 PASS。

**判断：**当前结果支持“当前局部边界接近零的关系更容易翻转”，不支持“历史下降趋势能稳定预测新的退化”。预测式 NDP 仍 `NOT_SUPPORTED`；若继续研究，应优先转为反应式、事后单侧保护，并且仍需证明其超出真实 Triplet 的覆盖增量。

## 可审计命令

{command_block(commands[5:7])}
"""
    write("NDP_RELIABILITY_INCREMENTAL_INFORMATION.md", rel_report)

    bidi_lines = [f"| {s} | {bidi[s]['image_pair_count']} | {bidi[s]['identity_pair_count']} | {bidi[s]['average_delta_negative_count']} | {bidi[s]['most_dangerous_direction_negative_count']} | {bidi[s]['either_flip']} |" for s in (1234, 4321)]
    bidi_report = f"""# NDP 双侧身份判别增量信息

双侧记录按同一 interval 与 relation index 对齐，保留 A→B、B→A 两个有向方向；没有用下一 transition 的 target_margin 代替当前 transition。结果：

| seed | image-pair records | identity pairs | 平均方向 delta<0 | 最危险方向 delta<0 | 任一严格翻转 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(bidi_lines)}

五个时间区间均被保留。两 seed 在该严格互反图像子集上没有严格翻转，但分别有 53/80 与 47/70 条记录至少一侧 delta 为负；这说明一侧改善/另一侧恶化的“增量风险”可能存在，但当前集合没有足够严格翻转来证明双侧保护优于单侧保护。identity-pair 仅 15/13 个，不能外推全训练集。

双侧关系因此目前是诊断信息，不是已验证的独立训练机制；需要更大的、预注册且不按未来风险筛选的双侧关系集合，以及可重复的真实训练梯度比较。

## 可审计命令

{command_block(commands[7:9])}
"""
    write("NDP_BIDIRECTIONAL_ADDITIONAL_INFORMATION.md", bidi_report)

    def mask_row(s, name):
        return reactive[s]["masks"][name]
    grad_lines = []
    for s in (1234, 4321):
        for name in ("G2_reference_margin", "G3_random_count_matched_G5", "G5_reliable_vulnerable", "G6_reciprocal_image_relation", "G7_reliable_vulnerable_reciprocal"):
            z = mask_row(s, name)
            grad_lines.append(f"| {s} | {name} | {z['selected_relations']} | {z['active_relations']} | {z['loss']:.6g} | {z['gradient_norm']:.6g} | {z['flip_200_400_count']} |")
    reactive_report = f"""# NDP 反应式特征梯度审计

## 口径

参考为 update0，当前为 update200；可靠性/脆弱性关系选择只使用 update200 以前信息，update400 只作为事后翻转标签。所有梯度是冻结身份特征上的一阶 autograd，`network_parameter_gradient=false`，没有 optimizer.step。

| seed | mask | selected | active（mask内） | loss | feature grad norm | 200→400 flips |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(grad_lines)}

修复后 `active_relations` 只统计被当前 mask 选中的正损失关系，`all_candidate_active_relations` 另行保留；因此不会把未选中的 raw active 关系误报成当前方法激活。`G7` 在 seed1234 为空，在 seed4321 只有 5 条且当前不激活；这说明严格双侧+可靠脆弱交集覆盖不足。

结果只能证明候选损失在 update200 特征空间中可计算、梯度有限、改善关系不惩罚；不能证明网络参数更新后的真实收益。特别是 G5 的 seed4321 选中 23 条但 loss/梯度为 0，说明“关系选择上看似危险”不等于当前参考单侧保护已经激活。

## 机制判断

固定关系上的 `ReLU(M_ref-M_cur-epsilon)` 仍与参考 Margin Hinge/自适应 Triplet 有直接数学重叠。当前新增信息主要来自关系选择（历史可靠性、双侧、跨 PID），而不是损失形式本身；在真实 miner 覆盖、A/A 和参数更新均未通过前，不能称为独立创新。

## 可审计命令

{command_block(commands[9:11])}
"""
    write("NDP_REACTIVE_FEATURE_GRADIENT_AUDIT.md", reactive_report)

    aa_report = f"""# NDP EVA02 完整/局部 A/A 数值复核

## 本轮固定输入探针

修复后的 PPU 探针在循环外一次性物化输入、上游梯度、参数，并逐次核对 SHA256。独立 LayerNorm 与真实 `EVA02 block22.norm1` 的 AMP/FP32、输入梯度和 affine 参数梯度重复均为 `max_abs=0`、`relative_l2=0`，参数未改变；旧的 LayerNorm 巨大差异标记为 `INVALID_UPSTREAM_CONTROL`，不能继续作为 kernel 不稳定证据。

本轮探针状态：`{aa['status']}`，PPU `{aa['device']}`，torch `{aa['torch']}`，重复 `{aa['repeats']}`。

## 门禁结论

这并没有解决完整 EVA02 训练图 A/A。历史完整模型同精度 AMP A/A 仍有 seed1234 相对 L2 约 `0.00410`、seed4321 约 `0.00471` 的参数梯度差异；该差异来自完整模块组合路径，尚未在固定上游梯度下缩小到 q/k/v、attention score/softmax、残差或 DDP 归约的唯一 kernel。因此：

`A/A_REPRODUCIBILITY = FAIL`

本轮没有执行 optimizer.step，也没有把局部 norm 稳定误报成完整模型通过。下一步必须继续做真实 block22 物化输入/上游梯度的组合路径隔离；在此之前不允许 NDP 效果比较。

## 可审计命令

{command_block(commands[11:12])}
"""
    write("NDP_EVA02_BLOCK22_AA_ISOLATION.md", aa_report)

    final_report = f"""# NDP 真实 Triplet 机制对照：最终研究决策

## 总结决策

- `A/A_REPRODUCIBILITY = FAIL`
- `NDP_MECHANISM = INCONCLUSIVE`
- `CONTROLLED_SHORT_PILOT = NO_GO`
- `NDP_FORMULA_CORRECTION = PASS`
- `RAW_TRIPLET_FEATURE_ALIGNMENT = PASS_READONLY`
- `MULTILEVEL_COVERAGE = PASS_RECONSTRUCTED_NOT_HISTORICAL`
- `REACTIVE_FEATURE_GRADIENT = PASS_FIRST_ORDER_ONLY`

## 已验证事实

1. 原始 Triplet 是未归一化 `feat[0]` 上的欧氏 Batch-Hard + SoftMarginLoss。旧重建公式方向错误，现已与 PyTorch 精确一致。
2. 归一化检索 bank 不能默认代表真实 Triplet 输入；seed4321 的 raw/normalized hard negative 已发生变化。
3. 重建 sampler 的 PID 级机会/选择约 8%，但固定图像级完整 NDP 关系为 0；因此“精确图像关系 0”不能等价为“Triplet 完全没有身份监督”，也不能等价为“真实 miner 覆盖已证实”。
4. 首次翻转应只在 update200 正确关系中评价。当前 Margin 接近零的关系最能解释事件；pre-200 趋势跨 seed 不稳定，预测式 NDP 没有获得可靠证据。
5. 严格互反图像子集存在不少一侧 delta<0，但没有严格翻转，样本量太小，双侧保护尚无独立价值证明。
6. update200 冻结特征上的反应式梯度有限，但这不是网络参数梯度，也不是效果证据；其核心 Hinge 仍与参考 Margin Triplet 重叠。
7. 修复后的 LayerNorm/直接 norm1 探针稳定，但完整 EVA02 A/A 仍 FAIL，旧完整模型梯度差异没有被解释。
8. 本批最终回归测试为 `33 passed`；失败的早期 miner/reliability 进程日志保留，修复后的重跑日志分别标记为 `*_rerun.log`/`*_pre200.log`。

## 对核心问题的回答

### 原始 Triplet 真正覆盖了多少 NDP 关系？

按当前重建口径，PID 级 hard-negative 覆盖为 seed1234 `197/2560=7.6953%`、seed4321 `200/2560=7.8125%`；固定图像级机会为 `5/2560` 与 `0/2560`，完整关系选中均为 0。由于没有历史 miner 日志且输入特征曾有 raw/normalized 差异，这些数字只能是近似覆盖，不是实际梯度覆盖率。

### 历史可靠性和双侧多身份是否提供独立信息？

当前没有足够证据。历史最小 Margin 的高 AUC主要反映边界接近度；趋势不稳定，双侧样本只有 80/70 个图像级转移，双侧+可靠交集过小。反应式梯度与普通参考 Margin Hinge 的差异主要来自 mask，而非已证明的额外判别监督。

### 最小值得保留的机制

仅保留“固定历史关系、当前已发生退化时的单侧 Margin 诊断函数”作为研究接口；暂不命名为已成立的 NDP Loss。若继续，下一实验必须先记录真实 raw Triplet miner、解决完整 EVA02 同精度 A/A，并在相同关系预算下比较可靠关系 mask 与随机/当前 Hard Margin mask。

## 研究边界

本轮所有模型计算均为 PPU 只读前向或无 optimizer.step 探针；LTCC train-only；未使用 query/gallery/test；GPC-v1、CE、Triplet、Bio、EVA02 和历史 checkpoint 未修改。

## 版本与校验

- 远端 main 检查值：`{remote_head}`。
- `/data/DIFFER` 工作树 HEAD：`{server_head}`。
- 基线文件 SHA256：见 `RUN_MANIFEST.json` 和本批 `source_snapshot/SHA256SUMS.json`。
- 本批源码快照已保存；未复制 checkpoint、训练图片或完整特征库。

## 可审计命令

{command_block(commands)}
"""
    write("NDP_FINAL_NEXT_STAGE_DECISION.md", final_report)

    manifest = {
        "run_time": "20261021_134148",
        "status": "COMPLETED_OFFLINE_AUDIT_NO_OPTIMIZER_STEP",
        "remote_main_checked": remote_head,
        "server_head": server_head,
        "python": "/usr/local/bin/python3.10",
        "torch": "2.6.0",
        "device_policy": "PPU/CUDA for model probes; CPU only for offline statistics",
        "official_query_gallery_used": False,
        "optimizer_step_executed": False,
        "decisions": {"A/A_REPRODUCIBILITY": "FAIL", "NDP_MECHANISM": "INCONCLUSIVE", "CONTROLLED_SHORT_PILOT": "NO_GO"},
        "baseline_sha256": baseline,
        "source_sha256": source_hashes,
        "checkpoint_sha256": {str(s): raw[s]["checkpoint_sha256"] for s in (1234, 4321)},
        "commands": commands,
        "artifacts": {
            "formula": str(RUN / "triplet_formula_audit.json"),
            "raw_alignment": [str(RUN / f"seed{s}/triplet_raw_feature_alignment_seed{s}.json") for s in (1234,4321)],
            "coverage": [str(RUN / f"miner_coverage/seed{s}/triplet_miner_coverage_summary.json") for s in (1234,4321)],
            "reliability": [str(RUN / f"reliability_incremental/seed{s}/reliability_incremental_summary.json") for s in (1234,4321)],
            "bidirectional": [str(RUN / f"bidirectional_additional/seed{s}/bidirectional_additional_summary.json") for s in (1234,4321)],
            "reactive_gradient": [str(RUN / f"reactive_feature_gradient/seed{s}/reactive_feature_gradient_summary.json") for s in (1234,4321)],
            "aa_probe": str(RUN / "ppu_same_precision_kernel_probe.json"),
        },
    }
    (RUN / "RUN_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPORTS_GENERATED", "reports": [str(REPORT / p) for p in ["NDP_REAL_TRIPLET_FORMULA_CORRECTION.md", "NDP_TRIPLET_RAW_FEATURE_ALIGNMENT.md", "NDP_MULTILEVEL_TRIPLET_COVERAGE.md", "NDP_RELIABILITY_INCREMENTAL_INFORMATION.md", "NDP_BIDIRECTIONAL_ADDITIONAL_INFORMATION.md", "NDP_REACTIVE_FEATURE_GRADIENT_AUDIT.md", "NDP_EVA02_BLOCK22_AA_ISOLATION.md", "NDP_FINAL_NEXT_STAGE_DECISION.md"]], "source_snapshot": str(SNAPSHOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
