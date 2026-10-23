#!/usr/local/bin/python3.10
"""Assemble the three requested NDP training reports from immutable artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path("/home/DIFFER")
REPORT_ID = "20261023_033405"
REPORT_DIR = ROOT / "reports" / f"ndp_training_{REPORT_ID}"
EVAL_DIR = ROOT / "outputs" / f"ndp_evaluation_{REPORT_ID}"
FULL = {1234: "20261023_032622", 4321: "20261023_033008"}
SMOKE = {1234: "20261023_032320", 4321: "20261023_032455"}
UPDATES = (0, 20, 50, 100, 200, 400)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, digits=6):
    if x is None or x == "":
        return "—"
    return f"{float(x):.{digits}f}"


def pct(x, digits=2):
    return f"{100.0 * float(x):.{digits}f}%"


def md_table(headers, body):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in body)
    return "\n".join(out)


def training_summary(seed: int):
    b0 = rows(ROOT / "outputs/ndp_overnight_20261020_201841" / f"trajectory_seed{seed}" / "trajectory_steps.csv")
    n1 = rows(ROOT / "outputs" / f"ndp_training_{FULL[seed]}" / f"full_seed{seed}" / "training_steps.csv")
    def one(rs, arm):
        return {
            "arm": arm, "updates": len(rs),
            "mean_loss": sum(float(r["loss"] if "loss" in r else r["total"]) for r in rs) / len(rs),
            "final_loss": float(rs[-1]["loss"] if "loss" in rs[-1] else rs[-1]["total"]),
            "mean_ce": sum(float(r["ce"]) for r in rs) / len(rs),
            "mean_triplet": sum(float(r["triplet"]) for r in rs) / len(rs),
            "mean_bio": sum(float(r["bio"]) for r in rs) / len(rs),
            "mean_gpc": sum(float(r["gpc"]) for r in rs) / len(rs),
            "gpc_nonzero": sum(float(r["gpc"]) != 0.0 for r in rs),
            "found_inf": sum(int(r["found_inf"]) for r in rs),
            "finite": all(r["parameter_finite"] == "True" for r in rs),
        }
    n1s = one(n1, "N1")
    n1s.update({
        "ndp_mean": sum(float(r["ndp"]) for r in n1) / len(n1),
        "ndp_nonzero_batches": sum(float(r["ndp"]) > 0 for r in n1),
        "ndp_selected": sum(int(r["ndp_selected_count"]) for r in n1),
        "ndp_active": sum(int(r["ndp_active_count"]) for r in n1),
        "ndp_active_batches": sum(r["ndp_active"] == "True" for r in n1),
        "ndp_active_batch_rate": sum(r["ndp_active"] == "True" for r in n1) / len(n1),
        "lr_values": sorted(set(r["lr"] for r in n1)),
    })
    b0s = one(b0, "B0")
    b0s.update({"ndp_mean": 0.0, "ndp_nonzero_batches": 0, "ndp_selected": 0,
                "ndp_active": 0, "ndp_active_batches": 0, "ndp_active_batch_rate": 0.0,
                "lr_values": sorted(set(r["lr"] for r in b0))})
    return b0s, n1s


def local_rows(seed: int):
    return rows(EVAL_DIR / f"local_metrics_seed{seed}" / f"local_node_summary_seed{seed}.csv")


def ltcc(seed: int):
    return json.loads((EVAL_DIR / f"ltcc_seed{seed}.json").read_text(encoding="utf-8"))["results"]


def build_paired_csv():
    out = []
    for seed in (1234, 4321):
        loc = {(r["arm"], int(r["update"])): r for r in local_rows(seed)}
        test = {(r["arm"], int(r["update"])): r["metrics"] for r in ltcc(seed)}
        for update in UPDATES:
            b, n = loc[("B0", update)], loc[("N1", update)]
            bt, nt = test[("B0", update)], test[("N1", update)]
            out.append({
                "seed": seed, "update": update,
                "b0_mean_margin": b["mean_margin"], "n1_mean_margin": n["mean_margin"],
                "delta_mean_margin": float(n["mean_margin"]) - float(b["mean_margin"]),
                "b0_correct_relation": b["correct_relation_count"], "n1_correct_relation": n["correct_relation_count"],
                "delta_correct_relation": int(n["correct_relation_count"]) - int(b["correct_relation_count"]),
                "b0_new_error_pid": b["new_error_pid_count_vs_update0"], "n1_new_error_pid": n["new_error_pid_count_vs_update0"],
                "delta_new_error_pid": int(n["new_error_pid_count_vs_update0"]) - int(b["new_error_pid_count_vs_update0"]),
                "b0_purity10": b["purity_at_10_exported_bank"], "n1_purity10": n["purity_at_10_exported_bank"],
                "delta_purity10": float(n["purity_at_10_exported_bank"]) - float(b["purity_at_10_exported_bank"]),
                "b0_rank1": bt["rank1"], "n1_rank1": nt["rank1"], "delta_rank1": nt["rank1"] - bt["rank1"],
                "b0_mAP": bt["mAP"], "n1_mAP": nt["mAP"], "delta_mAP": nt["mAP"] - bt["mAP"],
                "b0_mINP": bt["mINP"], "n1_mINP": nt["mINP"], "delta_mINP": nt["mINP"] - bt["mINP"],
            })
    path = REPORT_DIR / "NDP_TRAINING_PAIRED_SUMMARY.csv"
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(out[0])); writer.writeheader(); writer.writerows(out)
    return out


def build_hash_csv():
    rows_out = []
    paths = [
        ROOT / "scripts/ndp_e60_minimal_pair_runner.py",
        ROOT / "scripts/ndp_training_local_metrics.py",
        ROOT / "scripts/ndp_official_ltcc_nodes.py",
        ROOT / "scripts/ndp_margin_consistency.py",
        Path("/data/DIFFER/scripts/gpc_v2_train.py"),
        Path("/data/DIFFER/scripts/phase7i_relation_specificity.py"),
        ROOT / "outputs/gpc_v2_ltcc_20261015_155822/training/two_seed/B0_seed1234/config.yml",
        ROOT / "outputs/ndp_true_multipid_20261021_103952/reference_graph/seed1234/multi_pid_reference_graph.json",
        ROOT / "outputs/ndp_true_multipid_20261021_103952/reference_graph/seed4321/multi_pid_reference_graph.json",
    ]
    for p in paths:
        rows_out.append({"path": str(p), "exists": p.is_file(), "sha256": sha256(p) if p.is_file() else ""})
    for seed in (1234, 4321):
        for arm, root in (("B0", ROOT / "outputs/ndp_overnight_20261020_201841" / f"trajectory_seed{seed}"),
                          ("N1", ROOT / "outputs" / f"ndp_training_{FULL[seed]}" / f"full_seed{seed}")):
            for update in (20, 50, 100, 200, 400):
                p = root / f"node_update{update:04d}/model.pth"
                rows_out.append({"path": str(p), "exists": p.is_file(), "sha256": sha256(p) if p.is_file() else ""})
            source = (ROOT / "outputs/gpc_v2_ltcc_20261015_155822/training/two_seed" /
                      f"B0_seed{seed}/rank0_state_epoch060.pt")
            rows_out.append({"path": str(source), "exists": source.is_file(), "sha256": sha256(source) if source.is_file() else ""})
    path = REPORT_DIR / "NDP_CHECKPOINT_AND_CODE_SHA256.csv"
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=["path", "exists", "sha256"]); writer.writeheader(); writer.writerows(rows_out)
    return rows_out


def collateral(seed: int):
    rel = rows(EVAL_DIR / f"local_metrics_seed{seed}" / f"local_relation_metrics_seed{seed}.csv")
    b = {(int(r["update"]), int(r["relation_index"])): r for r in rel if r["arm"] == "B0"}
    n = {(int(r["update"]), int(r["relation_index"])): r for r in rel if r["arm"] == "N1"}
    deltas = []
    reliable = []
    for index in range(2560):
        rb, rn = b[(400, index)], n[(400, index)]
        d = float(rn["margin"]) - float(rb["margin"])
        deltas.append(d)
        if int(rn["reference_correct"]): reliable.append((int(rn["anchor_image_index"]), d))
    by_anchor = {}
    for a, d in reliable: by_anchor.setdefault(a, []).append(d)
    multi = [v for v in by_anchor.values() if len(v) >= 2]
    others = []
    for v in multi:
        for i in range(len(v)):
            others.extend(v[:i] + v[i+1:])
    return {
        "all_relation_mean_delta": sum(deltas) / len(deltas),
        "all_relation_harm_fraction": sum(d < 0 for d in deltas) / len(deltas),
        "reliable_relation_mean_delta": sum(d for _, d in reliable) / len(reliable),
        "reliable_relation_harm_fraction": sum(d < 0 for _, d in reliable) / len(reliable),
        "multi_reliable_anchor_count": len(multi),
        "multi_reliable_other_side_mean_delta": sum(others) / len(others) if others else None,
        "multi_reliable_other_side_harm_fraction": sum(d < 0 for d in others) / len(others) if others else None,
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paired = build_paired_csv(); build_hash_csv()
    training = {seed: training_summary(seed) for seed in (1234, 4321)}
    local = {seed: local_rows(seed) for seed in (1234, 4321)}
    test = {seed: ltcc(seed) for seed in (1234, 4321)}
    coll = {seed: collateral(seed) for seed in (1234, 4321)}
    # Update-400 paired rows are the main decision comparison.
    final = [r for r in paired if int(r["update"]) == 400]
    def final_row(seed): return next(r for r in final if int(r["seed"]) == seed)
    smoke_data = {}
    smoke_grads = {}
    for seed in (1234, 4321):
        smoke_data[seed] = json.loads((ROOT / "outputs" / f"ndp_training_{SMOKE[seed]}" / f"smoke_seed{seed}" / "run_summary.json").read_text())
        diagnostic_rows = rows(ROOT / "outputs" / f"ndp_training_{SMOKE[seed]}" / f"smoke_seed{seed}" / "training_steps.csv")
        active_diag = [r for r in diagnostic_rows if r["ndp_active"] == "True"]
        smoke_grads[seed] = {
            "ndp": ", ".join(f(r["ndp_feature_grad_norm"], 5) for r in active_diag),
            "main": ", ".join(f(r["main_feature_grad_norm"], 5) for r in active_diag),
            "probe": ", ".join(f(r["ndp_probe_param_grad_norm"], 5) for r in active_diag),
        }
    full_data = {}
    for seed in (1234, 4321):
        full_data[seed] = json.loads((ROOT / "outputs" / f"ndp_training_{FULL[seed]}" / f"full_seed{seed}" / "run_summary.json").read_text())
    common = f"""# NDP 训练实验报告（2026-10-23）

代码版本：`164d7b1a3242157c2e898caffd03d65445f3ab35`（远端 HEAD 与 main 核对一致）。本轮报告目录：`{REPORT_DIR}`。

固定约束：EVA02、feat[0] 1024 维、原始 CE/Triplet/clipBio、GPC-v1 matched-budget 选择及 `0.1` 系数均由现有训练引擎提供；本轮只新增最小 NDP。NDP 使用 update0/E60 的 train-only K=5 固定图，历史正确条件为 `M_ref(update0)>0`，当前项为 live anchor 与冻结端点的同一 `mean-positive cosine - max-negative cosine` Margin，`epsilon=0.02`、`lambda_ndp=0.05`。训练关系选择没有读取未来节点、query/gallery 或测试信息。

核心命令：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \\
  scripts/ndp_e60_minimal_pair_runner.py --mode smoke|full --seed 1234|4321 \\
  --run-id <timestamp> --run-dir <new-output-dir> --diagnostic-steps 20
```

新代码和 checkpoint 的 SHA256 见 [`NDP_CHECKPOINT_AND_CODE_SHA256.csv`](NDP_CHECKPOINT_AND_CODE_SHA256.csv)；逐关系 train-only 结果见两个 `local_relation_metrics_seed*.csv`，节点汇总见 `local_node_summary_seed*.csv`。这些 CSV 未使用官方 query/gallery。
"""
    impl = common + f"""
## 1. 实现与 Smoke Test

入口：[`scripts/ndp_e60_minimal_pair_runner.py`](/home/DIFFER/scripts/ndp_e60_minimal_pair_runner.py)。每个 batch 计算原始 base loss 和原 GPC loss，然后仅对当前 batch 中出现的 anchor 查 frozen reference bank；positive/negative endpoint 不需要同 batch 出现，只有 live anchor 特征保留 autograd。参考 Margin 用 `float(...)` 固定，当前 Margin 从 `feat[0]` 计算，NDP 梯度再与主损失相加后执行一次原训练 optimizer/scaler step。

Smoke 的真实 PPU 结果：

    {md_table(["seed", "updates", "历史正确/候选", "selected", "active", "active batch", "NDP mean", "NDP feature grad", "main feature grad", "probe param grad", "restore"], [
 [1234, smoke_data[1234]["effective_updates"], f"{smoke_data[1234]['ndp_historical_correct_relations']}/2560", smoke_data[1234]["ndp_selected_relations_local_rank0"], smoke_data[1234]["ndp_active_relations_local_rank0"], pct(smoke_data[1234]["ndp_active_rate_local_rank0"]), f(smoke_data[1234]["ndp_mean_loss_local_rank0"],8), smoke_grads[1234]["ndp"], smoke_grads[1234]["main"], smoke_grads[1234]["probe"], "PASS/PASS"],
 [4321, smoke_data[4321]["effective_updates"], f"{smoke_data[4321]['ndp_historical_correct_relations']}/2560", smoke_data[4321]["ndp_selected_relations_local_rank0"], smoke_data[4321]["ndp_active_relations_local_rank0"], pct(smoke_data[4321]["ndp_active_rate_local_rank0"]), f(smoke_data[4321]["ndp_mean_loss_local_rank0"],8), smoke_grads[4321]["ndp"], smoke_grads[4321]["main"], smoke_grads[4321]["probe"], "PASS/PASS"]])}

Smoke 中两 seed 均 20/20 `step_executed=True`、LR 为 `2e-7`、AMP scale 未下降、参数有限且无 NaN/Inf。seed1234 第12步 `ndp=0.006766`、`lambda_ndp*ndp≈0.000338`，总损失高于 main loss；seed4321 第2/7步均观察到非零 NDP feature/probe 参数梯度。因此 NDP 确实经过当前 EVA02 反向并参与 optimizer.step，而不是只记录日志。

新增 focused unit test：`tests/test_ndp_minimal_pair_math.py`，`2 passed`。它验证当前改善零惩罚、退化产生当前分支梯度，以及参考/当前共用同一 Margin 定义；未重复旧的 38 项测试。

结论：Smoke PASS。此前一次 seed1234 smoke 的诊断 NaN 是诊断窗口误用绝对 global step 的代码问题，未覆盖训练结果；修正后以新时间戳 `20261023_032320` 重跑通过。
"""
    # The expression is indented by Python source indentation; remove that
    # one leading indentation so the Markdown table is rendered as a table.
    impl = impl.replace("\n    | seed | updates", "\n| seed | updates")
    (REPORT_DIR / "NDP_TRAINING_IMPLEMENTATION.md").write_text(impl, encoding="utf-8")

    table = []
    for r in paired:
        if int(r["update"]) in (0, 20, 50, 100, 200, 400):
            table.append([r["seed"], r["update"], f(r["delta_mean_margin"],7), r["delta_correct_relation"], r["delta_new_error_pid"], f(r["delta_purity10"],7), pct(r["delta_rank1"]), f(r["delta_mAP"],7), f(r["delta_mINP"],7)])
    train_table = []
    for seed in (1234, 4321):
        for arm, value in (("B0", training[seed][0]), ("N1", training[seed][1])):
            train_table.append([seed, arm, f(value["mean_loss"], 6), f(value["final_loss"], 6),
                                f(value["mean_ce"], 6), f(value["mean_triplet"], 6),
                                f(value["mean_bio"], 6), f(value["mean_gpc"], 6),
                                value["gpc_nonzero"], value["ndp_active"], value["finite"]])
    b0n1 = common + f"""
## 1. 配对协议

B0 复用了已完成且核对为 400/400 有效更新的 `outputs/ndp_overnight_20261020_201841/trajectory_seed{{1234,4321}}`；其来源 phase、E60 节点、epoch tag=61、global step=35760→36160、LR=2e-7 与 N1 一致。N1 的 E60 source 分别为两个 B0 seed 的 `rank{{0,1}}_state_epoch060.pt`，新输出位于 `outputs/ndp_training_20261023_032622`（seed1234）和 `outputs/ndp_training_20261023_033008`（seed4321）。B0 没有重跑，也没有覆盖历史结果。

训练统计：

{md_table(["seed", "arm", "mean loss", "final loss", "mean CE", "mean Triplet", "mean Bio", "mean GPC", "GPC 非零 batch", "NDP active", "finite"], train_table)}

## 2. 固定节点结果（N1-B0）

{md_table(["seed", "update", "Δ mean Margin", "Δ 正确关系", "Δ 新错误 PID", "Δ Purity@10", "Δ Rank-1", "Δ mAP", "Δ mINP"], table)}

`Δ` 均为 N1−B0；Purity@10 是固定 K=5 endpoint train-only bank 内的 top-10 同 PID 比例。完整 B0/N1 数值及逐关系 CSV 保存在本报告目录的 `NDP_TRAINING_PAIRED_SUMMARY.csv`、`../../outputs/ndp_evaluation_{REPORT_ID}/local_metrics_seed{{1234,4321}}/`。官方 LTCC 使用固定 query/gallery、clothes-changing 过滤、归一化 feat[0]、欧氏距离、无 reranking；mINP 是对每个有效 query 最后一个正样本位置的 precision 取平均。

## 3. 400 步重点

{md_table(["seed", "NDP active batch/400", "NDP active relation", "Δ mean Margin", "Δ 正确关系", "Δ 新错误 PID", "Δ Purity@10", "Δ Rank-1", "Δ mAP", "Δ mINP"], [[s, f"{full_data[s]['ndp_active_batches_local_rank0']}/400 ({pct(full_data[s]['ndp_active_rate_local_rank0'])})", full_data[s]['ndp_active_relations_local_rank0'], f(final_row(s)['delta_mean_margin'],7), final_row(s)['delta_correct_relation'], final_row(s)['delta_new_error_pid'], f(final_row(s)['delta_purity10'],7), pct(final_row(s)['delta_rank1']), f(final_row(s)['delta_mAP'],7), f(final_row(s)['delta_mINP'],7)] for s in (1234,4321)])}

seed1234 的 400 步局部关系正确数 +4、新错误 PID −4、平均 Margin +0.001174；seed4321 分别为 +10、−5、平均 Margin −0.000059。官方 Rank-1 在两个 seed 都下降（约 −0.26 和 −0.77 个百分点），而 mAP/mINP 都只有很小正变化；因此不存在两个 seed 一致的全面检索改善。

## 4. 同 anchor 其他负 PID 的局部影响（观察性）

以下只比较 B0/N1 的固定 update400 特征，不是一次网络 step 的反事实因果估计。`reliable` 指 update0 Margin>0；`other-side` 对同一 anchor 的其它历史可靠负 PID 做汇总。285/276 个 anchor 至少有两个历史可靠关系。

{md_table(["seed", "可靠关系 ΔMargin", "可靠关系变差比例", "其它可靠 PID ΔMargin", "其它可靠 PID 变差比例"], [[s, f(coll[s]["reliable_relation_mean_delta"],7), pct(coll[s]["reliable_relation_harm_fraction"]), f(coll[s]["multi_reliable_other_side_mean_delta"],7), pct(coll[s]["multi_reliable_other_side_harm_fraction"])] for s in (1234,4321)])}

seed1234 的其它可靠 PID 平均 ΔMargin 为 +0.000789，变差比例 49.45%；seed4321 为 −0.000591，变差比例 52.42%。这支持“没有可证明的普遍邻域保护”，也提示 seed4321 存在轻微附带退化信号；由于 B0/N1 是整段 continuation 配对，不能把它等同于 NDP 单条关系的独立因果损害。

## 5. 统计限制

这里只有两个 seed、单个 400-step continuation 和固定节点配对差值，没有把两个 seed 当作充分样本做显著性检验，也没有用未来节点选 mask。PPU 完整模型 A/A 约 0.4% 的既有差异仍限制小幅结果的因果解释；本轮 N1/B0 是探索性对照，不能声称 NDP 已改善 LTCC。
"""
    (REPORT_DIR / "NDP_400STEP_B0_VS_N1.md").write_text(b0n1, encoding="utf-8")

    decision = common + f"""
## 结论摘要

| 问题 | 证据结论 |
|---|---|
| NDP 是否真正参与训练 | 是。两 seed 均有真实激活、非零 live feature/parameter gradient、20/400 有效 optimizer steps，且 checkpoint 保存/恢复 PASS。 |
| 有多少有效激活 | 400 步 rank0 统计：seed1234 为 54/400 batch、135 条关系；seed4321 为 42/400 batch、98 条关系。候选固定为 2560，历史正确分别 1197/1158。 |
| 是否改善局部身份判别 | 有限且不一致。400 步正确关系 +4/+10；平均 Margin +0.001174/−0.000059；Purity@10 −0.000977/−0.001172。 |
| LTCC Rank-1/mAP/mINP | Rank-1 两 seed 均下降；mAP 与 mINP 两 seed 均只有约 0.00007–0.00149 的小幅增加，不能视为可靠收益。 |
| 是否损害附近其他身份 | 固定 K=5 邻域的整体 Purity@10 两 seed 均轻微下降；这提示存在微弱局部代价，但不能仅由该 proxy 断言结构性伤害。逐关系证据在 CSV，且没有执行网络一步后的反事实邻域隔离实验。 |
| 两 seed 是否一致 | 只有“Rank-1 不升”一致；局部平均 Margin、Purity 和其它检索指标不形成一致正向模式。 |
| 是否值得继续 10/60 epoch | 当前 NO-GO。先不扩展 10 或 60 epoch，不增加复杂双侧/多 PID损失；应先修正/验证训练时关系覆盖和邻域 collateral 的独立因果诊断。 |
| 下一步最值得修改哪一处 | 不是阈值。优先记录每个 NDP active relation 的 anchor/negative PID，并在同一 batch/同一冻结状态下做目标关系与同 anchor 其他 4 个 PID 的梯度预算匹配；只有确认 collateral 不是由 endpoint 冻结近似或低覆盖造成，才考虑新的候选。 |

## 门禁判断

- NDP_MECHANISM：`INCONCLUSIVE`。
- A/A_REPRODUCIBILITY：`FAIL`（沿用既有完整模型约 0.4% A/A 限制，本轮未重做 A/A）。
- CONTROLLED_SHORT_PILOT：`NO_GO` 作为进一步正式扩展门禁；本轮 400 步本身是已授权的探索性 pilot，已完成且不应包装为正向证明。

## 因果边界

本轮证明了“最小 NDP 能以固定历史 bank 进入真实 EVA02 optimizer.step”，没有证明“选择性保护带来 LTCC 泛化收益”。N1 的局部正确关系增加可能来自关系重排，而非稳定识别结构；官方 Rank-1 在两个 seed 均下降，足以阻止继续堆叠复杂模块。mAP/mINP 的微小增加小于目前可由 PPU A/A 与两 seed 波动解释的范围，不能据此宣称收益。

邻域审计的 update400 配对 proxy 显示：seed1234 其它历史可靠负 PID 的平均 ΔMargin 为 +0.000789、变差比例 49.45%；seed4321 为 −0.000591、变差比例 52.42%。这不是单侧 NDP 的独立因果证明，但已足以要求下一轮先做逐 active relation 的邻域梯度预算审计。

失败/限制如实记录：首次官方评估尝试分别遇到 YACS 类型和错误 caption 路径，随后使用训练侧 Bio-only 构造及服务器实际 sidecar 路径修复；未加载错误模型、未产生训练结果。首次 seed1234 smoke 的诊断字段 NaN 已修复并以新目录重跑；失败目录保留。
"""
    (REPORT_DIR / "NDP_EXPERIMENT_DECISION.md").write_text(decision, encoding="utf-8")
    print(json.dumps({"report_dir": str(REPORT_DIR), "paired_csv": str(REPORT_DIR / "NDP_TRAINING_PAIRED_SUMMARY.csv")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
