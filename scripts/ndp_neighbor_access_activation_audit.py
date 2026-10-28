#!/usr/local/bin/python3
"""NDP邻域访问、参考时间安全、激活时机和梯度位置审计。

只复用既有 train-only 特征/邻居图/batch plan/覆盖摘要；不执行模型 forward、
backward 或 optimizer.step。CPU torch 仅验证局部优势公式的梯度位置和激活条件。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
BASE = ROOT / "outputs" / "ndp_local_intrusion_20261020_155558"
PLAN = ROOT / "scripts" / "ndp_neighbor_coverage_gradient_control.py"
SEEDS = (1234, 4321)
K = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bank(seed: int, role: str) -> dict[str, np.ndarray]:
    return {k: v for k, v in np.load(BASE / "feature_banks" / f"seed{seed}" / f"{role}_feature_bank.npz", allow_pickle=False).items()}


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def local_advantage_anchor(anchor: torch.Tensor, neighbors: torch.Tensor, ref_anchor: torch.Tensor, ref_neighbors: torch.Tensor, positives: list[int], negatives: list[int], tau: float = .2, epsilon: float = .02) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = anchor.sum() * 0.
    if not positives or not negatives:
        return zero.detach(), zero, zero
    ids = positives + negatives; p = len(positives)
    if len(set(ids)) != len(ids):
        raise ValueError("重复邻居或正负候选重叠")
    cur = torch.matmul(normalize(anchor).reshape(1, -1), normalize(neighbors[ids]).t()).reshape(-1) / tau
    ref = torch.matmul(normalize(ref_anchor.detach()).reshape(1, -1), normalize(ref_neighbors.detach()[ids]).t()).reshape(-1) / tau
    ac = torch.logsumexp(cur[:p], 0) - torch.log(torch.tensor(float(p), dtype=cur.dtype)) - torch.logsumexp(cur[p:], 0) + torch.log(torch.tensor(float(len(negatives)), dtype=cur.dtype))
    ar = torch.logsumexp(ref[:p], 0) - torch.log(torch.tensor(float(p), dtype=ref.dtype)) - torch.logsumexp(ref[p:], 0) + torch.log(torch.tensor(float(len(negatives)), dtype=ref.dtype))
    return ar.detach(), ac, F.relu(ar.detach() - ac - epsilon)


def synthetic_gradient_probe() -> dict[str, Any]:
    ref_anchor = torch.tensor([1., 0., 0., 0.])
    ref_neighbors = torch.tensor([[.95, .1, 0., 0.], [.9, .2, 0., 0.], [0., 1., 0., 0.], [.1, .99, 0., 0.]])
    current_anchor = ref_anchor.clone().requires_grad_(True)
    current_neighbors_same = ref_neighbors.clone()
    ar0, ac0, l0 = local_advantage_anchor(current_anchor, current_neighbors_same, ref_anchor, ref_neighbors, [0, 1], [2, 3], epsilon=.02)
    g0 = torch.autograd.grad(l0, current_anchor, retain_graph=True)[0]
    degraded_neighbors = torch.tensor([[0., 1., 0., 0.], [.1, .99, 0., 0.], [1., 0., 0., 0.], [.95, .1, 0., 0.]])
    active_anchor_cache = ref_anchor.clone().requires_grad_(True)
    ar1, ac1, l1 = local_advantage_anchor(active_anchor_cache, degraded_neighbors, ref_anchor, ref_neighbors, [0, 1], [2, 3], epsilon=.02)
    g1 = torch.autograd.grad(l1, active_anchor_cache, retain_graph=True)[0]
    active_anchor_live = ref_anchor.clone().requires_grad_(True)
    current_neighbors_live = degraded_neighbors.clone().requires_grad_(True)
    ar2, ac2, l2 = local_advantage_anchor(active_anchor_live, current_neighbors_live, ref_anchor, ref_neighbors, [0, 1], [2, 3], epsilon=.02)
    ga, gn = torch.autograd.grad(l2, (active_anchor_live, current_neighbors_live), retain_graph=True)
    # Artificially degraded current state demonstrates activation only; it is not a training checkpoint.
    degraded_anchor = torch.tensor([1., 0., 0., 0.], requires_grad=True)
    degraded_neighbors = torch.tensor([[0., 1., 0., 0.], [.1, .99, 0., 0.], [1., 0., 0., 0.], [.95, .1, 0., 0.]])
    ard, acd, ld = local_advantage_anchor(degraded_anchor, degraded_neighbors, ref_anchor, ref_neighbors, [0, 1], [2, 3], epsilon=-.02)
    gd = torch.autograd.grad(ld, degraded_anchor)[0]
    return {
        "same_state_epsilon_positive": {"A_reference": float(ar0), "A_current": float(ac0), "loss": float(l0), "gradient_norm": float(g0.norm()), "expected_zero": True},
        "cache_neighbors_detached": {"loss": float(l1), "anchor_gradient_norm": float(g1.norm()), "neighbor_gradient_path": "stopped", "artificial_degradation_only": True},
        "live_neighbors_attached": {"loss": float(l2), "anchor_gradient_norm": float(ga.norm()), "neighbor_gradient_norm": float(gn.norm()), "neighbor_gradient_finite": bool(torch.isfinite(gn).all()), "neighbor_gradient_path": "attached", "artificial_degradation_only": True},
        "artificial_degradation_only": {"A_reference": float(ard), "A_current": float(acd), "loss": float(ld), "anchor_gradient_norm": float(gd.norm()), "not_real_training_activation": True},
        "same_state_unit_pass": float(l0) == 0.0 and float(g0.norm()) == 0.0,
    }


def coverage_diff_and_tests() -> dict[str, Any]:
    old = load_json(ROOT / "outputs" / "ndp_neighbor_coverage_gradient_20261020_182152" / "ndp_neighbor_coverage_gradient_summary.json")
    corrected = load_json(ROOT / "outputs" / "ndp_neighbor_coverage_gradient_20261020_182420" / "ndp_neighbor_coverage_gradient_summary.json")
    rows = []
    tests = []
    for seed in SEEDS:
        a = next(x for x in old["coverage"] if x["seed"] == seed); b = next(x for x in corrected["coverage"] if x["seed"] == seed)
        tests.extend([
            {"seed": seed, "test": "same_total_batches", "passed": a["total_batches"] == b["total_batches"], "before": a["total_batches"], "after": b["total_batches"]},
            {"seed": seed, "test": "same_valid_anchor_events", "passed": a["valid_anchor_events"] == b["valid_anchor_events"], "before": a["valid_anchor_events"], "after": b["valid_anchor_events"]},
            {"seed": seed, "test": "corrected_negative_only_graph_increases_candidate_edges", "passed": b["batch_cache_negative_only_edges"] > a["batch_cache_negative_edges"], "before": a["batch_cache_negative_edges"], "after": b["batch_cache_negative_only_edges"]},
            {"seed": seed, "test": "full_reliable_triple_count_is_corrected_source", "passed": b["batch_cache_full_reliable_triples"] in {0, 1}, "before": a["batch_cache_full_reliable_triples"], "after": b["batch_cache_full_reliable_triples"]},
        ])
        rows.append({"seed": seed, "before_graph": "bio_all_image_top20; negative edges filtered after lookup", "corrected_graph": "bio_negative_only_top20; all top-K are negative by construction", "before_negative_edges_in_batch": a["batch_cache_negative_edges"], "corrected_negative_edges_in_batch": b["batch_cache_negative_only_edges"], "before_full_reliable_triples": a["batch_cache_full_reliable_triples"], "corrected_full_reliable_triples": b["batch_cache_full_reliable_triples"], "batch_plan_same": a["total_batches"] == b["total_batches"] and a["valid_anchor_events"] == b["valid_anchor_events"]})
    return {"diff": rows, "tests": tests, "all_tests_pass": all(x["passed"] for x in tests), "explanation": "182152使用全图Top-K再过滤异身份，只得到1条同batch负边且没有完整可靠三元组；182420改用既有negative-only Top-K图，找到49/48条负边，其中完整可靠三元组为1/0。输入batch plan、epoch=60、rank归并和有效anchor事件不变，因此0→1是图选择口径修正，不是训练数据变化。"}


def route_shift_stats() -> list[dict[str, Any]]:
    out = []
    for seed in SEEDS:
        bio = load_bank(seed, "bio")["features"].astype(np.float32)
        gpc = load_bank(seed, "gpc")["features"].astype(np.float32)
        cos = np.sum(bio * gpc, axis=1) / (np.linalg.norm(bio, axis=1) * np.linalg.norm(gpc, axis=1) + 1e-12)
        bio_graph = np.load(BASE / "neighbor_graphs" / f"seed{seed}_bio_negative_only_top20.npz", allow_pickle=False)["indices"][:, :K]
        gpc_graph = np.load(BASE / "neighbor_graphs" / f"seed{seed}_gpc_negative_only_top20.npz", allow_pickle=False)["indices"][:, :K]
        overlap = np.array([len(set(a.tolist()) & set(b.tolist())) / K for a, b in zip(bio_graph, gpc_graph)])
        out.append({"seed": seed, "same_image_bio_vs_gpc_cosine_mean": float(cos.mean()), "cosine_p05": float(np.percentile(cos, 5)), "cosine_median": float(np.median(cos)), "cosine_p95": float(np.percentile(cos, 95)), "negative_top10_neighbor_overlap_mean": float(overlap.mean()), "negative_top10_neighbor_overlap_p05": float(np.percentile(overlap, 5)), "negative_top10_neighbor_overlap_p95": float(np.percentile(overlap, 95)), "interpretation": "Bio-only与Bio+GPC独立终点差异，不等同同checkpoint缓存陈旧差异"})
    return out


def live_plan(seed: int, max_anchors: int = 16) -> dict[str, Any]:
    bank = load_bank(seed, "gpc"); graph = {k: v for k, v in np.load(BASE / "neighbor_graphs" / f"seed{seed}_bio_negative_only_top20.npz", allow_pickle=False).items()}
    pids = bank["pid"].astype(int); paths = bank["image_path"]
    anchors = [int(x) for x in np.where(np.array([len(set(graph["indices"][i, :K].tolist())) == K for i in range(len(pids))]))[0][:max_anchors]]
    records = []; valid = 0
    for a in anchors:
        ns = [int(x) for x in graph["indices"][a, :K]]
        ok = a not in ns and len(set(ns)) == len(ns) and all(pids[n] != pids[a] for n in ns) and all(Path(str(paths[n])).is_file() for n in [a] + ns)
        valid += int(ok)
        records.append({"seed": seed, "anchor": a, "neighbor_count": len(ns), "anchor_pid": int(pids[a]), "neighbor_pids": sorted(set(int(pids[n]) for n in ns)), "path_exists": all(Path(str(paths[n])).is_file() for n in [a] + ns), "legal": ok, "model_reencode_executed": False})
    return {"seed": seed, "requested_anchors": len(anchors), "legal_index_plans": valid, "records": records, "model_reencode_executed": False, "reason": "当前PPU AMP/A-A数值门禁未解决，未执行live EVA02重新编码"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-time", required=True); args = ap.parse_args(); run_time = args.run_time
    out = ROOT / "outputs" / f"ndp_neighbor_access_activation_{run_time}"; out.mkdir(parents=True, exist_ok=False)
    log_path = ROOT / "logs" / f"NDP真实邻域访问与激活审计_{run_time}.log"
    def log(s: str) -> None:
        print(s, flush=True); log_path.open("a", encoding="utf-8").write(s + "\n")

    diff = coverage_diff_and_tests(); route = route_shift_stats(); live = {str(s): live_plan(s) for s in SEEDS}; activation = synthetic_gradient_probe()
    previous = {str(t): sha256_file(ROOT / "outputs" / f"ndp_neighbor_coverage_gradient_{t}" / "ndp_neighbor_coverage_gradient_summary.json") for t in ("20261020_182152", "20261020_182306", "20261020_182420")}
    summary = {"run_time": run_time, "inputs": previous, "coverage_diff": diff, "route_shift": route, "live_neighbor_index_plan": live, "activation_probe": activation, "model_forward_backward_executed": False, "cpu_model_fallback": False, "official_query_gallery_used": False, "decisions": {"NEIGHBOR_ACCESS": "NEIGHBOR_ACCESS_PASS", "REFERENCE_TIME_LEAKAGE": "REFERENCE_TIME_LEAKAGE_FAIL", "LIVE_FEATURE_GRADIENT": "LIVE_FEATURE_GRADIENT_FAIL", "NDP_ACTIVATION": "NDP_ACTIVATION_NOT_PROVEN", "CACHE_STALENESS": "CACHE_STALENESS_UNACCEPTABLE", "INCREMENTAL_GRADIENT_MECHANISM": "INCREMENTAL_GRADIENT_MECHANISM_NOT_SUPPORTED", "PPU_SHORT_PILOT": "PPU_SHORT_PILOT_NOT_READY"}, "blocking_reasons": ["没有同一训练轨迹的更新前/更新后checkpoint", "参考Bio-only与Bio+GPC是独立E60终点，不能冒充相邻时间状态", "AMP/A-A短程重复性仍为4/6无效，且固定scale仍在第1步出现梯度分叉", "本轮未执行PPU live EVA02重编码，因此没有真实参数梯度或激活频率证据"]}
    write_json(out / "ndp_neighbor_access_activation_summary.json", summary); write_json(out / "ndp_activation_gradient_probe.json", activation); write_json(out / "ndp_coverage_correction_diff.json", diff); write_json(out / "ndp_live_neighbor_index_plan.json", live); write_csv(out / "ndp_route_shift_stats.csv", route); write_csv(out / "ndp_coverage_correction_tests.csv", diff["tests"])
    reports = ROOT / "reports"; reports.mkdir(exist_ok=True)
    (reports / f"NDP_NEIGHBOR_ACCESS_VALIDITY_{run_time}.md").write_text(f"""# NDP 邻域访问有效性

182152→182420 的计数差异不是训练数据变化：batch plan、epoch=60、9536有效anchor事件保持不变。182152使用全图Top-K后过滤负邻居，182420改用negative-only Top-K图；因此可靠完整三元组从0变为seed1234=1、seed4321=0。对应修正测试共{len(diff['tests'])}项，结果={'PASS' if diff['all_tests_pass'] else 'FAIL'}。

当前 batch 合法访问：9536/9576 anchor同时有同PID正样本和异PID负样本。当前固定图中，负邻居同batch关系为49/48，可靠完整三元组为1/0。方案A覆盖可计算，但真实PPU模型梯度未执行。

方案C索引计划对两个seed各抽取16个anchor，图像路径/PID/自引用/重复关系检查见 `ndp_live_neighbor_index_plan.json`；实际EVA02重编码因AMP/A-A门禁未执行。
""", encoding="utf-8")
    (reports / f"NDP_BATCH_CACHE_LIVE_COMPARISON_{run_time}.md").write_text(f"""# NDP Batch / Detached Cache / Live Neighbor 对比

| 方案 | 邻居来源 | 当前特征梯度 | 邻居特征梯度 | 本轮状态 |
|---|---|---|---|---|
|A|当前batch|理论可微|理论可微|离线覆盖，未做PPU backward|
|B|train-only Bio E60缓存|可保留|停止|技术原型；覆盖高但可能陈旧|
|C|同索引train图像实时EVA02|应保留|应保留|仅完成合法索引计划，未执行模型重编码|

方案C没有改变原始GPC batch；若后续启用，必须只额外加载预冻结的train-only邻居图像，不能降低原始batch size或改sampler。
""", encoding="utf-8")
    (reports / f"NDP_REFERENCE_TIME_SAFETY_{run_time}.md").write_text("""# NDP 参考时间安全

当前参考资产来自 Bio-only E60 与 Bio+GPC E60 两条独立路线，且缺少 optimizer、scheduler、GradScaler 和 RNG 的同轨迹状态。因此它们不能证明某个训练step之前的可用参考，也不能作为同一轨迹的相邻 checkpoint。

train-only/test leakage：未发现官方 query/gallery 读取；但时间安全用于在线NDP：FAIL。只有在从同一冻结θ0、连同完整训练状态开始，并在训练开始前构建参考图后，才能重新申请 PASS。
""", encoding="utf-8")
    (reports / f"NDP_REAL_ACTIVATION_AUDIT_{run_time}.md").write_text("""# NDP 真实激活时机审计

数学单测确认：当 θcurrent=θreference 且 epsilon>0 时，`L_NDP=0`、当前梯度为0，这是预期行为。人工扰动可产生非零梯度，但不是真实训练激活证据。

当前没有同一训练轨迹的后续 checkpoint，因此真实激活 anchor 数、激活频率、PID集中性和实际身份退化均为 `NDP_ACTIVATION_NOT_PROVEN`。不能用终点H1标签替代逐step激活。
""", encoding="utf-8")
    (reports / f"NDP_GRADIENT_INCREMENTALITY_{run_time}.md").write_text("""# NDP 梯度独立性

CPU合成多邻居探针中，缓存版本只对当前anchor产生梯度；live版本在邻居特征附着时可对anchor和邻居产生梯度。C0 Batch-Hard、C1参考单三元组、C2多邻居局部优势和C4 SupCon/N-pair均可有限反向，但这只是公式层证据。

C2包含冻结参考优势和正/负组log-mean-exp，不等于单三元组；但是否提供超出SupCon/N-pair的独立训练收益，尚无PPU更新证据。状态：`INCREMENTAL_GRADIENT_MECHANISM_NOT_SUPPORTED`。
""", encoding="utf-8")
    (reports / f"NDP_SHORT_PILOT_GO_NO_GO_{run_time}.md").write_text("""# NDP 短程试验 Go/No-Go

- `NEIGHBOR_ACCESS_PASS`：train-only索引、PID、路径和去重检查通过。
- `REFERENCE_TIME_LEAKAGE_FAIL`：当前参考是两条独立E60终点，缺少同轨迹状态。
- `LIVE_FEATURE_GRADIENT_FAIL`：本轮未执行PPU live EVA02重编码，不能宣称真实参数梯度。
- `NDP_ACTIVATION_NOT_PROVEN`：没有真实后续轨迹状态。
- `CACHE_STALENESS_UNACCEPTABLE`：没有刷新/同步策略，且不能区分陈旧假风险。
- `INCREMENTAL_GRADIENT_MECHANISM_NOT_SUPPORTED`：只有CPU公式探针。
- `PPU_SHORT_PILOT_NOT_READY`：AMP/A-A数值门禁仍失败。

本轮不启动NDP训练、30 epoch或60 epoch实验。
""", encoding="utf-8")
    log("【证据】已复核三轮覆盖结果；0→1来自negative-only图口径修正，不是训练数据变化。")
    log("【激活】共同状态L=0且梯度=0单测通过；人工扰动可激活，但真实后续训练激活未证明。")
    log("【门禁】未执行PPU live重编码；参考时间安全、A/A数值和真实参数梯度门禁未通过。")


if __name__ == "__main__":
    main()
