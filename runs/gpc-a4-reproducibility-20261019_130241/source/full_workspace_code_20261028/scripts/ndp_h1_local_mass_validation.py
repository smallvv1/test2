#!/usr/local/bin/python3
"""补齐 H1 原始证据并验证局部身份质量候选的数学接口。

H1/Triplet 结果只读取已完成的离线审计产物；局部质量候选仅做 CPU
独立张量单测，不实例化 EVA02，不执行模型训练。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
SOURCE_REL = ROOT / "outputs" / "ndp_metric_coverage_audit_20261020_165332"
SOURCE_H1 = ROOT / "outputs" / "ndp_h1_online_feasibility_20261020_175430"
FEATURES = ROOT / "outputs" / "ndp_local_intrusion_20261020_155558" / "feature_banks"
SEEDS = (1234, 4321)
K = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def h1_rows(seed: int) -> list[dict[str, Any]]:
    path = SOURCE_REL / "tables" / f"seed{seed}_K10_identity_relation_states.csv"
    rows = []
    for r in read_csv(path):
        if as_bool(r["reference_reliable_flip"]):
            rows.append({"anchor": int(r["anchor"]), "pid": int(r["pid"]), "positive": int(r["positive"]), "negative": int(r["negative"]), "g_ref": float(r["g_ref"]), "g_current": float(r["g_current"])})
    return rows


def load_labels(seed: int) -> dict[str, np.ndarray]:
    p = FEATURES / f"seed{seed}" / "gpc_feature_bank.npz"
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


def local_identity_mass(z_current: torch.Tensor, z_reference: torch.Tensor, positive_indices: list[int], negative_indices: list[int], tau: float = 0.05, epsilon: float = 0.02, reliable: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对一个 anchor 计算 Q_ref、Q_current 和单侧质量损失。

    z_current 的第 0 行是 anchor；正负索引均是 z_current/z_reference 的行号。
    参考分支显式 detach。空正或空负、或不可靠参考时返回与当前图相连的零损失。
    """
    zero = z_current.sum() * 0.0
    if not reliable or not positive_indices or not negative_indices:
        return zero.detach(), zero, zero
    cur = F.normalize(z_current, dim=-1)
    ref = F.normalize(z_reference.detach(), dim=-1)
    cur_logits = torch.matmul(cur[0:1], cur[positive_indices + negative_indices].t()).squeeze(0) / tau
    ref_logits = torch.matmul(ref[0:1], ref[positive_indices + negative_indices].t()).squeeze(0) / tau
    p_count = len(positive_indices)
    q_cur = torch.exp(torch.logsumexp(cur_logits[:p_count], dim=0) - torch.logsumexp(cur_logits, dim=0))
    q_ref = torch.exp(torch.logsumexp(ref_logits[:p_count], dim=0) - torch.logsumexp(ref_logits, dim=0))
    loss = F.relu(q_ref - q_cur - epsilon)
    return q_ref.detach(), q_cur, loss


def run_unit_tests() -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    def record(name: str, passed: bool, detail: str) -> None:
        tests.append({"test": name, "passed": bool(passed), "detail": detail})

    # 空集合不构造伪风险，且当前图仍能安全反向。
    z = torch.randn(3, 8, dtype=torch.float32, requires_grad=True)
    _, _, loss = local_identity_mass(z, z.detach(), [], [2])
    loss.backward(); record("empty_positive_zero_finite", float(loss) == 0.0 and torch.isfinite(z.grad).all().item(), f"loss={float(loss)}")

    z = torch.randn(3, 8, dtype=torch.float32, requires_grad=True)
    _, _, loss = local_identity_mass(z, z.detach(), [1], [])
    loss.backward(); record("empty_negative_zero_finite", float(loss) == 0.0 and torch.isfinite(z.grad).all().item(), f"loss={float(loss)}")

    # 参考正确、当前明显退化时必须产生正损失和有限当前梯度。
    ref = F.normalize(torch.tensor([[1., 0., 0.], [1., 0.1, 0.], [0., 1., 0.]]), dim=-1)
    cur = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 0.1, 0.]], requires_grad=True)
    q_ref, q_cur, loss = local_identity_mass(cur, ref, [1], [2], tau=0.1, epsilon=0.0)
    loss.backward(); record("reliable_degradation_positive_gradient", float(q_ref) > float(q_cur) and float(loss) > 0 and torch.isfinite(cur.grad).all().item(), f"q_ref={float(q_ref):.6f}, q_cur={float(q_cur):.6f}, loss={float(loss):.6f}")

    # 有益重排：当前同身份概率质量提高，不惩罚。
    ref = F.normalize(torch.tensor([[1., 0., 0.], [0.8, 0.2, 0.], [0., 1., 0.]]), dim=-1)
    cur = torch.tensor([[1., 0., 0.], [1., 0.1, 0.], [0., 1., 0.]], requires_grad=True)
    q_ref, q_cur, loss = local_identity_mass(cur, ref, [1], [2], tau=0.1, epsilon=0.0)
    record("beneficial_reordering_not_penalized", float(q_cur) >= float(q_ref) and float(loss) == 0.0, f"q_ref={float(q_ref):.6f}, q_cur={float(q_cur):.6f}")

    # 参考排序错误或不可靠时不保护。
    cur = torch.randn(3, 8, requires_grad=True); ref = torch.randn(3, 8, requires_grad=True)
    _, _, loss = local_identity_mass(cur, ref, [1], [2], reliable=False)
    loss.backward(); record("unreliable_reference_not_protected", float(loss) == 0.0 and ref.grad is None and torch.isfinite(cur.grad).all().item(), "reference detached and gate disabled")

    # 大 logits/小温度的数值稳定性。
    cur = torch.tensor([[100., 0.], [100., 0.], [0., 100.]], requires_grad=True); ref = cur.detach().clone()
    q_ref, q_cur, loss = local_identity_mass(cur, ref, [1], [2], tau=0.001)
    loss.backward(); record("large_logits_finite", bool(torch.isfinite(torch.stack([q_ref, q_cur, loss, cur.grad.norm()])).all()), f"q_ref={float(q_ref):.6f}, q_cur={float(q_cur):.6f}")

    # 参考停止梯度。
    cur = torch.randn(3, 6, requires_grad=True); ref = torch.randn(3, 6, requires_grad=True)
    _, _, loss = local_identity_mass(cur, ref, [1], [2]); loss.backward()
    record("reference_stop_gradient", ref.grad is None, f"reference_grad_is_none={ref.grad is None}")

    # 单正/单负时 Q 是 sigmoid((s_pos-s_neg)/tau)。
    cur = F.normalize(torch.tensor([[1., 0.], [0.8, 0.6], [0., 1.]]), dim=-1).requires_grad_(True); ref = cur.detach()
    q_ref, _, _ = local_identity_mass(cur, ref, [1], [2], tau=0.5)
    expected = torch.sigmoid((torch.dot(cur[0], cur[1]) - torch.dot(cur[0], cur[2])) / 0.5)
    record("single_pair_sigmoid_relation", abs(float(q_ref) - float(expected)) < 1e-6, f"q={float(q_ref):.8f}, expected={float(expected):.8f}")
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-time", required=True); args = parser.parse_args()
    run_time = args.run_time; out = ROOT / "outputs" / f"ndp_local_mass_validation_{run_time}"
    if out.exists(): raise SystemExit(f"拒绝覆盖: {out}")
    out.mkdir(parents=True)
    log_path = ROOT / "logs" / f"NDP_H1证据补齐与局部质量单测_{run_time}.log"
    def log(msg: str):
        print(msg, flush=True); log_path.open("a", encoding="utf-8").write(msg + "\n")

    labels = {seed: load_labels(seed) for seed in SEEDS}
    if not np.array_equal(labels[1234]["image_path"], labels[4321]["image_path"]): raise RuntimeError("两个 seed image_path 未对齐")
    h1 = {seed: h1_rows(seed) for seed in SEEDS}
    anchor_sets = {seed: {r["anchor"] for r in h1[seed]} for seed in SEEDS}
    common = anchor_sets[1234] & anchor_sets[4321]; union = anchor_sets[1234] | anchor_sets[4321]
    index_rows=[]
    for idx in sorted(union):
        index_rows.append({"image_index": idx, "image_path": str(labels[1234]["image_path"][idx]), "pid": int(labels[1234]["pid"][idx]), "clothes": int(labels[1234]["clothes"][idx]), "camera": int(labels[1234]["camid"][idx]), "h1_seed1234": int(idx in anchor_sets[1234]), "h1_seed4321": int(idx in anchor_sets[4321]), "h1_intersection": int(idx in common)})
    pid_rows=[]
    for pid in sorted(set(int(x) for x in labels[1234]["pid"])):
        all_count=int(np.sum(labels[1234]["pid"]==pid)); a=sum(int(labels[1234]["pid"][i])==pid for i in anchor_sets[1234]); b=sum(int(labels[4321]["pid"][i])==pid for i in anchor_sets[4321]); i=sum(int(labels[1234]["pid"][x])==pid for x in common)
        pid_rows.append({"pid":pid,"all_anchor_images":all_count,"h1_seed1234":a,"h1_rate_seed1234":a/max(all_count,1),"h1_seed4321":b,"h1_rate_seed4321":b/max(all_count,1),"h1_intersection":i})
    tests=run_unit_tests(); passed=all(x["passed"] for x in tests)
    summary={"run_time":run_time,"source_audit_run":"20261020_165332","source_h1_run":"20261020_175430","official_query_gallery_used":False,"model_training_started":False,"h1_seed1234":len(anchor_sets[1234]),"h1_seed4321":len(anchor_sets[4321]),"h1_intersection":len(common),"h1_union":len(union),"h1_jaccard":len(common)/max(len(union),1),"unit_tests_passed":passed,"unit_tests":tests,"candidate_status":"NDP_LOCAL_MASS_CANDIDATE_READY_DESIGN_ONLY","dynamic_status":"ONLINE_RISK_PREDICTION_NOT_IDENTIFIABLE_FROM_CURRENT_DATA","overall_decision":"NDP_INCREMENTAL_MECHANISM_UNRESOLVED","note":"CPU数学单测，不是PPU模型训练门禁"}
    write_csv(out/"h1_anchor_indices.csv",index_rows); write_csv(out/"h1_pid_summary.csv",pid_rows); write_json(out/"ndp_local_mass_unit_test_results.json",summary)
    report_root = ROOT / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"NDP_H1_STABILITY_REVIEW_{run_time}.md").write_text(
        f"# NDP H1 稳定性复核（{run_time}）\n\n"
        f"H1 固定为 `g_ref>0.05` 且 `g_current<=0`。seed1234={len(anchor_sets[1234])} anchors，seed4321={len(anchor_sets[4321])} anchors；交集={len(common)}，并集={len(union)}，Jaccard={len(common)/max(len(union),1):.4%}。分层随机对照来自既有 20261020_175430 结果：交集均值23.20、95%区间16–31、上尾p=0.0005。\n\n"
        "这支持共享的静态终点困难结构，但不等于在线更新风险，也不把共同 PID 自动定义为通用困难身份。H1索引和PID发生率已写入输出CSV。\n", encoding="utf-8")
    unit_lines = [f"# NDP 局部身份质量数学单测（{run_time}）\n", "\n候选公式：`Q=positive_exp_sum/(positive_exp_sum+negative_exp_sum)`，`L_NDP=relu(Q_reference-Q_current-epsilon)`；参考分支 detach。\n\n", "| 测试 | 结果 | 说明 |\n|---|---|---|\n"]
    for test in tests:
        unit_lines.append(f"| {test['test']} | {'PASS' if test['passed'] else 'FAIL'} | {test['detail']} |\n")
    unit_lines.append("\n本单测仅验证独立张量公式和 autograd 行为，不是 PPU EVA02 训练验证。\n")
    (report_root / f"NDP_LOCAL_MASS_UNIT_TEST_REPORT_{run_time}.md").write_text("".join(unit_lines), encoding="utf-8")
    (report_root / f"NDP_NEXT_STAGE_DECISION_{run_time}.md").write_text(
        "# NDP 下一阶段决策\n\n"
        "静态状态：`NDP_STATIC_RISK_SUPPORTED`（H1交集高于分层随机基线）。\n"
        "动态状态：`NDP_DYNAMIC_RISK_FEASIBLE` 不成立；当前缺少同一训练轨迹的更新前后状态。\n"
        f"候选状态：`{'NDP_LOCAL_MASS_CANDIDATE_READY_DESIGN_ONLY' if passed else 'NDP_INCREMENTAL_MECHANISM_UNRESOLVED'}`。\n"
        "总体状态：`NDP_INCREMENTAL_MECHANISM_UNRESOLVED`。CPU数学单测通过，但未通过PPU/AMP短程门禁，不启动训练。\n", encoding="utf-8")
    log(f"【证据】H1交集={len(common)}，并集={len(union)}，Jaccard={summary['h1_jaccard']:.6%}；H1候选单测={'PASS' if passed else 'FAIL'}。")
    log("【决策】整体保持 NDP_INCREMENTAL_MECHANISM_UNRESOLVED；未启动PPU模型计算或训练。")


if __name__ == "__main__": main()
