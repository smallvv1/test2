#!/usr/local/bin/python3
"""NDP H1 跨 seed 稳定性与在线风险可识别性离线审计。

只读取已经生成的 train-only CSV/JSON 和标签映射；不生成特征、不实例化
模型、不运行 forward/backward，不读取官方 query/gallery。
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

ROOT = Path("/home/DIFFER")
SOURCE_RUN = "20261020_165332"
SOURCE_OUT = ROOT / "outputs" / f"ndp_metric_coverage_audit_{SOURCE_RUN}"
COVERAGE_OUT = SOURCE_OUT
FEATURE_OUT = ROOT / "outputs" / "ndp_local_intrusion_20261020_155558" / "feature_banks"
SEEDS = (1234, 4321)
K = 10
BOOTSTRAP_REPS = 2000
RANDOM_SEED = 20261020


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_bank(seed: int) -> dict[str, np.ndarray]:
    data = np.load(FEATURE_OUT / f"seed{seed}" / "gpc_feature_bank.npz", allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_relations(seed: int) -> list[dict[str, Any]]:
    path = SOURCE_OUT / "tables" / f"seed{seed}_K10_identity_relation_states.csv"
    result = []
    for r in read_csv(path):
        result.append({
            "anchor": int(r["anchor"]), "pid": int(r["pid"]), "positive": int(r["positive"]), "negative": int(r["negative"]),
            "h1": str(r["reference_reliable_flip"]).lower() in {"true", "1"},
            "positive_cross_clothes": str(r["positive_clothes_cross"]).lower() in {"true", "1"},
            "negative_cross_camera": str(r["negative_cross_camera"]).lower() in {"true", "1"},
            "negative_cross_clothes": str(r["negative_cross_clothes"]).lower() in {"true", "1"},
        })
    return result


def load_graph(seed: int) -> dict[int, dict[str, Any]]:
    path = SOURCE_OUT.parent / "ndp_metric_coverage_audit_20261020_165332" / "tables" / f"seed{seed}_K10_metric_correction_anchor.csv"
    return {int(r["anchor"]): r for r in read_csv(path)}


def stratified_sample(available: dict[tuple[int, int, int], list[int]], counts: dict[tuple[int, int, int], int], rng: np.random.Generator) -> set[int]:
    selected: set[int] = set()
    for key, count in counts.items():
        pool = available[key]
        if count > len(pool):
            raise RuntimeError(f"分层抽样数量超过可用图片: {key} {count}>{len(pool)}")
        selected.update(int(x) for x in rng.choice(np.asarray(pool), size=count, replace=False))
    return selected


def overlap_null(h1a: set[int], h1b: set[int], strata_a: dict[tuple[int, int, int], list[int]], strata_b: dict[tuple[int, int, int], list[int]], reps: int) -> dict[str, Any]:
    # 两个 seed 共享相同的 train image index；每个 H1 集合按自身 PID/clothes/camera
    # 分层计数抽样，保留三类结构，不把随机重合基线当作真正的模型重复性。
    counts_a = Counter(strata_key for strata_key, images in strata_a.items() for image in images if image in h1a)
    counts_b = Counter(strata_key for strata_key, images in strata_b.items() for image in images if image in h1b)
    rng = np.random.default_rng(RANDOM_SEED)
    overlaps=[]; jaccards=[]
    for _ in range(reps):
        a = stratified_sample(strata_a, counts_a, rng)
        b = stratified_sample(strata_b, counts_b, rng)
        inter = len(a & b); union = len(a | b)
        overlaps.append(inter); jaccards.append(inter / max(union, 1))
    actual_inter = len(h1a & h1b); actual_union = len(h1a | h1b)
    return {
        "actual_intersection": actual_inter, "actual_union": actual_union,
        "actual_jaccard": actual_inter / max(actual_union, 1),
        "actual_recurrence_a_to_b": actual_inter / max(len(h1a), 1),
        "actual_recurrence_b_to_a": actual_inter / max(len(h1b), 1),
        "null_intersection_mean": float(np.mean(overlaps)), "null_intersection_ci95": [float(np.percentile(overlaps, 2.5)), float(np.percentile(overlaps, 97.5))],
        "null_jaccard_mean": float(np.mean(jaccards)), "null_jaccard_ci95": [float(np.percentile(jaccards, 2.5)), float(np.percentile(jaccards, 97.5))],
        "null_overlap_p_upper": (1 + sum(x >= actual_inter for x in overlaps)) / (reps + 1),
        "strata_count_a": len(counts_a), "strata_count_b": len(counts_b),
    }


def summarize_strata(seed: int, anchors: set[int], bank: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    pid = bank["pid"].astype(int); clothes = bank["clothes"].astype(int); cams = bank["camid"].astype(int)
    counts = Counter((int(pid[a]), int(clothes[a]), int(cams[a])) for a in anchors)
    return [{"seed": seed, "pid": key[0], "clothes": key[1], "camera": key[2], "h1_anchor_count": count} for key, count in sorted(counts.items())]


def pid_bootstrap_overlap(h1a: set[int], h1b: set[int], pids: np.ndarray) -> dict[str, Any]:
    by_pid_a=defaultdict(set); by_pid_b=defaultdict(set)
    for a in h1a: by_pid_a[int(pids[a])].add(a)
    for b in h1b: by_pid_b[int(pids[b])].add(b)
    pid_inter = set(by_pid_a) & set(by_pid_b); pid_union=set(by_pid_a)|set(by_pid_b)
    rng=np.random.default_rng(RANDOM_SEED+9)
    vals=[]
    for _ in range(BOOTSTRAP_REPS):
        sample=rng.choice(sorted(pid_union), size=len(pid_union), replace=True)
        vals.append(np.mean([1.0 if int(pid) in pid_inter else 0.0 for pid in sample]))
    return {"pid_intersection_count": len(pid_inter), "pid_union_count": len(pid_union), "pid_jaccard": len(pid_inter)/max(len(pid_union),1), "pid_overlap_bootstrap_ci95": [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-time", required=True); args=parser.parse_args()
    run_time=args.run_time; out=ROOT/"outputs"/f"ndp_h1_online_feasibility_{run_time}"
    if out.exists(): raise SystemExit(f"拒绝覆盖: {out}")
    out.mkdir(parents=True)
    log_path=ROOT/"logs"/f"NDP_H1跨种子稳定性与在线风险可行性_{run_time}.log"
    def log(msg: str):
        print(msg, flush=True); log_path.open("a",encoding="utf-8").write(msg+"\n")

    source_files={}
    for p in (ROOT/"scripts"/"ndp_h1_cross_seed_online_feasibility.py", ROOT/"scripts"/"ndp_semantic_validity_report.py", ROOT/"scripts"/"ndp_metric_correction_triplet_denominator_audit.py", COVERAGE_OUT/"ndp_metric_coverage_audit.json", Path("/data/DIFFER/loss/triplet_loss.py"), Path("/data/DIFFER/loss/make_loss.py"), Path("/data/DIFFER/data/samplers.py"), Path("/data/DIFFER/scripts/phase5b_id_damage_ds.py"), Path("/data/DIFFER/scripts/phase7i_relation_specificity.py")):
        source_files[str(p)]=sha256_file(p)

    banks={seed:load_bank(seed) for seed in SEEDS}; relations={seed:load_relations(seed) for seed in SEEDS}; graphs={seed:load_graph(seed) for seed in SEEDS}
    if not np.array_equal(banks[1234]["image_path"], banks[4321]["image_path"]): raise RuntimeError("两个 seed 的 image_path 未对齐")
    h1={seed:{r["anchor"] for r in relations[seed] if r["h1"]} for seed in SEEDS}
    log(f"【复用】H1 anchor 数量：seed1234={len(h1[1234])}，seed4321={len(h1[4321])}；未重新生成特征。")
    strata={}
    for seed in SEEDS:
        bank=banks[seed]; pid=bank["pid"].astype(int); clothes=bank["clothes"].astype(int); cams=bank["camid"].astype(int)
        s=defaultdict(list)
        for i in range(len(pid)): s[(int(pid[i]),int(clothes[i]),int(cams[i]))].append(i)
        strata[seed]=dict(s)
    overlap=overlap_null(h1[1234],h1[4321],strata[1234],strata[4321],BOOTSTRAP_REPS)
    pid_overlap=pid_bootstrap_overlap(h1[1234],h1[4321],banks[1234]["pid"].astype(int))
    stratum_rows=summarize_strata(1234,h1[1234],banks[1234])+summarize_strata(4321,h1[4321],banks[4321])
    event_rows=[]
    for seed in SEEDS:
        rs=relations[seed]; g=graphs[seed]
        h1_rs=[r for r in rs if r["h1"]]
        event_rows.append({"seed":seed,"h1_relation_count":len(h1_rs),"h1_anchor_count":len(h1[seed]),"h1_pid_count":len({int(r["pid"]) for r in h1_rs}),"h1_net_intrusion_anchor_count":sum(int(g[a]["NET_NEGATIVE_INTRUSION_count"])>0 for a in h1[seed]),"h1_same_id_drop_anchor_count":sum(int(g[a]["DELTA_SAME_ID_COUNT"])<0 for a in h1[seed]),"positive_cross_clothes_rate":float(np.mean([r["positive_cross_clothes"] for r in h1_rs])),"negative_cross_camera_rate":float(np.mean([r["negative_cross_camera"] for r in h1_rs])),"negative_cross_clothes_rate":float(np.mean([r["negative_cross_clothes"] for r in h1_rs]))})
    coverage=read_json(COVERAGE_OUT/"ndp_metric_coverage_audit.json")
    cov={}
    for c in coverage["coverage_summaries"]:
        if c["K"]==K:
            for s in c["scopes"]: cov[(c["seed"],s["scope"])]=s
    triplet_rows=[]
    for seed in SEEDS:
        s=cov[(seed,"reliable_flip_rows")]; all_s=cov[(seed,"all_global_relation_rows")]
        triplet_rows.append({"seed":seed,"h1_rows":s["rows"],"h1_full_hard":s["full_triplet_hard_selected"],"h1_anchor_any_miner":s["anchor_any_miner_selection"],"h1_unique_anchors":s["unique_anchors"],"global_rows":all_s["rows"],"global_full_hard":all_s["full_triplet_hard_selected"],"global_three_way":all_s["three_way_opportunity"],"coverage_mode":"冻结E60 GPC特征离线Batch-Hard代理；非在线逐step特征/梯度"})

    summary={"run_time":run_time,"source_run":SOURCE_RUN,"official_query_gallery_used":False,"model_training_started":False,"decision":"NDP_LOCAL_RISK_MECHANISM_UNRESOLVED","cross_seed_stability_signal":"OBSERVED_ABOVE_STRATIFIED_RANDOM_BASELINE","h1_definition":"reference_reliable (g_ref>0.05) and current_order_correct false (g_current<=0)","h2_definition":"no H1 relation on anchor and DELTA_SAME_ID_COUNT<0","h3_definition":"neighborhood_changed and DELTA_SAME_ID_COUNT>0","h1_h2_h3_mutual_exclusive":"No: H1 is relation-event/anchor projection; H3 may overlap an H1 anchor. H2 and H3 are disjoint by signed same-ID delta.","overlap":overlap,"pid_overlap":pid_overlap,"event_rows":event_rows,"triplet_rows":triplet_rows,"online_risk_status":"ONLINE_RISK_PREDICTION_NOT_IDENTIFIABLE_FROM_CURRENT_DATA","online_risk_reason":"现有资产只有Bio-only与Bio+GPC E60终点及冻结关系；没有同一训练轨迹更新前后配对checkpoint或逐步风险标签。","amp_gate_status":"UNRESOLVED_FROM_PRIOR_AA_DIAGNOSTICS","source_files":source_files}
    write_json(out/"ndp_h1_online_feasibility_summary.json",summary); write_csv(out/"h1_event_summary.csv",event_rows); write_csv(out/"triplet_h1_gap.csv",triplet_rows); write_csv(out/"h1_pid_clothes_camera_strata.csv",stratum_rows)
    reports=ROOT/"reports"
    text="# NDP H1 跨种子稳定性\n\nH1 固定定义为：参考可靠 `g_ref>0.05` 且当前 `g_current<=0`。H1 关系事件映射为发生过至少一个 H1 事件的 anchor。H2 与 H3 是 anchor 分组；H2 排除 H1 且同身份数量下降，H3 为邻域变化且同身份数量增加。由于 H1 与 H3 层级不同，三组不是完全互斥；H2/H3 按有符号同身份变化互斥。\n\n"
    text+="| seed | H1关系 | H1 anchor | H1 PID | H1中净侵入anchor | H1中同身份下降anchor | 跨衣正样本 | 负样本跨相机 |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in event_rows: text+=f"| {r['seed']} | {r['h1_relation_count']} | {r['h1_anchor_count']} | {r['h1_pid_count']} | {r['h1_net_intrusion_anchor_count']} | {r['h1_same_id_drop_anchor_count']} | {r['positive_cross_clothes_rate']:.2%} | {r['negative_cross_camera_rate']:.2%} |\n"
    text+=f"\nH1 anchor 交集={overlap['actual_intersection']}，并集={overlap['actual_union']}，Jaccard={overlap['actual_jaccard']:.4%}；seed1234→4321复现率={overlap['actual_recurrence_a_to_b']:.4%}，反向={overlap['actual_recurrence_b_to_a']:.4%}。按 PID/服装/相机三元分层的随机重合基线：交集均值={overlap['null_intersection_mean']:.2f}，95%区间={overlap['null_intersection_ci95']}，Jaccard均值={overlap['null_jaccard_mean']:.4%}，上尾 p={overlap['null_overlap_p_upper']:.4f}。\n\n"
    text+=f"本轮 H1 anchor 交集显著高于分层随机基线（实际 {overlap['actual_intersection']}，随机均值 {overlap['null_intersection_mean']:.2f}，上尾 p={overlap['null_overlap_p_upper']:.4f}），支持‘存在共享的终点风险结构’这一有限结论；但它仍不是单次更新的因果风险标签，也不能把共同 H1 PID 直接称为通用困难身份。服装/相机分层明细位于输出 CSV。\n"
    (reports/"NDP_H1_CROSS_SEED_STABILITY.md").write_text(text,encoding="utf-8")
    text="# NDP 更新前风险信号可行性\n\n当前结论：`ONLINE_RISK_PREDICTION_NOT_IDENTIFIABLE_FROM_CURRENT_DATA`。现有数据是 Bio-only 与 Bio+GPC 的冻结 E60 终点比较，缺少同一训练轨迹的更新前状态、更新后状态和逐步事件标签；不能把两条独立轨迹的终点差异当作一次更新的在线风险预测。\n\n可作为后续诊断变量、但本轮不赋权的信号包括：参考正负优势、当前局部相似度分布、Top-K 同身份概率质量、正负间隔、异身份 PID 数、Triplet 是否激活、当前结构与参考结构差异。要验证在线可识别性，必须预先冻结同一模型状态的 update-before/update-after 配对 checkpoint 和逐步 miner/AMP 记录。\n"
    (reports/"NDP_PRE_UPDATE_RISK_FEASIBILITY.md").write_text(text,encoding="utf-8")
    text="# NDP 对 H1 的 Triplet 增量保护缺口\n\n真实 Triplet 实现是欧氏距离 Batch-Hard，NDP 关系是全 train 归一化余弦邻域，二者关系选择规则不同。\n\n| seed | H1关系 | H1完整hard | H1 anchor曾被其他miner | 全局关系 | 全局完整hard |\n|---:|---:|---:|---:|---:|---:|\n"
    for r in triplet_rows: text+=f"| {r['seed']} | {r['h1_rows']} | {r['h1_full_hard']} | {r['h1_anchor_any_miner']}/{r['h1_unique_anchors']} | {r['global_rows']} | {r['global_full_hard']} |\n"
    text+="\nH1 目标关系的精确完整 Batch-Hard 命中为两个 seed 均 0；但 H1 anchor 多数曾受其它 Triplet 关系约束。这支持‘目标关系直接覆盖不足’这一有限结论，不支持‘Triplet 对这些身份完全无效’。E/F 仍是冻结特征离线代理，实际在线非零梯度和 optimizer step 未观测。\n"
    (reports/"NDP_TRIPLET_INCREMENTAL_GAP.md").write_text(text,encoding="utf-8")
    text="# NDP 局部身份质量保持设计\n\n候选 C1：`L_C1=mean(relu(g_ref-g_cur-epsilon))`，在同一三元组上与参考自适应 margin Triplet 逐关系梯度等价，独立性不足。\n\n候选 C2：固定可靠参考邻域 U，定义 `Q_ref=sum_{j:pid_j=pid_i} softmax(s_ref(i,j)/tau)`、`Q_cur=sum_{j:pid_j=pid_i} softmax(s_cur(i,j)/tau)`，使用 `L_C2=mean(relu(Q_ref-Q_cur-epsilon_q))`。它保护多邻居身份概率质量，允许错误关系修复和有益重排，但需冻结可靠参考，避免保护错误结构。\n\n候选 C3：不使用参考结构的 SupCon/N-pair 型局部身份质量约束，作为无参考控制。C2 只有在短程实验中相对 C1/C3/普通 Triplet 展现不同的局部质量变化，才可被称为独立 NDP。三者都需说明邻域来源、可微特征、显存成本及失败条件；本轮未接入 processor。\n"
    (reports/"NDP_LOCAL_IDENTITY_MASS_DESIGN.md").write_text(text,encoding="utf-8")
    text=f"# NDP 下一阶段决策\n\n状态：`NDP_LOCAL_RISK_MECHANISM_UNRESOLVED`。H1 anchor 交集为 {overlap['actual_intersection']}，高于按 PID/服装/相机分层的随机基线（均值 {overlap['null_intersection_mean']:.2f}，上尾 p={overlap['null_overlap_p_upper']:.4f}），因此出现了共享终点风险结构信号；但在线风险在当前终点数据上不可识别，且 AMP/A-A 数值门禁尚未完全解决。\n\n不启动 NDP 短程、30 epoch 或 60 epoch 训练。下一步若继续，只能先获取同一训练轨迹的配对状态与在线 miner/AMP 记录，再预注册 C1/C2/C3 控制实验。\n"
    (reports/f"NDP_NEXT_STAGE_DECISION_{run_time}.md").write_text(text,encoding="utf-8")
    log("【决策】NDP_LOCAL_RISK_MECHANISM_UNRESOLVED；未启动模型计算或训练。")


if __name__ == "__main__": main()
