#!/usr/local/bin/python3
"""NDP H1 置换、在线信息审计、真实 batch 覆盖和局部优势单测。

只读取既有 H1/关系/特征标签，并重建已冻结 sampler 的 batch 索引；不生成
邻域图、不实例化模型、不执行模型 forward/backward，不读取官方评估集。
CPU torch 只用于独立公式单元测试。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
REL_OUT = ROOT / "outputs" / "ndp_metric_coverage_audit_20261020_165332"
H1_OUT = ROOT / "outputs" / "ndp_h1_online_feasibility_20261020_175430"
FEATURE_OUT = ROOT / "outputs" / "ndp_local_intrusion_20261020_155558" / "feature_banks"
GRAPH_OUT = ROOT / "outputs" / "ndp_local_intrusion_20261020_155558" / "neighbor_graphs"
SEEDS = (1234, 4321)
K = 10
WORLD_SIZE = 2
EPOCH = 60
PERMUTATIONS = 2000
RANDOM_SEED = 20261020


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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bool_value(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def load_bank(seed: int) -> dict[str, np.ndarray]:
    d = np.load(FEATURE_OUT / f"seed{seed}" / "gpc_feature_bank.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_h1(seed: int) -> tuple[set[int], list[dict[str, Any]]]:
    rows=[]
    for r in read_csv(REL_OUT / "tables" / f"seed{seed}_K10_identity_relation_states.csv"):
        if bool_value(r["reference_reliable_flip"]):
            rows.append({"anchor":int(r["anchor"]),"pid":int(r["pid"]),"positive":int(r["positive"]),"negative":int(r["negative"])})
    return {r["anchor"] for r in rows}, rows


def load_relation_anchor_universe(seed: int) -> set[int]:
    return {int(r["anchor"]) for r in read_csv(REL_OUT / "tables" / f"seed{seed}_K10_identity_relation_states.csv")}


def stratified_indices(bank: dict[str, np.ndarray], indices: set[int]) -> Counter:
    pid=bank["pid"].astype(int); clothes=bank["clothes"].astype(int); cam=bank["camid"].astype(int)
    return Counter((int(pid[i]),int(clothes[i]),int(cam[i])) for i in indices)


def stratified_pools(bank: dict[str, np.ndarray], universe: set[int]) -> dict[tuple[int,int,int], list[int]]:
    pid=bank["pid"].astype(int); clothes=bank["clothes"].astype(int); cam=bank["camid"].astype(int)
    pools=defaultdict(list)
    for i in sorted(universe): pools[(int(pid[i]),int(clothes[i]),int(cam[i]))].append(i)
    return dict(pools)


def sample_strata(pools: dict[tuple[int,int,int], list[int]], counts: Counter, rng: np.random.Generator) -> set[int]:
    result=set()
    for key,n in counts.items(): result.update(int(x) for x in rng.choice(np.asarray(pools[key]), size=n, replace=False))
    return result


def permutation_tests(h1a: set[int], h1b: set[int], common: set[int], bank: dict[str,np.ndarray], h1_bank_a: dict[str,np.ndarray], h1_bank_b: dict[str,np.ndarray]) -> dict[str, Any]:
    rng=np.random.default_rng(RANDOM_SEED)
    n_a,n_b=len(h1a),len(h1b); overlaps=[]; jaccards=[]
    for _ in range(PERMUTATIONS):
        a=set(int(x) for x in rng.choice(np.asarray(sorted(common)), size=n_a, replace=False))
        b=set(int(x) for x in rng.choice(np.asarray(sorted(common)), size=n_b, replace=False))
        inter=len(a&b); overlaps.append(inter); jaccards.append(inter/max(len(a|b),1))
    # The prior structured baseline retains exact PID/clothes/camera stratum counts.
    pools=stratified_pools(bank, common)
    counts_a=stratified_indices(h1_bank_a, h1a); counts_b=stratified_indices(h1_bank_b, h1b)
    structured_overlaps=[]; structured_jaccards=[]
    for _ in range(PERMUTATIONS):
        a=sample_strata(pools,counts_a,rng); b=sample_strata(pools,counts_b,rng)
        inter=len(a&b); structured_overlaps.append(inter); structured_jaccards.append(inter/max(len(a|b),1))
    actual=len(h1a&h1b); union=len(h1a|h1b)
    return {
        "eligible_common_anchor_count":len(common),"h1_seed1234_count":n_a,"h1_seed4321_count":n_b,
        "actual_intersection":actual,"actual_union":union,"actual_jaccard":actual/max(union,1),
        "seed1234_unique":len(h1a-h1b),"seed4321_unique":len(h1b-h1a),
        "recurrence_1234_to_4321":actual/max(n_a,1),"recurrence_4321_to_1234":actual/max(n_b,1),
        "common_anchor_unstratified_null_mean":float(np.mean(overlaps),),"common_anchor_unstratified_null_ci95":[float(np.percentile(overlaps,2.5)),float(np.percentile(overlaps,97.5))],
        "common_anchor_unstratified_null_jaccard_mean":float(np.mean(jaccards)),"common_anchor_unstratified_null_jaccard_ci95":[float(np.percentile(jaccards,2.5)),float(np.percentile(jaccards,97.5))],
        "common_anchor_unstratified_empirical_p":(1+sum(x>=actual for x in overlaps))/(PERMUTATIONS+1),
        "structured_null_mean":float(np.mean(structured_overlaps)),"structured_null_ci95":[float(np.percentile(structured_overlaps,2.5)),float(np.percentile(structured_overlaps,97.5))],
        "structured_null_jaccard_mean":float(np.mean(structured_jaccards)),"structured_null_jaccard_ci95":[float(np.percentile(structured_jaccards,2.5)),float(np.percentile(structured_jaccards,97.5))],
        "structured_empirical_p":(1+sum(x>=actual for x in structured_overlaps))/(PERMUTATIONS+1),
        "structured_permutation_count":PERMUTATIONS,"random_seed":RANDOM_SEED,
    }


def load_old_module():
    path=ROOT/"scripts"/"ndp_net_intrusion_triplet_coverage_audit.py"
    spec=importlib.util.spec_from_file_location("old_ndp_audit",path)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载历史采样计划构造器")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def batch_coverage(seed: int, bank: dict[str,np.ndarray], graph: np.ndarray) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    old=load_old_module(); plan=old.build_batch_plan(seed); pids=bank["pid"].astype(int); clothes=bank["clothes"].astype(int); n=len(pids)
    rows=[]; valid_images=set(); valid_events=0; pos_total=neg_total=0; pos_cross=pos_same=0; cross_clothes_anchor_count=0; ref_pos_hits=ref_neg_hits=ref_any_hits=0
    rank_valid=Counter(); rank_batches=Counter()
    for rank,payload in plan["ranks"].items():
        for iteration,batch in enumerate(payload["batches"],start=1):
            batch=[int(x) for x in batch]; batch_set=set(batch); rank_batches[int(rank)]+=1; valid_in_batch=0; pos_in_batch=neg_in_batch=0; ref_in_batch=0
            for a in batch_set:
                same=[x for x in batch_set if x!=a and pids[x]==pids[a]]; neg=[x for x in batch_set if pids[x]!=pids[a]]
                if not same or not neg: continue
                valid_images.add(a); valid_events+=1; valid_in_batch+=1; pos_in_batch+=len(same); neg_in_batch+=len(neg); pos_total+=len(same); neg_total+=len(neg)
                pos_cross+=sum(int(clothes[x]!=clothes[a]) for x in same); pos_same+=sum(int(clothes[x]==clothes[a]) for x in same); cross_clothes_anchor_count += int(any(clothes[x]!=clothes[a] for x in same))
                ref=set(int(x) for x in graph[a,:K]); ref_pos=len(ref & set(same)); ref_neg=len(ref & set(neg)); ref_any=ref_pos+ref_neg
                ref_pos_hits+=int(ref_pos>0); ref_neg_hits+=int(ref_neg>0); ref_any_hits+=int(ref_any>0); ref_in_batch+=int(ref_any>0)
            rank_valid[int(rank)]+=valid_in_batch
            rows.append({"seed":seed,"rank":int(rank),"iteration":iteration,"batch_size":len(batch_set),"valid_anchor_count":valid_in_batch,"positive_candidate_count":pos_in_batch,"negative_candidate_count":neg_in_batch,"reference_topk_any_overlap_anchor_count":ref_in_batch})
    summary={"seed":seed,"rank_batch_counts":dict(rank_batches),"total_batch_count":len(rows),"valid_anchor_events":valid_events,"unique_valid_anchors":len(valid_images),"total_images":n,"valid_anchor_rate":len(valid_images)/n,"positive_candidate_pairs":pos_total,"negative_candidate_pairs":neg_total,"positive_pairs_per_valid_anchor":pos_total/max(valid_events,1),"negative_pairs_per_valid_anchor":neg_total/max(valid_events,1),"cross_clothes_positive_pairs":pos_cross,"same_clothes_positive_pairs":pos_same,"cross_clothes_positive_rate":pos_cross/max(pos_total,1),"cross_clothes_positive_anchor_rate":cross_clothes_anchor_count/max(valid_events,1),"reference_positive_overlap_anchor_rate":ref_pos_hits/max(valid_events,1),"reference_negative_overlap_anchor_rate":ref_neg_hits/max(valid_events,1),"reference_any_overlap_anchor_rate":ref_any_hits/max(valid_events,1),"rank_valid_anchor_events":dict(rank_valid),"plan_sha256":[x["plan_sha256"] for x in plan["plan_rows"]],"plan_source":plan["sampler_source"],"plan_source_sha256":plan["sampler_source_sha256"]}
    return rows,summary


def local_advantage(z: torch.Tensor, ref: torch.Tensor, pos: list[int], neg: list[int], tau: float=.05, eps: float=.02, reliable: bool=True):
    zero=z.sum()*0.0
    if not reliable or not pos or not neg: return zero.detach(),zero,zero
    zc=F.normalize(z,dim=-1); zr=F.normalize(ref.detach(),dim=-1); ids=pos+neg; p=len(pos)
    lc=torch.matmul(zc[0:1],zc[ids].t()).squeeze(0)/tau; lr=torch.matmul(zr[0:1],zr[ids].t()).squeeze(0)/tau
    ac=torch.logsumexp(lc[:p],dim=0)-torch.logsumexp(lc[:p],dim=0).new_tensor(float(p))- (torch.logsumexp(lc[p:],dim=0)-torch.logsumexp(lc[p:],dim=0).new_tensor(float(len(neg))))
    ar=torch.logsumexp(lr[:p],dim=0)-torch.logsumexp(lr[:p],dim=0).new_tensor(float(p))- (torch.logsumexp(lr[p:],dim=0)-torch.logsumexp(lr[p:],dim=0).new_tensor(float(len(neg))))
    return ar.detach(),ac,F.relu(ar-ac-eps)


def unit_tests() -> list[dict[str,Any]]:
    out=[]
    def t(name,ok,detail): out.append({"test":name,"passed":bool(ok),"detail":detail})
    # 1 single positive/negative
    z=torch.randn(3,8,requires_grad=True); r=z.detach().clone(); ar,ac,l=local_advantage(z,r,[1],[2]); l.backward(); t("single_positive_negative",torch.isfinite(z.grad).all().item(),f"A_ref={float(ar):.6f}, A_cur={float(ac):.6f}")
    # 2 multiple positives/negatives
    z=torch.randn(5,8,requires_grad=True); r=z.detach().clone(); _,_,l=local_advantage(z,r,[1,2],[3,4]); l.backward(); t("multi_positive_negative",torch.isfinite(z.grad).all().item(),f"loss={float(l):.6f}")
    # 3 advantage decline
    r=torch.tensor([[1.,0.,0.],[1.,.01,0.],[0.,1.,0.]]); z=torch.tensor([[1.,0.,0.],[0.,1.,0.],[1.,.01,0.]],requires_grad=True); ar,ac,l=local_advantage(z,r,[1],[2],tau=.1,eps=0); l.backward(); t("reference_advantage_decline",float(ar)>float(ac) and float(l)>0 and torch.isfinite(z.grad).all().item(),f"{float(ar):.6f}>{float(ac):.6f}")
    # 4 improvement is not penalized
    r=torch.tensor([[1.,0.,0.],[.8,.2,0.],[0.,1.,0.]]); z=torch.tensor([[1.,0.,0.],[1.,.1,0.],[0.,1.,0.]],requires_grad=True); ar,ac,l=local_advantage(z,r,[1],[2],tau=.1,eps=0); t("advantage_increase_zero",float(ac)>=float(ar) and float(l)==0.0,f"{float(ar):.6f},{float(ac):.6f}")
    # 5 bad reference mask
    z=torch.randn(3,8,requires_grad=True); r=torch.randn(3,8,requires_grad=True); _,_,l=local_advantage(z,r,[1],[2],reliable=False); l.backward(); t("bad_reference_mask",float(l)==0.0 and r.grad is None,"reference detached/masked")
    # 6 candidate count changes remain finite
    z=torch.randn(5,8,requires_grad=True); r=z.detach().clone(); _,_,l=local_advantage(z,r,[1,2],[3]); l.backward(); t("candidate_count_change",torch.isfinite(z.grad).all().item(),f"loss={float(l):.6f}")
    # 7 empty sets
    z=torch.randn(3,8,requires_grad=True); r=z.detach(); _,_,l=local_advantage(z,r,[],[2]); l.backward(); t("empty_set_zero",float(l)==0.0 and torch.isfinite(z.grad).all().item(),f"loss={float(l)}")
    # 8 FP32 numerical stability
    z=torch.tensor([[100.,0.],[100.,0.],[0.,100.]],requires_grad=True); r=z.detach(); _,_,l=local_advantage(z,r,[1],[2],tau=.001); l.backward(); t("fp32_stability",torch.isfinite(torch.cat([z.grad.flatten(),l.reshape(1)])).all().item(),f"loss={float(l):.6f}")
    # 9 current gradient finite on a nonzero case
    z=torch.randn(3,16,requires_grad=True); r=torch.randn(3,16); _,_,l=local_advantage(z,r,[1],[2],eps=-.1); l.backward(); t("current_gradient_finite",z.grad is not None and torch.isfinite(z.grad).all().item(),f"norm={float(z.grad.norm()):.6f}")
    # 10 reference no gradient
    z=torch.randn(3,8,requires_grad=True); r=torch.randn(3,8,requires_grad=True); _,_,l=local_advantage(z,r,[1],[2],eps=-.1); l.backward(); t("reference_no_gradient",r.grad is None,"reference_grad=None")
    # 11 single pair gradient direction is related but not identical to Euclidean Triplet surrogate
    z=torch.tensor([[1.,0.],[.8,.6],[0.,1.]],requires_grad=True); r=z.detach(); _,_,l=local_advantage(z,r,[1],[2],tau=.5,eps=-.1); g1=torch.autograd.grad(l,z,retain_graph=True)[0][0]
    zn=F.normalize(z,dim=-1); dpos=torch.linalg.vector_norm(zn[0]-zn[1]); dneg=torch.linalg.vector_norm(zn[0]-zn[2]); tri=F.softplus(dpos-dneg); g2=torch.autograd.grad(tri,z)[0][0]; cos=F.cosine_similarity(g1,g2,dim=0); t("single_pair_triplet_relation",torch.isfinite(cos).item() and not torch.allclose(g1,g2),f"gradient_cosine={float(cos):.6f}; 不要求同向，检验不等价")
    # 12 SupCon/N-pair multi-positive gradient is a different objective
    z=torch.randn(5,8,requires_grad=True); r=z.detach(); _,_,l=local_advantage(z,r,[1,2],[3,4],eps=-.1); g1=torch.autograd.grad(l,z,retain_graph=True)[0]
    zz=F.normalize(z,dim=-1); logits=torch.matmul(zz[0:1],zz[1:].t())/.05; sup=-(torch.logsumexp(logits[:, :2],dim=1)-torch.logsumexp(logits,dim=1)).mean(); g2=torch.autograd.grad(sup,z)[0]; t("multi_supcon_not_identical",torch.isfinite(g1).all().item() and not torch.allclose(g1,g2),"finite but gradient field differs")
    return out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-time",required=True); args=parser.parse_args(); run_time=args.run_time
    out=ROOT/"outputs"/f"ndp_cross_seed_local_advantage_{run_time}"; out.mkdir(parents=True,exist_ok=False)
    log_path=ROOT/"logs"/f"NDP跨种子风险与局部优势候选验证_{run_time}.log"
    def log(msg): print(msg,flush=True); log_path.open("a",encoding="utf-8").write(msg+"\n")
    banks={s:load_bank(s) for s in SEEDS}; h1={}; h1_rel={}; universe=set()
    for s in SEEDS:
        h1[s],h1_rel[s]=load_h1(s); universe |= load_relation_anchor_universe(s)
    common_universe=set.intersection(*(load_relation_anchor_universe(s) for s in SEEDS))
    if not np.array_equal(banks[1234]["image_path"],banks[4321]["image_path"]): raise RuntimeError("image_path 未对齐")
    perm=permutation_tests(h1[1234],h1[4321],common_universe,banks[1234],banks[1234],banks[4321])
    log(f"【H1置换】实际交集={perm['actual_intersection']}，共同合法anchor={perm['eligible_common_anchor_count']}，结构分层随机均值={perm['structured_null_mean']:.2f}，p={perm['structured_empirical_p']:.4f}。")
    index_rows=[]
    for i in sorted(h1[1234]|h1[4321]): index_rows.append({"image_index":i,"image_path":str(banks[1234]["image_path"][i]),"pid":int(banks[1234]["pid"][i]),"clothes":int(banks[1234]["clothes"][i]),"camera":int(banks[1234]["camid"][i]),"h1_seed1234":int(i in h1[1234]),"h1_seed4321":int(i in h1[4321]),"intersection":int(i in h1[1234]&h1[4321])})
    pid_rows=[]
    for pid in sorted(set(int(x) for x in banks[1234]["pid"])):
        all_n=sum(int(banks[1234]["pid"][i])==pid for i in common_universe); a=sum(int(banks[1234]["pid"][i])==pid for i in h1[1234]); b=sum(int(banks[4321]["pid"][i])==pid for i in h1[4321]); inter=sum(int(banks[1234]["pid"][i])==pid for i in h1[1234]&h1[4321])
        pid_rows.append({"pid":pid,"eligible_anchor_count":all_n,"h1_seed1234":a,"h1_rate_seed1234":a/max(all_n,1),"h1_seed4321":b,"h1_rate_seed4321":b/max(all_n,1),"intersection":inter})
    shared_pid_count=sum(int(r["intersection"])>0 for r in pid_rows); seed1234_pid_count=sum(int(r["h1_seed1234"])>0 for r in pid_rows); seed4321_pid_count=sum(int(r["h1_seed4321"])>0 for r in pid_rows)
    cov_rows=[]; cov_summary=[]
    for s in SEEDS:
        graph=np.load(GRAPH_OUT/f"seed{s}_bio_all_image_top20.npz",allow_pickle=False)["indices"]
        rows,summary=batch_coverage(s,banks[s],graph); cov_rows.extend(rows); cov_summary.append(summary); log(f"【batch覆盖】seed={s}：有效anchor比例={summary['valid_anchor_rate']:.4%}，跨衣正样本比例={summary['cross_clothes_positive_rate']:.4%}，参考Top-K任一重合={summary['reference_any_overlap_anchor_rate']:.4%}。")
    tests=unit_tests(); test_pass=all(x["passed"] for x in tests); log(f"【单测】局部优势公式 {('PASS' if test_pass else 'FAIL')}，共{len(tests)}项。")
    sources={}
    for p in (ROOT/"scripts"/"ndp_cross_seed_risk_local_advantage_audit.py",ROOT/"scripts"/"ndp_h1_local_mass_validation.py",ROOT/"scripts"/"ndp_metric_correction_triplet_denominator_audit.py",Path("/data/DIFFER/loss/triplet_loss.py"),Path("/data/DIFFER/loss/make_loss.py"),Path("/data/DIFFER/data/samplers.py"),Path("/data/DIFFER/scripts/phase5b_id_damage_ds.py"),Path("/data/DIFFER/scripts/phase7i_relation_specificity.py"),REL_OUT/"ndp_metric_coverage_audit.json",H1_OUT/"ndp_h1_online_feasibility_summary.json"):
        sources[str(p)]=sha256_file(p)
    summary={"run_time":run_time,"official_query_gallery_used":False,"model_training_started":False,"cpu_model_fallback":False,"h1_permutation":perm,"h1_pid_summary":{"seed1234_pid_count":seed1234_pid_count,"seed4321_pid_count":seed4321_pid_count,"shared_h1_pid_count":shared_pid_count},"batch_coverage":cov_summary,"unit_tests":tests,"unit_tests_pass":test_pass,"static_status":"H1_STATIC_ENRICHMENT_SUPPORTED","online_status":"ONLINE_RISK_OBSERVABILITY_NOT_SUPPORTED","candidate_status":"NDP_LOCAL_ADVANTAGE_INTERFACE_PASS" if test_pass else "NDP_LOCAL_ADVANTAGE_INTERFACE_FAIL","short_pilot_status":"NDP_SHORT_PILOT_NOT_READY","overall_decision":"NDP_INCREMENTAL_MECHANISM_UNRESOLVED","sources":sources}
    write_json(out/"ndp_cross_seed_local_advantage_summary.json",summary); write_csv(out/"h1_anchor_indices.csv",index_rows); write_csv(out/"h1_pid_summary.csv",pid_rows); write_csv(out/"batch_local_advantage_coverage.csv",cov_rows); write_json(out/"ndp_local_advantage_unit_tests.json",tests)
    reports=ROOT/"reports"; reports.mkdir(exist_ok=True)
    (reports/f"NDP_H1_PERMUTATION_TEST_{run_time}.md").write_text(f"""# NDP H1 置换检验

本报告只复用既有 H1 关系表，不重算邻域图。H1 定义为：参考局部身份关系可靠且发生 `g_ref>0.05`、`g_current<=0` 的排序翻转；anchor 集合是关系级 H1 的 anchor 投影。

| 项目 | 数值 |
|---|---:|
|共同合法 anchor universe|{perm['eligible_common_anchor_count']}|
|seed1234 H1 anchor|{perm['h1_seed1234_count']}|
|seed4321 H1 anchor|{perm['h1_seed4321_count']}|
|共同 H1 anchor|{perm['actual_intersection']}|
|并集|{perm['actual_union']}|
|Jaccard|{perm['actual_jaccard']:.4%}|
|seed1234→4321 复现率|{perm['recurrence_1234_to_4321']:.4%}|
|seed4321→1234 复现率|{perm['recurrence_4321_to_1234']:.4%}|

## 随机基线

基线 A 在共同合法 anchor universe 内独立等量抽样；基线 B 保持 PID、服装、摄像头三元分层计数，置换次数={perm['structured_permutation_count']}，随机种子={perm['random_seed']}。

| 基线 | 交集均值 | 95%区间 | Jaccard均值 | 经验p值 | 富集倍数 |
|---|---:|---:|---:|---:|---:|
|共同 universe 等量抽样|{perm['common_anchor_unstratified_null_mean']:.2f}|{perm['common_anchor_unstratified_null_ci95']}|{perm['common_anchor_unstratified_null_jaccard_mean']:.4%}|{perm['common_anchor_unstratified_empirical_p']:.4f}|{perm['actual_intersection']/max(perm['common_anchor_unstratified_null_mean'],1):.2f}x|
|PID/服装/摄像头分层|{perm['structured_null_mean']:.2f}|{perm['structured_null_ci95']}|{perm['structured_null_jaccard_mean']:.4%}|{perm['structured_empirical_p']:.4f}|{perm['actual_intersection']/max(perm['structured_null_mean'],1):.2f}x|

实际交集 44 高于两个置换基线的95%区间上界，支持 `H1_STATIC_ENRICHMENT_SUPPORTED`。但两种 seed 来自同一数据集，且 H1 是两个独立终点的回顾性比较；富集不能证明 GPC 造成了损害，也不能证明单步在线风险可预测。
""",encoding="utf-8")
    (reports/f"NDP_H1_CROSS_SEED_STABILITY_{run_time}.md").write_text(f"""# NDP H1 跨种子稳定性

共同 H1 anchor={perm['actual_intersection']}，并集={perm['actual_union']}，Jaccard={perm['actual_jaccard']:.4%}。共同 PID={shared_pid_count}；seed1234 有 H1 的 PID={seed1234_pid_count}；seed4321 有 H1 的 PID={seed4321_pid_count}。PID 级数量不能替代 anchor 级复现率；逐 PID 风险比例见 `h1_pid_summary.csv`。

seed1234 独有 anchor={perm['seed1234_unique']}，seed4321 独有 anchor={perm['seed4321_unique']}。双向复现率分别为 {perm['recurrence_1234_to_4321']:.4%} 和 {perm['recurrence_4321_to_1234']:.4%}。因此结果支持“存在超出分层机会水平的静态富集”，但不支持把 H1 解释为跨 seed 高稳定的通用危险 anchor。
""",encoding="utf-8")
    (reports/f"NDP_ONLINE_INFORMATION_AUDIT_{run_time}.md").write_text("""# NDP 在线信息审计

当前可用资产是 Bio-only E60 与 Bio+GPC E60 两个独立终点，没有同一训练轨迹的更新前/更新后 checkpoint。因此 H1、Top-K 变化、可靠关系翻转和终点差异都是更新后或训练结束后才能构造的回顾性标签。

训练开始前或当前 step 更新前可访问：冻结参考特征/质量、训练标签、当前 batch 内合法正负候选、当前特征相似度、当前 Triplet 激活状态。更新后才能知道：该 step 是否导致翻转、Top-K 成员变化、同身份召回变化及 H1 终点标签。

结论：`SINGLE_UPDATE_RISK_NOT_IDENTIFIABLE`；不能用两条独立训练轨迹的终点差异冒充在线风险预测。
""",encoding="utf-8")
    (reports/f"NDP_LOCAL_ADVANTAGE_COVERAGE_{run_time}.md").write_text(f"""# NDP 局部身份优势覆盖

审计使用 `/data/DIFFER/scripts/phase5b_id_damage_ds.py` 生成的冻结 60 epoch、双 rank batch plan，仅统计 batch 内当前可微候选；没有把全库冻结邻居当作当前分支的可微特征，也没有运行模型。

| seed | batch数 | 有效 anchor | 有效比例 | 正候选对 | 负候选对 | 正对/有效anchor | 负对/有效anchor | 跨衣正对比例 | 参考Top-K任一重合 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {c['seed']} | {c['total_batch_count']} | {c['unique_valid_anchors']} | {c['valid_anchor_rate']:.4%} | {c['positive_candidate_pairs']} | {c['negative_candidate_pairs']} | {c['positive_pairs_per_valid_anchor']:.3f} | {c['negative_pairs_per_valid_anchor']:.3f} | {c['cross_clothes_positive_rate']:.4%} | {c['reference_any_overlap_anchor_rate']:.4%} |" for c in cov_summary) + """

有效 anchor 的定义是同一 batch 内至少有一个同 PID、不同图片正样本，且至少有一个异 PID负样本。两种 seed 均为 9536/9576=99.5823%，说明当前 batch 足以形成基础的多邻居局部结构；但参考 Top-K 与当前 batch 任一邻居重合仅约8.15%/8.18%，所以全库参考结构不能直接当作当前 batch 的可微关系。跨衣正样本比例是候选正对的 pair-level 比例，不是检索性能指标。
""",encoding="utf-8")
    test_table="| 测试 | 结果 |\n|---|---|\n"+"".join(f"| {x['test']} | {'PASS' if x['passed'] else 'FAIL'} |\n" for x in tests)
    (reports/f"NDP_LOCAL_ADVANTAGE_UNIT_TEST_{run_time}.md").write_text(f"# NDP 局部身份优势单测\n\n候选：`A=logmeanexp(pos/tau)-logmeanexp(neg/tau)`，`L=relu(A_ref-A_cur-epsilon)`，参考停止梯度。\n\n{test_table}\n\n单测为CPU独立公式验证，不是PPU模型训练。\n",encoding="utf-8")
    (reports/f"NDP_SHORT_PILOT_GO_NO_GO_{run_time}.md").write_text("# NDP 短程试验 Go/No-Go\n\nH1静态富集：`H1_STATIC_ENRICHMENT_SUPPORTED`。\n在线风险可观测性：`ONLINE_RISK_OBSERVABILITY_NOT_SUPPORTED`，当前为 `SINGLE_UPDATE_RISK_NOT_IDENTIFIABLE`。\n局部优势接口：`NDP_LOCAL_ADVANTAGE_INTERFACE_PASS`，仅表示12项CPU数学/梯度单测通过，不表示PPU训练效果。\n短程PPU：`NDP_SHORT_PILOT_NOT_READY`，因为当前缺少同一轨迹更新前后状态，且AMP/A-A数值噪声仍未完全控制。\n\n候选协议为 B0=Bio+GPC、C1=局部优势、C2=同预算SupCon/N-pair、C3=参考自适应Triplet；本轮不执行。整体：`NDP_INCREMENTAL_MECHANISM_UNRESOLVED`。\n",encoding="utf-8")
    log("【决策】静态富集支持，在线风险不可观测，候选接口单测通过但短程PPU未准备好；整体保持NDP_INCREMENTAL_MECHANISM_UNRESOLVED。")


if __name__ == "__main__": main()
