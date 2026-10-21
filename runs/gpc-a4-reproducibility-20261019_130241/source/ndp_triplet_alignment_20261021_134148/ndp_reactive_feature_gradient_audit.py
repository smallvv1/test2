#!/usr/bin/env python3
"""基于已有同轨迹 update0/200/400 特征的反应式 NDP 冻结特征审计。"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))


def cos(a,b): return F.cosine_similarity(a,b,dim=-1,eps=1e-12)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True,choices=(1234,4321)); ap.add_argument("--features",type=Path,required=True); ap.add_argument("--groups",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    bank=np.load(args.features,allow_pickle=False); updates=[int(x) for x in bank["updates"]]; update_to_i={u:i for i,u in enumerate(updates)}; image_to_i={int(x):i for i,x in enumerate(bank["image_index"])}
    if not {0,20,50,100,200,400}.issubset(update_to_i): raise RuntimeError("trajectory features missing required nodes")
    groups=load(args.groups)["groups"]
    valid=[g for g in groups if all(int(x) in image_to_i for x in [g["anchor"],*g["positives"],*g["negatives"]])]
    mats=[bank["features"][update_to_i[u]].astype(np.float32) for u in updates]
    def margin(matrix,g):
        a=matrix[image_to_i[int(g["anchor"])]]; p=matrix[[image_to_i[int(x)] for x in g["positives"]]]; n=matrix[[image_to_i[int(x)] for x in g["negatives"]]]
        ps=p@a; ns=n@a; return float(np.mean(ps)-np.max(ns))
    rows=[]
    for i,g in enumerate(valid):
        ms={str(u):margin(mats[update_to_i[u]],g) for u in updates}
        hist=[ms[str(u)] for u in (0,20,50,100,200)]
        high_rel=all(x>0 for x in hist) and len(hist)>=5
        vuln=ms["200"]<=0.05
        rows.append({"group_index":i,"anchor":int(g["anchor"]),"pid":int(g["pid"]),"positive":int(g["positives"][0]),"negative":int(g["negatives"][0]),"m0":ms["0"],"m20":ms["20"],"m50":ms["50"],"m100":ms["100"],"m200":ms["200"],"m400":ms["400"],"high_reliability_pre200":int(high_rel),"high_vulnerability_at200":int(vuln),"degraded_0_200":int(ms["200"]<ms["0"]-0.02),"flip_200_400":int(ms["200"]>0 and ms["400"]<=0)} )
    n=len(rows); masks={
        "G1_fixed_margin":np.ones(n,dtype=bool),
        "G2_reference_margin":np.array([r["m200"]<r["m0"]-0.02 for r in rows]),
        "G4_current_hard_margin":np.array([r["m200"]<=0.05 for r in rows]),
        "G5_reliable_vulnerable":np.array([r["high_reliability_pre200"] and r["high_vulnerability_at200"] for r in rows],dtype=bool),
        "G6_reciprocal_image_relation":np.zeros(n,dtype=bool),
    }
    key={(r["anchor"],r["negative"]):i for i,r in enumerate(rows)}
    for r,i in key.items():
        rev=key.get((r[1],r[0])); masks["G6_reciprocal_image_relation"][i]=rev is not None
    masks["G7_reliable_vulnerable_reciprocal"]=masks["G5_reliable_vulnerable"]&masks["G6_reciprocal_image_relation"]
    rng=np.random.default_rng(20261021+args.seed); masks["G3_random_count_matched_G5"]=np.zeros(n,dtype=bool); k=int(masks["G5_reliable_vulnerable"].sum());
    if k: masks["G3_random_count_matched_G5"][rng.choice(n,size=k,replace=False)]=True
    # G0 是当前 update200 的 Batch-Hard 代理：仅使用 groups 中的真实 batch_indices，
    # 并明确不冒充历史 miner。这里按 normalized trajectory feature 做距离重建。
    current=mats[update_to_i[200]]; g0=np.zeros(n,dtype=bool)
    for i,g in enumerate(valid):
        batch=[x for x in g.get("batch_indices",[]) if int(x) in image_to_i]; aidx=image_to_i[int(g["anchor"])]
        if not batch: continue
        bidx=[image_to_i[int(x)] for x in batch]; labels=np.asarray(bank["pid"])[bidx]; ai=bidx.index(aidx) if aidx in bidx else -1
        if ai<0: continue
        d=np.sqrt(np.maximum(np.sum((current[bidx]-current[aidx])**2,axis=1),1e-12)); pos=[j for j,l in enumerate(labels) if l==labels[ai] and j!=ai]; neg=[j for j,l in enumerate(labels) if l!=labels[ai]]
        if not pos or not neg: continue
        hp=pos[int(np.argmax(d[pos]))]; hn=neg[int(np.argmin(d[neg]))]
        g0[i]=int(bidx[hp])==int(g["positives"][0]) and int(bidx[hn])==int(g["negatives"][0])
    masks["G0_batch_hard_reconstructed_proxy"]=g0

    # 关系/梯度统计：同一 update200 特征张量，非网络参数梯度。
    all_indices=sorted({int(x) for g in valid for x in [g["anchor"],*g["positives"],*g["negatives"]]}); pos_idx=torch.tensor([[image_to_i[int(g["positives"][0])]] for g in valid]); anc_idx=torch.tensor([image_to_i[int(g["anchor"])] for g in valid]); neg_idx=torch.tensor([[image_to_i[int(g["negatives"][0])]] for g in valid]); x0=torch.tensor(current,dtype=torch.float32)
    ref_margin=torch.tensor([r["m0"] for r in rows],dtype=torch.float32)
    def grad_case(name,mask):
        x=x0.detach().clone().requires_grad_(True); a=F.normalize(x[anc_idx],dim=-1); p=F.normalize(x[pos_idx[:,0]],dim=-1); q=F.normalize(x[neg_idx[:,0]],dim=-1); cm=cos(a,p)-cos(a,q); mm=torch.tensor(mask,dtype=torch.bool)
        if name.startswith("G0"): raw=torch.nn.functional.softplus(torch.sqrt(torch.clamp(2-2*cos(a,p),min=1e-12))-torch.sqrt(torch.clamp(2-2*cos(a,q),min=1e-12)))
        elif name.startswith("G1") or name.startswith("G3") or name.startswith("G4"): raw=F.relu(torch.tensor(0.2)-cm)
        else: raw=F.relu(ref_margin-cm-0.02)
        selected=raw[mm]; loss=selected.mean() if selected.numel() else cm.sum()*0.0; grad=torch.autograd.grad(loss,x)[0]; flat=grad.reshape(-1)
        selected_active = (raw > 0) & mm
        return {
            "selected_relations": int(mm.sum()),
            "active_relations": int(selected_active.sum()),
            "all_candidate_active_relations": int((raw > 0).sum()),
            "loss": float(loss),
            "gradient_norm": float(grad.norm()),
            "gradient_finite": bool(torch.isfinite(grad).all()),
            "gradient_vector": flat.detach(),
        }
    gradients={}; public={}
    for name,mask in masks.items():
        z=grad_case(name,mask); gradients[name]=z["gradient_vector"]; public[name]={k:v for k,v in z.items() if k!="gradient_vector"}; public[name]["pid_count"]=len({rows[i]["pid"] for i in np.flatnonzero(mask)}); public[name]["negative_pid_count"]=len({rows[i]["negative"] for i in np.flatnonzero(mask)}); public[name]["flip_200_400_count"]=sum(rows[i]["flip_200_400"] for i in np.flatnonzero(mask))
    cosines={}
    for a,ga in gradients.items():
        for b,gb in gradients.items():
            if a<b and float(ga.norm()) and float(gb.norm()): cosines[f"{a}__{b}"]=float(torch.dot(ga,gb)/(ga.norm()*gb.norm()))
    with (args.output_dir/"reactive_feature_rows.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"status":"PASS_UPDATE200_REACTIVE_FEATURE_GRADIENT_AUDIT","seed":args.seed,"group_count":n,"valid_group_count":n,"updates":updates,"masks":public,"gradient_cosines":cosines,"feature_source":str(args.features),"reference_update":0,"current_update":200,"validation_update":400,"network_parameter_gradient":False,"feature_space_first_order_only":True,"optimizer_step_executed":False,"official_query_gallery_used":False,"interpretation":"update200前的信息选择关系；update400只用于验证标签；冻结特征梯度不等于网络训练收益"}
    (args.output_dir/"reactive_feature_gradient_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__": main()
