#!/usr/bin/env python3
"""K=5 固定关系上的原始欧氏 SoftMargin Triplet 与反应式余弦 NDP 一阶对照。

这是冻结特征空间诊断：Triplet 使用 raw feat[0] 和原始 euclidean hard mining
公式，NDP 使用同一关系集合上统一的 mean-positive/max-negative cosine Margin。
不执行网络 forward、optimizer.step，也不把该代理称为真实 Batch-Hard 历史覆盖。
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ndp_margin_consistency import relation_margin

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--graph",type=Path,required=True); ap.add_argument("--features",type=Path,required=True); ap.add_argument("--reactive-rows",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--epsilon",type=float,default=.02); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.graph.read_text()); rels=[r for r in data["relations"] if r["graph_type"]=="difficult" and int(r["k_pid"])==5]; rels.sort(key=lambda r:(int(r["anchor_order"]),int(r["negative_pid"])))
    bank=np.load(args.features,allow_pickle=False); ui={int(u):i for i,u in enumerate(bank["updates"])}; ii={int(x):i for i,x in enumerate(bank["image_index"])}; raw=torch.tensor(bank["raw_features"][ui[200]],dtype=torch.float32,requires_grad=True); ref=torch.tensor(bank["features"][ui[0]],dtype=torch.float32)
    audit=list(csv.DictReader(args.reactive_rows.open())); audit=[r for r in audit if r["mode"]=="mean"]; assert len(audit)==len(rels)
    raw_trip=[]; ndp_margin=[]
    for r in rels:
        a=raw[ii[int(r["anchor_image_index"])] ]; p=raw[[ii[int(x)] for x in r["positive_indices"]]]; n=raw[[ii[int(x)] for x in r["negative_indices"]]]
        d_ap=torch.cdist(a[None],p).reshape(-1); d_an=torch.cdist(a[None],n).reshape(-1); raw_trip.append(F.softplus(d_ap.max()-d_an.min()))
        z=F.normalize(raw,dim=-1); av=z[ii[int(r["anchor_image_index"])] ]; pv=z[[ii[int(x)] for x in r["positive_indices"]]]; nv=z[[ii[int(x)] for x in r["negative_indices"]]]; ndp_margin.append(pv@av .mean() if False else relation_margin(raw[ii[int(r["anchor_image_index"])]],raw[[ii[int(x)] for x in r["positive_indices"]]],raw[[ii[int(x)] for x in r["negative_indices"]]],"mean"))
    trip=torch.stack(raw_trip); cur=torch.stack(ndp_margin); refs=torch.tensor([float(r["reference_margin"]) for r in audit],dtype=torch.float32); reliable=torch.tensor([r["reliability"]=="A_STABLE_CORRECT" for r in audit]); observed=torch.tensor([bool(int(r["observed_degradation"])) for r in audit]);
    cases={"T0_triplet_all":torch.ones(len(rels),dtype=torch.bool),"T1_margin_all":torch.ones(len(rels),dtype=torch.bool),"T2_reactive_reliable":reliable,"T3_reactive_reliable_observed":reliable&observed,"T4_reactive_observed":observed}
    vals={"T0_triplet_all":trip,"T1_margin_all":F.relu(torch.tensor(.2)-cur),"T2_reactive_reliable":F.relu(refs-cur-args.epsilon),"T3_reactive_reliable_observed":F.relu(refs-cur-args.epsilon),"T4_reactive_observed":F.relu(refs-cur-args.epsilon)}
    gradients={}; summary={}
    for name,mask in cases.items():
        x=raw.detach().clone().requires_grad_(True); tv=[]; nv=[]
        for r in rels:
            aa=x[ii[int(r["anchor_image_index"])] ]; pp=x[[ii[int(y)] for y in r["positive_indices"]]]; nn=x[[ii[int(y)] for y in r["negative_indices"]]]; dap=torch.cdist(aa[None],pp).reshape(-1); dan=torch.cdist(aa[None],nn).reshape(-1); tv.append(F.softplus(dap.max()-dan.min())); nv.append(relation_margin(aa,pp,nn,"mean"))
        if name.startswith("T0"): per=torch.stack(tv)
        elif name.startswith("T1"): per=F.relu(torch.tensor(.2)-torch.stack(nv))
        else: per=F.relu(refs-torch.stack(nv)-args.epsilon)
        selected=per[mask]; loss=selected.mean() if len(selected) else per.sum()*0.; g=torch.autograd.grad(loss,x)[0]; gradients[name]=g.reshape(-1); summary[name]={"selected":int(mask.sum()),"active":int(((per>0)&mask).sum()),"loss":float(loss.detach()),"gradient_norm":float(g.norm()),"gradient_finite":bool(torch.isfinite(g).all()),"pid_count":len({int(rels[i]["anchor_pid"]) for i in np.flatnonzero(mask.numpy())}),"negative_pid_count":len({int(rels[i]["negative_pid"]) for i in np.flatnonzero(mask.numpy())})}
    cosines={}
    for a,ga in gradients.items():
      for b,gb in gradients.items():
       if a<b and float(ga.norm()) and float(gb.norm()):
        aa=ga.double(); bb=gb.double(); cosines[f"{a}__{b}"]=float(torch.dot(aa,bb)/(aa.norm()*bb.norm()))
    rows=[]
    for i,(r,a) in enumerate(zip(rels,audit)): rows.append({"relation_index":i,"anchor_image_index":r["anchor_image_index"],"anchor_pid":r["anchor_pid"],"negative_pid":r["negative_pid"],"triplet_loss":float(trip[i].detach()),"ndp_margin_current":float(cur[i].detach()),"reference_margin":float(refs[i]),"reliable":int(reliable[i]),"observed_degradation":int(observed[i]),"triplet_active_continuous":int(float(trip[i].detach())>0),"ndp_active":int(float(vals["T3_reactive_reliable_observed"][i])>0)})
    with (args.output_dir/f"triplet_complementarity_rows_seed{args.seed}.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out={"status":"PASS_FROZEN_FEATURE_TRIPLET_NDP_COMPLEMENTARITY","seed":args.seed,"relation_count":len(rels),"raw_feature_shape":list(bank["raw_features"][ui[200]].shape),"raw_feature_contract":"feat[0] unnormalized; original Triplet euclidean distance", "ndp_feature_contract":"same raw tensor through FP32 cosine normalization; mean positive minus max negative", "methods":summary,"gradient_cosines_float64":cosines,"network_parameter_gradient":False,"optimizer_step_executed":False,"official_query_gallery_used":False,"interpretation":"关系级 SoftMargin 是固定候选关系代理，不是历史真实 Batch-Hard 记录"}
    (args.output_dir/f"triplet_complementarity_summary_seed{args.seed}.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(out,ensure_ascii=False))

if __name__=="__main__": main()
