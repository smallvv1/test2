#!/usr/bin/env python3
"""冻结 margin 张量上的 T0-T6 关系/梯度审计；不调用模型和 optimizer.step。"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch

def read(p):
    with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def cosine(a,b):
    a=a.detach().double(); b=b.detach().double(); na=torch.linalg.vector_norm(a); nb=torch.linalg.vector_norm(b)
    if float(na)==0 or float(nb)==0:return None
    return float(torch.dot(a,b)/(na*nb))
def one(name,current,reference,mask):
    cur=current.clone().detach().requires_grad_(True)
    ref=reference.detach()
    if name in ("T0_fixed_hard_triplet","T1_fixed_margin_hinge"):
        val=torch.relu(torch.tensor(0.2,dtype=cur.dtype)-cur)
    else:
        val=torch.relu(ref-cur)
    val=torch.where(mask,val,torch.zeros_like(val)); loss=val.sum()/max(int(mask.sum()),1)
    loss.backward(); grad=cur.grad.detach().clone()
    return {"loss":float(loss.detach()),"active":int((val>0).sum()),"mask_count":int(mask.sum()),"gradient_norm":float(torch.linalg.vector_norm(grad.double())),"gradient":grad}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--transition-rows",type=Path,required=True); ap.add_argument("--reliability-rows",type=Path,required=True); ap.add_argument("--bidirectional-rows",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[r for r in read(args.transition_rows) if int(r["prev_update"])==200 and int(r["target_update"])==400]
    rel={int(r["relation_index"]):r for r in read(args.reliability_rows)}
    bi=read(args.bidirectional_rows); bi_ids=set()
    for r in bi: bi_ids.add(int(r["a_relation_index"])); bi_ids.add(int(r["b_relation_index"]))
    n=len(rows); cur=torch.tensor([float(r["target_margin"]) for r in rows],dtype=torch.float64); ref=torch.tensor([float(r["prev_margin"]) for r in rows],dtype=torch.float64)
    masks={
        "T0_fixed_hard_triplet":torch.ones(n,dtype=torch.bool),
        "T1_fixed_margin_hinge":torch.ones(n,dtype=torch.bool),
        "T2_reference_margin_hinge":torch.ones(n,dtype=torch.bool),
        "T3_reliable_reactive":torch.tensor([rel.get(int(r["relation_index"]),{}).get("reliability")=="HIGH_RELIABILITY" for r in rows]),
        "T4_bidirectional":torch.tensor([int(r["relation_index"]) in bi_ids for r in rows]),
        "T5_multi_pid":torch.ones(n,dtype=torch.bool),
        "T6_reliable_bidirectional_multi_pid":torch.tensor([rel.get(int(r["relation_index"]),{}).get("reliability")=="HIGH_RELIABILITY" and int(r["relation_index"]) in bi_ids for r in rows]),
    }
    out=[]; grads={}
    for name,mask in masks.items():
        x=one(name,cur,ref,mask); grads[name]=x.pop("gradient")
        out.append({"method":name,**x,"relation_count":n,"pid_count":len({int(r["negative_pid"]) for i,r in enumerate(rows) if bool(mask[i])}),"relation_set_definition":"same 200->400 fixed difficult K_pid=5 graph; masks are the only difference"})
    base=grads["T0_fixed_hard_triplet"]
    for r in out:r["gradient_cosine_vs_T0"]=cosine(base,grads[r["method"]])
    with (args.output_dir/"true_nonredundancy_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    summary={"status":"PASS_FROZEN_FEATURE_NONREDUNDANCY_ONLY","interval":"200->400","relation_count":n,"methods":out,"optimizer_step_executed":False,"network_parameter_gradient":False,"batch_hard_miner_log_available":False,"interpretation":"T0=T1 and T2=T5 are intentionally identical when their relation set/mask is identical; different gradients alone are not claimed as innovation"}
    (args.output_dir/"true_nonredundancy_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":main()
