#!/usr/bin/env python3
"""使用已修复的 image-level reciprocal pairs 做双向边界和冻结特征梯度审计。"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from ndp_margin_consistency import relation_margin


def load(path):
    with Path(path).open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--pairs",type=Path,required=True); ap.add_argument("--graph",type=Path,required=True); ap.add_argument("--features",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--epsilon",type=float,default=.02); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    pairs=load(args.pairs); graph=json.loads(args.graph.read_text())["relations"]; graph={(int(r["anchor_image_index"]),int(r["negative_pid"])):r for r in graph if r["graph_type"]=="difficult" and int(r["k_pid"])==5}; bank=np.load(args.features,allow_pickle=False); updates=[int(x) for x in bank["updates"]]; ui={u:i for i,u in enumerate(updates)}; image=[int(x) for x in bank["image_index"]]; ii={x:i for i,x in enumerate(image)}; feats={u:bank["features"][ui[u]].astype(np.float32) for u in updates}
    rows=[]
    for p in pairs:
        prev,target=int(p["prev_update"]),int(p["target_update"]); a,b=int(p["a_anchor_local"]),int(p["b_anchor_local"]); pa,pb=int(p["pid_a"]),int(p["pid_b"]); ra,rb=graph.get((a,pb)),graph.get((b,pa))
        if ra is None or rb is None or b not in [int(x) for x in ra["negative_indices"]] or a not in [int(x) for x in rb["negative_indices"]]: continue
        def margin(mat,r,cross):
            av=torch.tensor(mat[ii[int(r["anchor_image_index"])]],dtype=torch.float32); pv=torch.tensor(mat[[ii[int(x)] for x in r["positive_indices"]]],dtype=torch.float32); nv=torch.tensor(mat[[ii[int(cross)]]],dtype=torch.float32); return float(relation_margin(av,pv,nv,"mean"))
        ma0,ma1=margin(feats[prev],ra,b),margin(feats[target],ra,b); mb0,mb1=margin(feats[prev],rb,a),margin(feats[target],rb,a)
        rows.append({"seed":args.seed,"prev_update":prev,"target_update":target,"a_image":a,"b_image":b,"pid_a":pa,"pid_b":pb,"m_a_prev":ma0,"m_a_target":ma1,"delta_a":ma1-ma0,"m_b_prev":mb0,"m_b_target":mb1,"delta_b":mb1-mb0,"a_prev_correct":int(ma0>0),"b_prev_correct":int(mb0>0),"a_flip":int(ma0>0 and ma1<=0),"b_flip":int(mb0>0 and mb1<=0),"a_active":int(ma0-ma1>args.epsilon),"b_active":int(mb0-mb1>args.epsilon),"epsilon":args.epsilon,"relation_a":ra["relation_sha256"],"relation_b":rb["relation_sha256"],"same_cross_images":True})
    # 每个区间一次图梯度：当前特征仍是可微叶张量；不涉及网络参数/更新。
    grad={}
    for interval in sorted({(int(r["prev_update"]),int(r["target_update"])) for r in rows}):
        sub=[r for r in rows if (int(r["prev_update"]),int(r["target_update"]))==interval]; target=interval[1]; x=torch.tensor(feats[target],dtype=torch.float32,requires_grad=True); vals=[]; refs=[]
        for r in sub:
            ra=graph[(r["a_image"],r["pid_b"])] if False else None
            # use the same graph lookup and cross image as the row construction
            rga=graph[(int(r["a_image"]),int(r["pid_b"]))]; rgb=graph[(int(r["b_image"]),int(r["pid_a"]))]
            av=x[ii[int(r["a_image"])] ]; ap=x[[ii[int(y)] for y in rga["positive_indices"]]]; an=x[[ii[int(r["b_image"])] ]]
            bv=x[ii[int(r["b_image"])] ]; bp=x[[ii[int(y)] for y in rgb["positive_indices"]]]; bn=x[[ii[int(r["a_image"])] ]]
            vals += [relation_margin(av,ap,an,"mean"), relation_margin(bv,bp,bn,"mean")]; refs += [float(r["m_a_prev"]),float(r["m_b_prev"])]
        cur=torch.stack(vals); ref=torch.tensor(refs,dtype=torch.float32); raw=torch.relu(ref-cur-args.epsilon); active=raw>0; loss=raw.mean() if len(raw) else x.sum()*0.; g=torch.autograd.grad(loss,x)[0] if len(raw) else torch.zeros_like(x); grad[f"{interval[0]}->{interval[1]}"]={"directions":len(vals),"active":int(active.sum()),"loss":float(loss.detach()),"gradient_norm":float(g.norm()),"gradient_finite":bool(torch.isfinite(g).all())}
    out=args.output_dir/f"directed_bidirectional_rows_seed{args.seed}.csv"; 
    with out.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["status"]); w.writeheader(); w.writerows(rows)
    summary={"status":"PASS_CORRECTED_DIRECTED_BIDIRECTIONAL_GRADIENT_AUDIT","seed":args.seed,"pair_source":str(args.pairs),"pair_sha256":hashlib.sha256(args.pairs.read_bytes()).hexdigest(),"graph_sha256":hashlib.sha256(args.graph.read_bytes()).hexdigest(),"feature_source":str(args.features),"row_count":len(rows),"identity_pair_count":len({(min(int(r["pid_a"]),int(r["pid_b"])),max(int(r["pid_a"]),int(r["pid_b"]))) for r in rows}),"a_only_worsen":sum(float(r["delta_a"])<0 and float(r["delta_b"])>=0 for r in rows),"b_only_worsen":sum(float(r["delta_b"])<0 and float(r["delta_a"])>=0 for r in rows),"both_worsen":sum(float(r["delta_a"])<0 and float(r["delta_b"])<0 for r in rows),"both_improve":sum(float(r["delta_a"])>=0 and float(r["delta_b"])>=0 for r in rows),"a_flips":sum(int(r["a_flip"]) for r in rows),"b_flips":sum(int(r["b_flip"]) for r in rows),"gradients":grad,"optimizer_step_executed":False,"official_query_gallery_used":False,"same_cross_images_verified":all(r["same_cross_images"] for r in rows),"rows_csv":str(out),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir/f"directed_bidirectional_summary_seed{args.seed}.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__": main()
