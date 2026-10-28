#!/usr/bin/env python3
"""K=5 多 PID 关系上的一致 Margin、历史可靠性和反应式梯度离线审计。"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from ndp_margin_consistency import relation_margin


def read_graph(path, graph_type, k):
    data=json.loads(Path(path).read_text()); rows=[r for r in data["relations"] if r["graph_type"]==graph_type and int(r["k_pid"])==k]; rows.sort(key=lambda r:(int(r["anchor_order"]),int(r["negative_pid"])))
    if len(rows)!=512*k: raise RuntimeError(f"预期 512*{k} 关系，实际 {len(rows)}")
    return rows


def calc_numpy(mat, idx, row, mode):
    a=mat[idx[int(row["anchor_image_index"])]]; p=mat[[idx[int(x)] for x in row["positive_indices"]]]; n=mat[[idx[int(x)] for x in row["negative_indices"]]]
    pa=p@a; na=n@a
    if mode=="q25": return float(np.quantile(pa,.25)-np.max(na))
    if mode=="mean": return float(np.mean(pa)-np.max(na))
    return float(pa[0]-na[0])


def safe_zero(x):
    return torch.where(torch.isfinite(x),x,torch.zeros_like(x)).sum()*0.0


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--graph",type=Path,required=True); ap.add_argument("--features",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--epsilon",type=float,default=.02); ap.add_argument("--vulnerability-margin",type=float,default=.05); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    graph=read_graph(args.graph,"difficult",5); bank=np.load(args.features,allow_pickle=False); updates=[int(x) for x in bank["updates"]]; ui={u:i for i,u in enumerate(updates)}; image_index=[int(x) for x in bank["image_index"]]; ii={x:i for i,x in enumerate(image_index)}; mats={u:bank["features"][ui[u]].astype(np.float32) for u in updates}; modes=["q25","mean","single"]
    rows=[]
    for rid,r in enumerate(graph):
        vals={m:{u:calc_numpy(mats[u],ii,r,m) for u in updates} for m in modes}
        # 0..200 是可用于 update200 决策的历史；400 只作离线结果观察。
        for m in modes:
            hist=[vals[m][u] for u in (0,20,50,100,200)]; std=float(np.std(hist)); correct=all(x>0 for x in hist);
            rel="A_STABLE_CORRECT" if len(hist)>=3 and correct and min(hist)>=.05 and std<=.05 else ("B_CORRECT_NEAR_OR_VARIABLE" if correct else "C_HISTORY_NOT_RELIABLE")
            vuln=vals[m][200] <= args.vulnerability_margin or std > .05
            degradation=vals[m][0]-vals[m][200]
            rows.append({"seed":args.seed,"relation_index":rid,"mode":m,"anchor_image_index":int(r["anchor_image_index"]),"anchor_pid":int(r["anchor_pid"]),"positive_indices":json.dumps([int(x) for x in r["positive_indices"]]),"negative_indices":json.dumps([int(x) for x in r["negative_indices"]]),"negative_pid":int(r["negative_pid"]),"negative_pid_count":len({int(bank["pid"][ii[int(x)]]) for x in r["negative_indices"]}),"reference_margin":vals[m][0],"current_margin":vals[m][200],"validation_margin":vals[m][400],"margin_drop_0_200":degradation,"epsilon":args.epsilon,"raw_hinge":max(0.0,degradation-args.epsilon),"reliability":rel,"history_node_count":len(hist),"history_correct_rate":float(np.mean(np.asarray(hist)>0)),"history_min_margin":min(hist),"history_std":std,"vulnerability":int(vuln),"observed_degradation":int(degradation>args.epsilon),"current_correct":int(vals[m][200]>0),"validation_flip":int(vals[m][200]>0 and vals[m][400]<=0),"positive_cross_clothes":int(bool(r.get("positive_cross_clothes",False))),"negative_cross_camera":int(bool(r.get("negative_cross_camera",False))),"bidirectional_flag":int(bool(r.get("bidirectional_image_knn20",False)))})
    rows_by_mode={m:[x for x in rows if x["mode"]==m] for m in modes}; mask_specs={"reliability":lambda x:x["reliability"]=="A_STABLE_CORRECT","vulnerability":lambda x:bool(x["vulnerability"]),"observed_degradation":lambda x:bool(x["observed_degradation"]),"reliability_vulnerability":lambda x:x["reliability"]=="A_STABLE_CORRECT" and bool(x["vulnerability"]),"reliability_observed_degradation":lambda x:x["reliability"]=="A_STABLE_CORRECT" and bool(x["observed_degradation"]),"reliability_vulnerability_observed_degradation":lambda x:x["reliability"]=="A_STABLE_CORRECT" and bool(x["vulnerability"]) and bool(x["observed_degradation"])}
    # 仅对 mean 作为主候选做特征空间梯度；A/C 仍完整输出数学激活统计。
    rels=graph; x=torch.tensor(mats[200],dtype=torch.float32,requires_grad=True); refvals=[]; curvals=[]
    for r in rels:
        refvals.append(calc_numpy(mats[0],ii,r,"mean"))
        curvals.append(relation_margin(x[ii[int(r["anchor_image_index"])]],x[[ii[int(y)] for y in r["positive_indices"]]],x[[ii[int(y)] for y in r["negative_indices"]]],"mean"))
    cur=torch.stack(curvals); ref=torch.tensor(refvals,dtype=torch.float32); base=ref-cur-args.epsilon
    grad_summary={}
    for name,pred in mask_specs.items():
        mask=torch.tensor([bool(pred(row)) for row in rows_by_mode["mean"]],dtype=torch.bool)
        raw=torch.relu(base); selected=mask & torch.isfinite(raw); loss=raw[selected].mean() if bool(selected.any()) else safe_zero(cur); grad=torch.autograd.grad(loss,x,retain_graph=True)[0]
        grad_summary[name]={"selected":int(mask.sum()),"active":int((mask&(raw>0)).sum()),"loss":float(loss.detach()),"gradient_norm":float(grad.norm()),"gradient_finite":bool(torch.isfinite(grad).all())}
    csv_path=args.output_dir/f"full_multipid_reactive_rows_seed{args.seed}.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={"status":"PASS_K5_MULTIPID_REACTIVE_AUDIT","seed":args.seed,"graph":str(args.graph),"features":str(args.features),"relation_count":len(graph),"anchor_count":len({int(r['anchor_image_index']) for r in graph}),"distinct_anchor_pid_count":len({int(r['anchor_pid']) for r in graph}),"negative_pid_per_anchor":5,"modes":{m:{"rows":len(rs),"reliable_A":sum(x['reliability']=='A_STABLE_CORRECT' for x in rs),"vulnerable":sum(x['vulnerability'] for x in rs),"observed_degradation":sum(x['observed_degradation'] for x in rs),"active":sum(x['raw_hinge']>0 and x['reliability']=='A_STABLE_CORRECT' for x in rs),"validation_flips":sum(x['validation_flip'] for x in rs),"negative_pid_count":len({x['negative_pid'] for x in rs})} for m,rs in rows_by_mode.items()},"mask_gradient_summary_mean":grad_summary,"epsilon":args.epsilon,"vulnerability_margin":args.vulnerability_margin,"selection_uses_future_update400":False,"optimizer_step_executed":False,"official_query_gallery_used":False,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"rows_csv":str(csv_path)}
    (args.output_dir/f"full_multipid_reactive_summary_seed{args.seed}.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__": main()
