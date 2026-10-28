#!/usr/bin/env python3
"""在同一 K=5 关系池上做预算匹配的离线选择价值审计。"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--rows",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--repeats",type=int,default=200); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(args.rows.open())); rows=[r for r in rows if r["mode"]=="mean"]; n=len(rows); rng=np.random.default_rng(20261021+args.seed)
    base={"current_hard":np.array([float(r["current_margin"])<=.05 for r in rows]),"reliability":np.array([r["reliability"]=="A_STABLE_CORRECT" for r in rows]),"reliability_observed":np.array([r["reliability"]=="A_STABLE_CORRECT" and bool(int(r["observed_degradation"])) for r in rows]),"observed":np.array([bool(int(r["observed_degradation"])) for r in rows])}
    results=[]
    def stat(name,m):
        idx=np.flatnonzero(m); return {"method":name,"selected":int(len(idx)),"active":int(sum(float(rows[i]["raw_hinge"])>0 for i in idx)),"future_flips":int(sum(int(rows[i]["validation_flip"]) for i in idx)),"future_flip_rate":float(sum(int(rows[i]["validation_flip"]) for i in idx)/len(idx)) if len(idx) else None,"anchor_pid_count":len({rows[i]["anchor_pid"] for i in idx}),"negative_pid_count":len({rows[i]["negative_pid"] for i in idx})}
    for name,m in base.items(): results.append(stat(name,m))
    for source,m in base.items():
        k=int(m.sum())
        if k==0: continue
        for j in range(args.repeats): results.append(stat(f"random_match_{source}_{j:03d}",np.isin(np.arange(n),rng.choice(n,size=k,replace=False))))
    with (args.output_dir/f"matched_relation_selection_rows_seed{args.seed}.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    summary={"status":"PASS_BUDGET_MATCHED_RELATION_SELECTION","seed":args.seed,"candidate_pool":n,"repeats":args.repeats,"methods":{name:stat(name,m) for name,m in base.items()},"random_control":"same candidate pool and selected relation count; future update400 labels used only for evaluation","future_information_used_for_selection":False,"optimizer_step_executed":False,"official_query_gallery_used":False}
    (args.output_dir/f"matched_relation_selection_summary_seed{args.seed}.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__": main()
