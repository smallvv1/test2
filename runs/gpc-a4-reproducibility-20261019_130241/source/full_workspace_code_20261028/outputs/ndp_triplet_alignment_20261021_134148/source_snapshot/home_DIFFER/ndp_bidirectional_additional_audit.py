#!/usr/bin/env python3
"""双侧身份边界的信息增量审计；只读固定关系，不构造新训练损失。"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path


def read(p):
    with Path(p).open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pairs",type=Path,required=True); ap.add_argument("--transition-rows",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--seed",type=int,required=True); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    pairs=read(args.pairs); trans=read(args.transition_rows)
    by_interval_idx={(int(r["prev_update"]),int(r["target_update"]),int(r["relation_index"])):r for r in trans}
    rows=[]
    for p in pairs:
        interval=(int(p["prev_update"]),int(p["target_update"]))
        a=by_interval_idx.get(interval+(int(p["relation_index_a"]),)); b=by_interval_idx.get(interval+(int(p["relation_index_b"]),))
        if not a or not b: continue
        da=float(a["delta_margin"]); db=float(b["delta_margin"]); fa=a["strict_flip"]=="True"; fb=b["strict_flip"]=="True"
        rows.append({**p,"delta_a_to_b":da,"delta_b_to_a":db,"a_flip":int(fa),"b_flip":int(fb),"either_flip":int(fa or fb),"both_flip":int(fa and fb),"average_delta":(da+db)/2.0,"most_dangerous_delta":min(da,db)})
    with (args.output_dir/"bidirectional_additional_rows.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    n=len(rows); only_a=sum(x["a_flip"] and not x["b_flip"] for x in rows); only_b=sum(x["b_flip"] and not x["a_flip"] for x in rows); both=sum(x["both_flip"] for x in rows); either=sum(x["either_flip"] for x in rows)
    by_interval=defaultdict(list)
    for x in rows: by_interval[f"{x['prev_update']}->{x['target_update']}"].append(x)
    summary={"status":"PASS_BIDIRECTIONAL_ADDITIONAL_INFORMATION_AUDIT","seed":args.seed,"pair_count":n,"a_only_flip":only_a,"b_only_flip":only_b,"both_flip":both,"either_flip":either,"a_only_fraction":only_a/n if n else None,"b_only_fraction":only_b/n if n else None,"average_delta_negative_count":sum(x["average_delta"]<0 for x in rows),"most_dangerous_direction_negative_count":sum(x["most_dangerous_delta"]<0 for x in rows),"identity_pair_count":len({(min(int(x["pid_a"]),int(x["pid_b"])),max(int(x["pid_a"]),int(x["pid_b"]))) for x in rows}),"image_pair_count":n,"by_interval":{k:{"n":len(v),"a_only_flip":sum(x["a_flip"] and not x["b_flip"] for x in v),"b_only_flip":sum(x["b_flip"] and not x["a_flip"] for x in v),"either_flip":sum(x["either_flip"] for x in v)} for k,v in sorted(by_interval.items())},"future_information_used_for_selection":False,"official_query_gallery_used":False,"optimizer_step_executed":False,"interpretation":"图像级双侧诊断；小样本，不代表双侧损失收益"}
    (args.output_dir/"bidirectional_additional_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__": main()
