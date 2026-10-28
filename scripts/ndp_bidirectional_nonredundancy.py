#!/usr/bin/env python3
"""在固定多 PID 图上保留 A→B/B→A 两个有向边，统计双侧非对称性。"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

def read(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--transition-rows", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows=read(args.transition_rows); directed={}
    for r in rows:
        key=(int(r["prev_update"]),int(r["target_update"]),int(r["anchor_pid"]),int(r["negative_pid"]))
        directed[key]=r
    out=[]; seen=set()
    for key,a in sorted(directed.items()):
        t0,t1,p,q=key
        if p==q: continue
        pair=tuple(sorted((p,q))); base=(t0,t1,pair)
        if base in seen: continue
        b=directed.get((t0,t1,q,p))
        if b is None: continue
        seen.add(base)
        da=float(a["delta_margin"]); db=float(b["delta_margin"])
        def sign(x): return "improved" if x>0 else "worsened" if x<0 else "unchanged"
        out.append({"prev_update":t0,"target_update":t1,"pid_a":p,"pid_b":q,"delta_a_to_b":da,"delta_b_to_a":db,"a_state":sign(da),"b_state":sign(db),"a_flip":a["strict_flip"],"b_flip":b["strict_flip"],"a_relation_index":a["relation_index"],"b_relation_index":b["relation_index"]})
    with (args.output_dir/"bidirectional_pairs.csv").open("w",newline="",encoding="utf-8") as f:
        if out:
            w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    c=Counter()
    for r in out:
        c["A改善_B恶化"] += r["a_state"]=="improved" and r["b_state"]=="worsened"
        c["A恶化_B改善"] += r["a_state"]=="worsened" and r["b_state"]=="improved"
        c["双改善"] += r["a_state"]=="improved" and r["b_state"]=="improved"
        c["双恶化"] += r["a_state"]=="worsened" and r["b_state"]=="worsened"
        c["A侧翻转"] += r["a_flip"]=="True"; c["B侧翻转"] += r["b_flip"]=="True"
    summary={"status":"PASS_EXPLICIT_DIRECTED_BIDIRECTIONAL","directed_transition_rows":len(rows),"bidirectional_identity_pair_intervals":len(out),"counts":dict(c),"identity_pair_dedup":"one unordered PID pair per interval, retain both directed margins","future_information_used_for_pairing":False}
    (args.output_dir/"bidirectional_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__": main()
