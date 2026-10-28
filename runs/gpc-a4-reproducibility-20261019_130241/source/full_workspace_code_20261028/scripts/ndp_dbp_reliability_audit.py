#!/usr/bin/env python3
"""LTCC train-only 的 LRD-C 漏检与 NDP-DBP 双尺度离线审计。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def lme(x, tau):
    y = x.astype(np.float64) / tau; m = y.max()
    return float(m + np.log(np.exp(y - m).mean()))


def top(scores, pool, k):
    return pool[np.argsort(-scores[pool], kind="stable")[:min(k, len(pool))]]


def probs(pos, neg, tau):
    z = np.concatenate([pos, neg]).astype(np.float64) / tau; z -= z.max(); e = np.exp(z); q = e / e.sum()
    return q, float(q[:len(pos)].sum())


def kl(p, q):
    return float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300)))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--tau", type=float, default=0.07); ap.add_argument("--mass-epsilon", type=float, default=0.02); ap.add_argument("--edge-epsilon", type=float, default=0.02); ap.add_argument("--sample-anchors", type=int, default=1024)
    args = ap.parse_args()
    root = Path("/home/DIFFER/outputs/ndp_local_intrusion_20261020_155558/feature_banks") / f"seed{args.seed}"
    bio, gpc = np.load(root / "bio_feature_bank.npz", allow_pickle=True), np.load(root / "gpc_feature_bank.npz", allow_pickle=True)
    ref = bio["features"].astype(np.float32); cur = gpc["features"].astype(np.float32)
    ref /= np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12); cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(20261021 + args.seed); anchors = np.sort(rng.choice(len(ref), min(args.sample_anchors, len(ref)), replace=False)); rs = ref[anchors] @ ref.T
    rows=[]
    for row, idx in enumerate(anchors):
        pp=np.flatnonzero(bio["pid"]==bio["pid"][idx]); pp=pp[pp!=idx]; nn=np.flatnonzero(bio["pid"]!=bio["pid"][idx]); pos=top(rs[row],pp,4); neg=top(rs[row],nn,20)
        rp,rn=ref[idx]@ref[pos].T,ref[idx]@ref[neg].T; cp,cn=cur[idx]@cur[pos].T,cur[idx]@cur[neg].T
        qref,qref_pos=probs(rp,rn,args.tau); qcur,qcur_pos=probs(cp,cn,args.tau)
        gref=lme(rp,args.tau)-lme(rn,args.tau); gcur=lme(cp,args.tau)-lme(cn,args.tau)
        eref=float(rp.min()-rn.max()); ecur=float(cp.min()-cn.max())
        reliable=bool(gref>0.05); edge_rel=bool(eref>0.0)
        rows.append({
            "anchor":int(idx),"image_path":str(bio["image_path"][idx]),"pid":int(bio["pid"][idx]),"clothes":int(bio["clothes"][idx]),"camid":int(bio["camid"][idx]),
            "positive_indices":[int(x) for x in pos],"positive_paths":[str(x) for x in bio["image_path"][pos]],"negative_indices":[int(x) for x in neg],"negative_paths":[str(x) for x in bio["image_path"][neg]],
            "positive_pid_set":sorted(set(int(x) for x in bio["pid"][pos])),"negative_pid_set":sorted(set(int(x) for x in bio["pid"][neg])),
            "reference_q_positive":qref_pos,"current_q_positive":qcur_pos,"q_positive_change":qcur_pos-qref_pos,
            "reference_advantage":gref,"current_advantage":gcur,"reference_edge":eref,"current_edge":ecur,"edge_change":ecur-eref,
            "reference_min_positive":float(rp.min()),"reference_max_negative":float(rn.max()),"current_min_positive":float(cp.min()),"current_max_negative":float(cn.max()),
            "reference_margin_correct":bool(eref>0),"current_margin_correct":bool(ecur>0),"reference_reliable_mass":reliable,"reference_reliable_edge":edge_rel,
            "lrd_a_active":bool(reliable and qref_pos-qcur_pos>args.mass_epsilon),"lrd_c_kl":kl(qref,qcur),"lrd_c_group_kl":kl(np.array([qref_pos,1-qref_pos]),np.array([qcur_pos,1-qcur_pos])),
            "lrd_c_gradient_sign_wrt_g":qcur_pos-qref_pos,"dbp_edge_active":bool(edge_rel and eref-ecur>args.edge_epsilon),"positive_cross_clothes":bool(np.any(bio["clothes"][pos]!=bio["clothes"][idx])),"negative_cross_camera":int(np.sum(bio["camid"][neg]!=bio["camid"][idx])),
        })
    reliable=[r for r in rows if r["reference_reliable_mass"]]; bad=[r for r in reliable if r["reference_margin_correct"] and not r["current_margin_correct"]]; improved_q=[r for r in bad if r["q_positive_change"]>0]
    def count(cond, xs=bad): return int(sum(bool(cond(r)) for r in xs))
    result={"status":"PASS","run_type":"train_only_lrd_c_dbp_reliability_audit","causal_warning":"Bio E60 与 GPC E60 是独立终点，不是单步训练因果证据。","seed":args.seed,"parameters":{"tau":args.tau,"mass_epsilon":args.mass_epsilon,"edge_epsilon":args.edge_epsilon,"mass_reliability":"g_ref>0.05","edge_reliability":"g_edge_ref>0"},"counts":{"valid_anchor":len(rows),"mass_reliable":len(reliable),"reference_correct_current_wrong":len(bad),"q_positive_improved_among_bad":len(improved_q),"lrd_a_active_among_bad":count(lambda r:r["lrd_a_active"]),"lrd_c_nonzero_among_bad":count(lambda r:r["lrd_c_kl"]>1e-8),"lrd_c_reverse_direction_among_bad":count(lambda r:r["q_positive_change"]>0 and r["lrd_c_kl"]>1e-8),"dbp_active_among_bad":count(lambda r:r["dbp_edge_active"]),"edge_reliable_bad":count(lambda r:r["reference_reliable_edge"]),"bad_positive_improved_dbp_active":count(lambda r:r["q_positive_change"]>0 and r["dbp_edge_active"],improved_q)},"examples":{"anchor_10":next((r for r in rows if r["anchor"]==10),None)},"rows":rows,"sources":{"bio":str(root/"bio_feature_bank.npz"),"gpc":str(root/"gpc_feature_bank.npz"),"train_only":True}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with args.output.with_suffix(".csv").open("w",newline="",encoding="utf-8") as fh:
        fields=list(rows[0]); w=csv.DictWriter(fh,fieldnames=fields); w.writeheader();
        for r in rows: w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v) for k,v in r.items()})
    print(json.dumps({"status":"PASS","seed":args.seed,"counts":result["counts"],"anchor10":result["examples"]["anchor_10"]},ensure_ascii=False))


if __name__=="__main__": main()
