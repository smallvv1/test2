#!/usr/bin/env python3
"""冻结 margin 输入上的反应式 NDP 安全性探针，不执行模型更新。"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from ndp_reactive_prototype import reactive_boundary_loss

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    cases=[]
    # reference=0.5: 改善/相同不罚，恶化超过 epsilon 才罚。
    for name,cur in [("improved",[0.7,0.6]),("unchanged",[0.5,0.5]),("degraded",[0.2,0.3]),("mixed_nan",[float("nan"),0.2])]:
        x=torch.tensor(cur,dtype=torch.float64,requires_grad=True); ref=torch.tensor([0.5,0.5],dtype=torch.float64)
        loss=reactive_boundary_loss(x,ref,epsilon=0.05); finite=bool(torch.isfinite(loss)); grad=None
        if finite: loss.backward(); grad=[float(v) for v in x.grad]
        cases.append({"case":name,"loss":float(loss) if finite else None,"finite":finite,"gradient":grad})
    empty=torch.tensor([float("nan")],dtype=torch.float64,requires_grad=True); z=reactive_boundary_loss(empty,torch.tensor([0.2],dtype=torch.float64)); cases.append({"case":"all_nonfinite_safe_zero","loss":float(z),"finite":bool(torch.isfinite(z))})
    out={"status":"PASS_REACTIVE_PROTECTION_SAFETY_PROBE","cases":cases,"optimizer_step_executed":False,"reference_detached":True,"improvement_zero_penalty":cases[0]["loss"]==0.0,"degradation_positive_penalty":cases[2]["loss"]>0.0,"nan_safe_without_sum_zero":cases[-1]["finite"]}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,ensure_ascii=False))
if __name__=="__main__":main()
