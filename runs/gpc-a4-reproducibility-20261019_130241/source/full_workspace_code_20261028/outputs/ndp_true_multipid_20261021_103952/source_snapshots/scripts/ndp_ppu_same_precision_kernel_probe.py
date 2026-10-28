#!/usr/bin/env python3
"""PPU 同精度、固定输入/上游梯度的无更新算子重复性探针。

这是隔离实验，不等价于 EVA02 block22 fused backward；若单算子稳定而完整模型
仍不稳定，根因仍应保留为 fused/组合路径未定位。
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import torch

def digest(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def compare(a,b):
    ad=(a-b).abs().double(); denom=torch.linalg.vector_norm(a.double()).clamp_min(1e-30)
    return {"max_abs":float(ad.max()),"relative_l2":float(torch.linalg.vector_norm((a-b).double())/denom),"cosine":float(torch.dot(a.double().flatten(),b.double().flatten())/(torch.linalg.vector_norm(a.double())*torch.linalg.vector_norm(b.double()).clamp_min(1e-30))),"finite_a":bool(torch.isfinite(a).all()),"finite_b":bool(torch.isfinite(b).all())}
def run_once(kind, device, seed=20261021):
    g=torch.Generator(device="cpu").manual_seed(seed)
    dtype=torch.float32
    if kind=="layernorm":
        m=torch.nn.LayerNorm(1024).to(device); m.weight.data.fill_(1.0); m.bias.data.zero_(); x=torch.randn((2,64,1024),generator=g,dtype=dtype); up=torch.randn_like(x)
        x=x.to(device).requires_grad_(True); up=up.to(device); y=m(x); y.backward(up); return x.grad.detach().cpu(),m.weight.grad.detach().cpu()
    if kind=="q_projection":
        m=torch.nn.Linear(1024,1024,bias=True).to(device); torch.manual_seed(seed); torch.nn.init.normal_(m.weight,std=.01); torch.nn.init.zeros_(m.bias); x=torch.randn((2,64,1024),generator=g,dtype=dtype).to(device).requires_grad_(True); up=torch.randn((2,64,1024),generator=g,dtype=dtype).to(device); y=m(x); y.backward(up); return x.grad.detach().cpu(),m.weight.grad.detach().cpu()
    if kind=="attention":
        q=torch.randn((2,4,64,64),generator=g,dtype=dtype).to(device).requires_grad_(True); k=torch.randn((2,4,64,64),generator=g,dtype=dtype).to(device).requires_grad_(True); v=torch.randn((2,4,64,64),generator=g,dtype=dtype).to(device).requires_grad_(True); up=torch.randn((2,4,64,64),generator=g,dtype=dtype).to(device); score=torch.matmul(q,k.transpose(-2,-1))/8.0; p=torch.softmax(score,dim=-1); y=torch.matmul(p,v); y.backward(up); return q.grad.detach().cpu(),k.grad.detach().cpu(),v.grad.detach().cpu()
    if kind=="softmax":
        x=torch.randn((2,4,64,64),generator=g,dtype=dtype).to(device).requires_grad_(True); up=torch.randn((2,4,64,64),generator=g,dtype=dtype).to(device); y=torch.softmax(x,dim=-1); y.backward(up); return x.grad.detach().cpu()
    if kind=="output_projection":
        m=torch.nn.Linear(1024,1024,bias=True).to(device); torch.manual_seed(seed+1); torch.nn.init.normal_(m.weight,std=.01); torch.nn.init.zeros_(m.bias); x=torch.randn((2,64,1024),generator=g,dtype=dtype).to(device).requires_grad_(True); up=torch.randn((2,64,1024),generator=g,dtype=dtype).to(device); y=m(x); y.backward(up); return x.grad.detach().cpu(),m.weight.grad.detach().cpu()
    raise ValueError(kind)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--repeats",type=int,default=3); args=ap.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("PPU/CUDA 不可用，禁止 CPU fallback")
    device=torch.device("cuda:0"); results=[]
    for kind in ("layernorm","q_projection","attention","softmax","output_projection"):
        try:
            runs=[run_once(kind,device) for _ in range(args.repeats)]
            torch.cuda.synchronize()
            per=[]
            for i in range(1,len(runs)):
                for j,(a,b) in enumerate(zip(runs[0],runs[i])): per.append({"tensor_index":j,"repeat":i,**compare(a,b)})
            results.append({"op":kind,"status":"PASS_EXECUTED","input_protocol":"fixed CPU generator, float32, fixed upstream gradient, no optimizer.step","comparisons":per,"output_hashes":[[digest(x) for x in r] for r in runs]})
        except Exception as e:
            results.append({"op":kind,"status":"FAILED","error":repr(e)})
    out={"status":"PASS_ISOLATION_PROBE_COMPLETED","device":str(device),"torch":torch.__version__,"python":sys.executable,"platform":platform.platform(),"repeats":args.repeats,"results":results,"scope_limit":"not the fused EVA02 block22 backward; stable isolated ops do not resolve full-model A/A"}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,ensure_ascii=False))
if __name__=="__main__":main()
