#!/usr/bin/env python3
"""DBP 边界候选的数学与反例测试。"""
import torch
import torch.nn.functional as F
from ndp_dbp_candidate import dbp_edge_loss, edge_margin


def main():
    refp=torch.tensor([[0.8,0.7],[0.6,0.55]],dtype=torch.float64)
    refn=torch.tensor([[0.2,0.1,0.0],[0.5,0.45,0.4]],dtype=torch.float64)
    curp=refp.clone().requires_grad_(True); curn=refn.clone().requires_grad_(True)
    loss, info=dbp_edge_loss(refp,refn,curp,curn,epsilon=0.02)
    assert float(loss)==0.0 and info["active"]==0
    badp=torch.tensor([[0.3,0.25],[0.6,0.55]],dtype=torch.float64,requires_grad=True)
    badn=torch.tensor([[0.65,0.6,0.55],[0.5,0.45,0.4]],dtype=torch.float64,requires_grad=True)
    loss, info=dbp_edge_loss(refp,refn,badp,badn,epsilon=0.02); loss.backward()
    assert float(loss)>0 and torch.isfinite(badp.grad).all() and torch.isfinite(badn.grad).all()
    # 固定多正/多负集合时，DBP 的 min/max 边界就是固定 hard-positive/hard-negative
    # 的自适应 margin；独立性只能来自参考可靠性/双尺度门控等额外信息。
    ref_edge=edge_margin(refp,refn); cur_edge=edge_margin(badp.detach(),badn.detach())
    expected=F.relu(ref_edge-cur_edge-0.02).mean()
    assert torch.allclose(loss.detach(),expected)
    print("DBP_CANDIDATE_UNIT_TEST_PASS")


if __name__=="__main__": main()
