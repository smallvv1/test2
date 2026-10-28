#!/usr/bin/env python3
"""LRD-C Bernoulli KL 导数与 float64 余弦边界测试。"""
from __future__ import annotations
import torch
from ndp_lrd_abc import lrd_c_cross_identity_kl


def cosine64(x, y):
    x=x.double().flatten(); y=y.double().flatten(); nx=torch.linalg.vector_norm(x); ny=torch.linalg.vector_norm(y)
    if float(nx)==0 or float(ny)==0: return None
    return torch.dot(x,y)/(nx*ny)


def main():
    qref=torch.tensor([0.4,0.55,0.8],dtype=torch.float64)
    g=torch.tensor([-1.2,0.2,1.4],dtype=torch.float64,requires_grad=True)
    qcur=torch.sigmoid(g)
    loss=-(qref*torch.log(qcur)+(1-qref)*torch.log1p(-qcur)).sum()
    loss.backward()
    expected=qcur.detach()-qref
    assert torch.allclose(g.grad,expected,atol=1e-12,rtol=1e-12),(g.grad,expected)
    # LRD-C 对 g 是双向对称约束；Q_cur>Q_ref 时导数为正，梯度下降会降低 g。
    assert float(g.grad[2])>0 and float(g.grad[0])<0

    # 对实际 LRD-C 实现做一正一负关系回归：
    # dL/d(pos) = Q_cur-Q_ref，dL/d(neg) = -(Q_cur-Q_ref)。
    ref_pos = torch.tensor([[0.4]], dtype=torch.float64)
    ref_neg = torch.tensor([[0.1]], dtype=torch.float64)
    cur_pos = torch.tensor([[0.9]], dtype=torch.float64, requires_grad=True)
    cur_neg = torch.tensor([[0.2]], dtype=torch.float64, requires_grad=True)
    actual = lrd_c_cross_identity_kl(ref_pos, ref_neg, cur_pos, cur_neg, tau=1.0)
    actual.backward()
    qref_actual = torch.sigmoid(ref_pos.detach() - ref_neg.detach())
    qcur_actual = torch.sigmoid(cur_pos.detach() - cur_neg.detach())
    expected_actual = qcur_actual - qref_actual
    assert torch.allclose(cur_pos.grad, expected_actual, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cur_neg.grad, -expected_actual, atol=1e-6, rtol=1e-6)
    one=torch.ones(8,dtype=torch.float64); other=torch.ones(8,dtype=torch.float64)
    c=cosine64(one,other); assert c is not None and 0.999999999999<=float(c)<=1.000000000001
    z=torch.zeros(8,dtype=torch.float64); assert cosine64(z,other) is None
    print("LRD_C_FLOAT64_MATH_TEST_PASS")


if __name__=="__main__": main()
