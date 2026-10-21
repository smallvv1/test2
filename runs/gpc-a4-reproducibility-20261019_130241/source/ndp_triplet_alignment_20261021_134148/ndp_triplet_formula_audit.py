#!/usr/bin/env python3
"""对照 DIFFER 原始 TripletLoss 与逐元素 SoftMarginLoss 公式。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    difference = torch.tensor([-20.0, -2.0, 0.0, 2.0, 20.0], dtype=torch.float64, requires_grad=True)
    target = torch.ones_like(difference)
    ref_loss = torch.nn.SoftMarginLoss(reduction="none")(difference, target)
    ref_grad = torch.autograd.grad(ref_loss.sum(), difference)[0]
    manual_loss = torch.logaddexp(torch.zeros_like(difference), -difference)
    manual_grad = -torch.sigmoid(-difference.detach())
    old_wrong_loss = torch.logaddexp(torch.zeros_like(difference), difference.detach())
    values = {
        "status": "PASS_SOFTMARGIN_FORMULA_CORRECTION",
        "input": difference.detach().tolist(),
        "torch_loss": ref_loss.tolist(), "manual_loss": manual_loss.tolist(),
        "torch_grad": ref_grad.tolist(), "manual_grad": manual_grad.tolist(),
        "old_wrong_loss": old_wrong_loss.tolist(),
        "loss_max_abs_error": float((ref_loss - manual_loss).abs().max()),
        "gradient_max_abs_error": float((ref_grad - manual_grad).abs().max()),
        "loss_direction_dan_increase": bool(float(manual_loss[0]) > float(manual_loss[-1])),
        "correct_derivative_negative": bool(torch.all(manual_grad < 0).item()),
        "hard_factor": 0.0,
        "euclidean_clip_min": 1e-12,
        "self_positive_mask": "original hard_example_mining includes diagonal; max non-diagonal when batch has another same-PID image",
        "active_threshold_is_reporting_only": True,
        "official_query_gallery_used": False,
        "optimizer_step_executed": False,
    }
    if values["loss_max_abs_error"] > 1e-12 or values["gradient_max_abs_error"] > 1e-12:
        raise AssertionError(values)
    args.output.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(values, ensure_ascii=False))


if __name__ == "__main__": main()
