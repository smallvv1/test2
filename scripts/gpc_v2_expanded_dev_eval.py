#!/usr/local/bin/python3.10
"""使用 PPU 对已有 GPC-v2 LTCC checkpoint 做扩展开发验证。

本脚本不训练、不读取官方 query/gallery/test，不选择 checkpoint。模型 forward
在 PPU 上执行；特征转到 CPU 仅用于离线距离、指标、逐 Query 配对和 bootstrap。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader


ROOT = Path("/home/DIFFER")
ENGINE = Path("/data/DIFFER")
EXPECTED_DEVICE = "PPU-ZW810E"
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20261016

sys.path.insert(0, str(ENGINE))
from config import cfg as GLOBAL_CFG  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from data import build_img_transforms  # noqa: E402
from model.build import factory  # noqa: E402


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path):
    cfg = GLOBAL_CFG.clone()
    cfg.defrost()
    # yacs 会把落盘 YAML 中的数字字符串再次解码为整数；将目标节点预设为
    # 同一类型，避免 merge_from_file 在历史配置上报类型冲突。评估不读取 caption。
    cfg.DATA.BIO_INDEX = 0
    cfg.MODEL.DEVICE_ID = 0
    cfg.merge_from_file(str(path))
    cfg.freeze()
    return cfg


def make_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append(
            {
                "image_path": row["image_path"],
                "pid": int(row["pid"]),
                "camid": int(row["camera"]) - 1,
                "clothes_id": int(row["clothes"]),
            }
        )
    return result


def runtime_gate() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("PPU-only 评估拒绝执行：torch.cuda.is_available() 为 False；禁止 CPU model fallback")
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    if count < 1 or any(name != EXPECTED_DEVICE for name in names):
        raise RuntimeError(f"PPU 设备审计失败：期望至少一个 {EXPECTED_DEVICE}，实际 {names}")
    torch.cuda.set_device(0)
    print(f"中文说明：评估模型使用 PPU；设备数量={count}；设备名称={names}；CPU model fallback=False", flush=True)
    return torch.device("cuda:0")


def infer_num_classes(state: dict[str, torch.Tensor]) -> int:
    for key in ("head.weight", "module.head.weight"):
        if key in state:
            return int(state[key].shape[0])
    raise KeyError("checkpoint 中没有 head.weight，无法确定训练 PID 数")


def load_model(config_path: Path, checkpoint: Path, device: torch.device):
    cfg = load_config(config_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint 不是 state_dict：{type(state)}")
    state = {str(key).replace("module.", "", 1): value for key, value in state.items()}
    num_classes = infer_num_classes(state)
    # 12 是 LTCC 原始 train 的 camera 数；分类 head 的 PID 数从 checkpoint 严格推断。
    # checkpoint 已包含完整模型参数；关闭 timm 的额外 pretrained 初始化，
    # 保持同一 factory/拓扑但避免评估阶段重复加载预训练权重。
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=num_classes, camera_num=12)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint strict load 失败：missing={missing[:8]} unexpected={unexpected[:8]}")
    model.to(device)
    model.eval()
    return model, cfg, {"num_classes": num_classes, "checkpoint_sha256": sha256_file(checkpoint)}


def extract_features(model, cfg, query_rows: list[dict[str, Any]], gallery_rows: list[dict[str, Any]], device: torch.device) -> np.ndarray:
    _, transform_test = build_img_transforms(cfg)
    query_items = make_items(query_rows)
    gallery_items = make_items(gallery_rows)
    dataset = ConcatDataset([ImageDataset(query_items, transform=transform_test), ImageDataset(gallery_items, transform=transform_test)])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.DATA.TEST_BATCH),
        num_workers=int(cfg.DATA.NUM_WORKERS),
        pin_memory=False,
        drop_last=False,
        shuffle=False,
    )
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            cams = batch["camid"].to(device)
            features = model(images, cam_label=cams)
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError(f"PPU forward 产生非有限特征，batch={index}")
            outputs.append(features.detach().float().cpu())
            if index == 1 or index % 10 == 0:
                print(f"中文说明：PPU 评估进度 batch={index}/{len(loader)} feature_shape={tuple(features.shape)}", flush=True)
    return torch.cat(outputs, dim=0).numpy()


def query_metrics(features: np.ndarray, query_rows: list[dict[str, Any]], gallery_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_features = torch.from_numpy(features)
    all_features = F.normalize(all_features, dim=1).numpy()
    qf = all_features[: len(query_rows)]
    gf = all_features[len(query_rows) :]
    # 与仓库 R1_mAP_eval_LTCC 的 euclidean_distance 排序等价：normalized feature 上 2-2*cos。
    dist = 2.0 - 2.0 * np.matmul(qf, gf.T)
    q_pid = np.asarray([int(row["pid"]) for row in query_rows])
    g_pid = np.asarray([int(row["pid"]) for row in gallery_rows])
    q_cam = np.asarray([int(row["camera"]) - 1 for row in query_rows])
    g_cam = np.asarray([int(row["camera"]) - 1 for row in gallery_rows])
    q_cloth = np.asarray([int(row["clothes"]) for row in query_rows])
    g_cloth = np.asarray([int(row["clothes"]) for row in gallery_rows])
    paths = [row["image_path"] for row in query_rows + gallery_rows]
    per_query: list[dict[str, Any]] = []
    cmc_rows: list[np.ndarray] = []
    aps: list[float] = []
    for qi in range(len(query_rows)):
        order = np.argsort(dist[qi], kind="quicksort")
        remove = (g_pid[order] == q_pid[qi]) & (g_cam[order] == q_cam[qi])
        remove = remove | ((g_pid[order] == q_pid[qi]) & (g_cloth[order] == q_cloth[qi]))
        keep_order = order[~remove]
        matches = (g_pid[keep_order] == q_pid[qi]).astype(np.int32)
        legal_positive_count = int(matches.sum())
        if legal_positive_count == 0:
            raise RuntimeError(f"扩展协议出现无合法正样本 Query：{query_rows[qi]['image_path']}")
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        # 过滤同 camera/同 clothing 后，每个 Query 的剩余 gallery 长度可能不同；
        # 开发协议只需要固定 Rank-1/5/10，不能直接堆叠完整变长 CMC。
        cmc_rows.append(cmc[:10])
        ranks = {f"rank{k}": int(cmc[k - 1]) for k in (1, 5, 10)}
        positions = np.flatnonzero(matches)
        precision = matches.cumsum() / np.arange(1, len(matches) + 1)
        ap = float((precision * matches).sum() / legal_positive_count)
        aps.append(ap)
        per_query.append(
            {
                "query_index": qi,
                "image_path": query_rows[qi]["image_path"],
                "pid": int(query_rows[qi]["pid"]),
                "raw_pid": int(query_rows[qi]["raw_pid"]),
                "query_clothes": int(query_rows[qi]["clothes"]),
                "query_camera": int(query_rows[qi]["camera"]),
                "legal_positive_count": legal_positive_count,
                "removed_same_camera_count": int(remove.sum()),
                "rank1": ranks["rank1"],
                "rank5": ranks["rank5"],
                "rank10": ranks["rank10"],
                "AP": ap,
                "top1_path": paths[len(query_rows) + int(keep_order[0])],
                "top1_pid": int(g_pid[keep_order[0]]),
                "top1_camera": int(g_cam[keep_order[0]] + 1),
                "top1_clothes": int(g_cloth[keep_order[0]]),
                "positive_rank_positions_first": [int(x + 1) for x in positions[:10]],
            }
        )
    cmc = np.asarray(cmc_rows, dtype=np.float32).sum(axis=0) / len(cmc_rows)
    result = {
        "query_count": len(query_rows),
        "gallery_count": len(gallery_rows),
        "effective_pid_count": len({int(row["pid"]) for row in query_rows}),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": float(np.mean(aps)),
        "metric": "LTCC CC: remove same PID+camera and same PID+clothing; euclidean distance after L2 normalization",
        "same_image_matches": 0,
    }
    return result, per_query


def aggregate_pid(per_query: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pid: dict[int, list[dict[str, Any]]] = {}
    for row in per_query:
        by_pid.setdefault(int(row["raw_pid"]), []).append(row)
    result = []
    for raw_pid in sorted(by_pid):
        rows = by_pid[raw_pid]
        result.append(
            {
                "raw_pid": raw_pid,
                "pid": int(rows[0]["pid"]),
                "query_count": len(rows),
                "rank1": float(np.mean([row["rank1"] for row in rows])),
                "rank5": float(np.mean([row["rank5"] for row in rows])),
                "rank10": float(np.mean([row["rank10"] for row in rows])),
                "mAP": float(np.mean([row["AP"] for row in rows])),
                "legal_positive_count_mean": float(np.mean([row["legal_positive_count"] for row in rows])),
            }
        )
    return result


def bootstrap_pid(pid_rows: list[dict[str, Any]], repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(pid_rows)
    if n == 0:
        raise RuntimeError("没有 PID 指标，无法 bootstrap")
    metrics = {}
    for key in ("rank1", "rank5", "rank10", "mAP"):
        values = np.asarray([float(row[key]) for row in pid_rows], dtype=np.float64)
        samples = values[rng.integers(0, n, size=(repeats, n))].mean(axis=1)
        metrics[key] = {
            "estimate_pid_macro": float(values.mean()),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repeats": repeats,
        }
    return metrics


def evaluate_one(label: str, checkpoint: Path, config: Path, query_rows: list[dict[str, Any]], gallery_rows: list[dict[str, Any]], device: torch.device, output: Path) -> dict[str, Any]:
    print(f"中文说明：开始 PPU checkpoint 评估 arm={label} checkpoint={checkpoint}", flush=True)
    set_seed(1234)
    model, cfg, model_meta = load_model(config, checkpoint, device)
    features = extract_features(model, cfg, query_rows, gallery_rows, device)
    metrics, per_query = query_metrics(features, query_rows, gallery_rows)
    pid_rows = aggregate_pid(per_query)
    stage, arm, seed_text = label.rsplit("_", 2)
    result = {
        "created_utc": now(),
        "stage": stage,
        "arm": arm,
        "seed": int(seed_text.replace("seed", "")),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": model_meta["checkpoint_sha256"],
        "config": str(config),
        "runtime": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "cpu_model_fallback": False,
            "model_forward_on_ppu": True,
        },
        "metrics": metrics,
        "pid_metrics": pid_rows,
        "pid_bootstrap_95ci": bootstrap_pid(pid_rows),
        "query_metrics": per_query,
        "model_meta": model_meta,
    }
    path = output / f"{label}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model, features
    torch.cuda.empty_cache()
    print(f"中文结果：{label} Rank-1={metrics['rank1']:.6f} Rank-5={metrics['rank5']:.6f} Rank-10={metrics['rank10']:.6f} mAP={metrics['mAP']:.6f}", flush=True)
    return result


def paired(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    aa = {row["image_path"]: row for row in a["query_metrics"]}
    bb = {row["image_path"]: row for row in b["query_metrics"]}
    paths = sorted(set(aa) & set(bb))
    r1_delta = [int(bb[path]["rank1"]) - int(aa[path]["rank1"]) for path in paths]
    ap_delta = [float(bb[path]["AP"]) - float(aa[path]["AP"]) for path in paths]
    return {
        "baseline": a["arm"],
        "candidate": b["arm"],
        "seed": b["seed"],
        "paired_query_count": len(paths),
        "rank1_wins": int(sum(value > 0 for value in r1_delta)),
        "rank1_ties": int(sum(value == 0 for value in r1_delta)),
        "rank1_losses": int(sum(value < 0 for value in r1_delta)),
        "rank1_delta_mean": float(np.mean(r1_delta)),
        "AP_delta_mean": float(np.mean(ap_delta)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--training-root", default=str(ROOT / "outputs/gpc_v2_ltcc_20261015_155822/training"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    query_rows = read_rows(Path(args.protocol).with_name("expanded_query.csv"))
    gallery_rows = read_rows(Path(args.protocol).with_name("expanded_gallery.csv"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = runtime_gate()
    training_root = Path(args.training_root)
    specs = [
        ("screen_B0_seed1234", training_root / "screen/B0_seed1234"),
        ("screen_A1_seed1234", training_root / "screen/A1_seed1234"),
        ("screen_A2_seed1234", training_root / "screen/A2_seed1234"),
        ("screen_A3_seed1234", training_root / "screen/A3_seed1234"),
        ("screen_A4_seed1234", training_root / "screen/A4_seed1234"),
        ("screen_A5_seed1234", training_root / "screen/A5_seed1234"),
        ("two_seed_B0_seed1234", training_root / "two_seed/B0_seed1234"),
        ("two_seed_A5_seed1234", training_root / "two_seed/A5_seed1234"),
        ("two_seed_B0_seed4321", training_root / "two_seed/B0_seed4321"),
        ("two_seed_A5_seed4321", training_root / "two_seed/A5_seed4321"),
    ]
    results: dict[str, dict[str, Any]] = {}
    for label, directory in specs:
        checkpoint = directory / "model_final.pth"
        config = directory / "config.yml"
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(f"缺少已有 checkpoint/config：{checkpoint} / {config}")
        results[label] = evaluate_one(label, checkpoint, config, query_rows, gallery_rows, device, output)

    paired_rows = []
    for candidate in ("screen_A1_seed1234", "screen_A2_seed1234", "screen_A3_seed1234", "screen_A4_seed1234", "screen_A5_seed1234"):
        paired_rows.append(paired(results["screen_B0_seed1234"], results[candidate]))
    paired_rows.append(paired(results["two_seed_B0_seed1234"], results["two_seed_A5_seed1234"]))
    paired_rows.append(paired(results["two_seed_B0_seed4321"], results["two_seed_A5_seed4321"]))
    summary = {
        "created_utc": now(),
        "run_time": args.run_time,
        "protocol": str(Path(args.protocol)),
        "protocol_sha256": sha256_file(Path(args.protocol)),
        "query_count": len(query_rows),
        "gallery_count": len(gallery_rows),
        "effective_pid_count": len({int(row["raw_pid"]) for row in query_rows}),
        "results": results,
        "paired_comparisons": paired_rows,
        "historical_old_results_preserved": str(ROOT / "outputs/gpc_v2_ltcc_20261015_155822/dev_evaluation_screen.json"),
        "official_query_gallery_used": False,
        "cpu_model_fallback": False,
    }
    (output / "expanded_dev_evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "query_count": len(query_rows), "gallery_count": len(gallery_rows), "paired_comparisons": paired_rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
