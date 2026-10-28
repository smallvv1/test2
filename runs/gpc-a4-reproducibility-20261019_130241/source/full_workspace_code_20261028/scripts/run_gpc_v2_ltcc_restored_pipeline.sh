#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/DIFFER
ENGINE=/data/DIFFER
PYTHON=/usr/local/bin/python3.10
RUN_TIME="$1"
OUT="$ROOT/outputs/gpc_v2_ltcc_$RUN_TIME"
LOG="$ROOT/logs/GPC第二版LTCC性能提升实验_$RUN_TIME.log"
STATUS="$OUT/状态_$RUN_TIME.json"
SPLIT="$ROOT/gpc_v2/ltcc_dev_split_20261015_145338.json"
PORT=29920
PUBLISHER="$ROOT/scripts/publish_run_to_github.sh"
PUBLISH_ATTEMPTED=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LIBRARY_PATH="${LIBRARY_PATH-}"

mkdir -p "$ROOT/logs" "$ROOT/reports" "$OUT"
rm -f "$OUT/.停止心跳"
exec > >(tee -a "$LOG") 2>&1

status() {
  local phase="恢复协议准备" state="运行中" seed=1234 arm="未开始" epoch=0 checkpoint="无"
  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--set" ]]; then
      local assignment="$2" key="${2%%=*}" value="${2#*=}"
      case "$key" in
        阶段) phase="$value" ;;
        状态) state="$value" ;;
        seed) seed="${value%%,*}" ;;
        arm) arm="$value" ;;
        epoch) epoch="$value" ;;
        最近checkpoint) checkpoint="$value" ;;
      esac
      shift 2
    else
      shift
    fi
  done
  "$PYTHON" "$ROOT/scripts/gpc_v2_status.py" --path "$STATUS" --run-time "$RUN_TIME" --output-root "$OUT" --phase "$phase" --state "$state" --seed "$seed" --arm "$arm" --epoch "$epoch" --checkpoint "$checkpoint" --pid "$$" --message "恢复版 GPC-v2 流水线；历史运行 20261015_145338、20261015_152045 保持不变。"
}

heartbeat() {
  "$PYTHON" - "$STATUS" "$OUT" "$$" <<'PY'
import csv, json, os, sys, time
from pathlib import Path
status_path, out, pid = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if not status_path.is_file():
    raise SystemExit(0)
try:
    state = json.loads(status_path.read_text(encoding="utf-8"))
    stage = {"单因素筛选":"screen", "开发验证":"screen", "双种子复核":"two_seed", "完整LTCC正式验证":"formal", "正式评估":"formal"}.get(state.get("阶段"))
    latest = []
    if stage:
        root = out / "training" / stage
        for path in root.glob("*_seed*/metrics_per_epoch.csv") if root.is_dir() else []:
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            if rows:
                latest.append((path.stat().st_mtime, path, rows[-1]))
    if latest:
        _, metrics, row = max(latest)
        final = metrics.parent / "model_final.pth"
        best = metrics.parent / "model_best.pth"
        state["epoch"] = int(row.get("epoch", 0))
        state["最近checkpoint"] = str(final if final.is_file() else best if best.is_file() else "无")
        state["最近训练目录"] = str(metrics.parent)
        state["最近训练损失"] = float(row.get("train_loss", "nan"))
    state["pipeline_pid"] = pid
    state["host_pid_exists"] = os.path.exists("/proc/" + str(pid))
    state["最后心跳时间"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(status_path)
except Exception:
    pass
PY
}

heartbeat_loop() {
  while [[ ! -f "$OUT/.停止心跳" ]]; do
    heartbeat || true
    sleep 20
  done
}

run_arm() {
  local stage="$1" arm="$2" seed="$3" epochs="$4" full="$5"
  local dir="$OUT/training/$stage/$arm""_seed$seed"
  if [[ -f "$dir/RUN_COMPLETE.json" && -f "$dir/model_final.pth" ]]; then
    echo "中文说明：恢复时跳过已完成 $stage $arm seed=$seed。"
    status --set "阶段=$stage" --set "状态=已完成" --set "arm=$arm" --set "seed=$seed" --set "epoch=$epochs" --set "最近checkpoint=$dir/model_final.pth"
    return
  fi
  status --set "阶段=$stage" --set "状态=运行中" --set "arm=$arm" --set "seed=$seed" --set "epoch=0" --set "最近checkpoint=无"
  echo "中文说明：开始 $stage $arm seed=$seed epochs=$epochs full_data=$full。"
  source /usr/local/PPU_SDK/envsetup.sh ppu
  local extra=""
  [[ "$full" == true ]] && extra="--full-data"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 --master_port="$PORT" "$ENGINE/scripts/gpc_v2_train.py" --output "$OUT" --run-time "$RUN_TIME" --arm "$arm" --stage "$stage" --seed "$seed" --epochs "$epochs" $extra 2>&1 | tee -a "$ROOT/logs/GPC训练_$stage""_$arm""_seed$seed""_$RUN_TIME.log"
  PORT=$((PORT + 1))
  status --set "阶段=$stage" --set "状态=已完成" --set "arm=$arm" --set "seed=$seed" --set "epoch=$epochs" --set "最近checkpoint=$dir/model_final.pth"
  echo "中文说明：完成 $stage $arm seed=$seed。"
}

finish_manifest() {
  "$PYTHON" - "$OUT" "$RUN_TIME" "$LOG" "$STATUS" <<'PY'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1])
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""): h.update(block)
    return h.hexdigest()
files = {}
for p in sorted(out.rglob("*")):
    if p.is_file() and p.name != "MANIFEST.json":
        files[str(p)] = {"size": p.stat().st_size, "sha256": digest(p)}
payload = {"run_time": sys.argv[2], "linked_previous_runs": ["20261015_145338", "20261015_152045"], "main_log": sys.argv[3], "status_json": sys.argv[4], "frozen_split": "/home/DIFFER/gpc_v2/ltcc_dev_split_20261015_145338.json", "official_query_gallery_used_during_training": False, "historical_gpc_v1_untouched": True, "prcc_touched": False, "ndp_included": False, "files": files}
(out / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

publish_on_exit() {
  if [[ "${PUBLISH_ATTEMPTED}" -eq 0 ]]; then
    PUBLISH_ATTEMPTED=1
    echo "中文说明：运行结束，开始发布代码快照和本次日志；发布失败时只保留本地结果。"
    if [[ -x "${PUBLISHER}" ]]; then
      "${PUBLISHER}" "${RUN_TIME}" "${OUT}" || echo "中文解释：GitHub 自动发布未完成；请检查 github_upload_status_${RUN_TIME}.json。"
    else
      echo "中文解释：发布器不存在：${PUBLISHER}；本次运行结果只保留在本地。"
    fi
  fi
}

trap 'touch "$OUT/.停止心跳" 2>/dev/null || true; if [[ -n "${HEARTBEAT_PID-}" ]]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi; publish_on_exit' EXIT
status --set "阶段=恢复协议准备" --set "状态=启动" --set "arm=未开始" --set "seed=1234" --set "epoch=0" --set "最近checkpoint=无"
heartbeat_loop &
HEARTBEAT_PID=$!

echo "GPC-v2 LTCC 恢复流水线；RUN_TIME=$RUN_TIME。"
echo "中文说明：关联 20261015_145338、20261015_152045；历史记录不覆盖。"

"$PYTHON" - "$OUT/caption_alignment_audit.json" "$OUT/caption_bio_gpc_ppu_preflight.json" "$SPLIT" <<'PY'
import json, sys
from pathlib import Path
asset, preflight, split = map(Path, sys.argv[1:])
if json.loads(asset.read_text(encoding="utf-8")).get("status") != "PASS": raise SystemExit("Caption 对齐未 PASS")
data = json.loads(preflight.read_text(encoding="utf-8"))
required = ["bio_loss_forward_backward", "gpc_loss_forward_backward", "combined_optimizer_step", "amp_stable", "no_nan_inf", "model_optimizer_on_ppu"]
if data.get("status") != "PASS" or any(data.get("checks", {}).get(k) is not True for k in required): raise SystemExit("Bio/GPC/AMP 预检未完整 PASS")
if len(data.get("ranks", [])) != 2 or any(x.get("status") != "PASS" for x in data["ranks"]): raise SystemExit("双 PPU rank 未同时 PASS")
if not split.is_file(): raise SystemExit("冻结划分不存在")
print("门禁 PASS：Caption、Bio、GPC、联合更新、双 PPU。")
PY

if [[ ! -f "$OUT/development_split_manifest.json" ]]; then
  "$PYTHON" "$ROOT/scripts/prepare_gpc_v2_restored_protocol.py" --run-time "$RUN_TIME" --output "$OUT" --frozen-split "$SPLIT" 2>&1 | tee -a "$ROOT/logs/恢复GPC第二版LTCC实验_$RUN_TIME.log"
fi
if [[ ! -f "$OUT/saturation_threshold.json" ]]; then
  status --set "阶段=A5阈值冻结" --set "状态=运行中" --set "arm=A5" --set "seed=1234" --set "epoch=0"
  source /usr/local/PPU_SDK/envsetup.sh ppu
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 --master_port="$PORT" "$ROOT/scripts/freeze_gpc_v2_saturation_threshold.py" --output "$OUT" 2>&1 | tee -a "$ROOT/logs/GPC冻结A5阈值_$RUN_TIME.log"
  PORT=$((PORT + 1))
fi

for arm in B0 A1 A2 A3 A4 A5; do run_arm screen "$arm" 1234 30 false; done
status --set "阶段=开发验证" --set "状态=运行中" --set "arm=B0,A1,A2,A3,A4,A5" --set "seed=1234" --set "epoch=30"
source /usr/local/PPU_SDK/envsetup.sh ppu
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 --master_port="$PORT" "$ENGINE/scripts/gpc_v2_dev_eval.py" --output "$OUT" --stage screen --epochs 30 --arms B0 A1 A2 A3 A4 A5 2>&1 | tee -a "$ROOT/logs/GPC训练时机实验_$RUN_TIME.log"
PORT=$((PORT + 1))
"$PYTHON" "$ENGINE/scripts/gpc_v2_select_candidates.py" --output "$OUT" --mode screen --report "$ROOT/reports/GPC_V2_LTCC_CANDIDATE_REPORT_$RUN_TIME.md"
selected=$("$PYTHON" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["selected_arms"]))' "$OUT/candidate_selection.json")
if [[ -z "$selected" ]]; then
  status --set "阶段=单因素筛选" --set "状态=已完成_无候选优于B0" --set "arm=无" --set "seed=1234" --set "epoch=30"
  echo "中文结论：E30 没有候选严格优于 B0，按协议停止。"
  finish_manifest
  exit 0
fi

for seed in 1234 4321; do
  run_arm two_seed B0 "$seed" 60 false
  for arm in $selected; do run_arm two_seed "$arm" "$seed" 60 false; done
done
status --set "阶段=双种子复核" --set "状态=开发验证中" --set "arm=$selected" --set "seed=1234,4321" --set "epoch=60"
source /usr/local/PPU_SDK/envsetup.sh ppu
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 --master_port="$PORT" "$ENGINE/scripts/gpc_v2_dev_eval.py" --output "$OUT" --stage two_seed --epochs 60 --arms B0 $selected 2>&1 | tee -a "$ROOT/logs/GPC双种子复核_$RUN_TIME.log"
PORT=$((PORT + 1))
"$PYTHON" "$ENGINE/scripts/gpc_v2_select_candidates.py" --output "$OUT" --mode two_seed --report "$ROOT/reports/GPC_V2_LTCC_TWO_SEED_REPORT_$RUN_TIME.md"
passed=$("$PYTHON" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["passed_arms"]))' "$OUT/two_seed_selection.json")
if [[ -z "$passed" ]]; then
  status --set "阶段=双种子复核" --set "状态=MIXED或未通过" --set "arm=无" --set "seed=1234,4321" --set "epoch=60"
  echo "中文结论：双 seed 没有一致通过候选，正式验证不启动。"
  finish_manifest
  exit 0
fi

for seed in 1234 4321; do
  run_arm formal B0 "$seed" 60 true
  for arm in $passed; do run_arm formal "$arm" "$seed" 60 true; done
done
status --set "阶段=正式评估" --set "状态=运行中" --set "arm=B0,$passed" --set "seed=1234,4321" --set "epoch=60"
for seed in 1234 4321; do
  for arm in B0 $passed; do
    checkpoint="$OUT/training/formal/$arm""_seed$seed/model_final.pth"
    config="$OUT/training/formal/$arm""_seed$seed/config.yml"
    result="$OUT/official_eval_$arm""_seed$seed.json"
    source /usr/local/PPU_SDK/envsetup.sh ppu
    "$PYTHON" "$ENGINE/scripts/evaluate_official_ltcc_independent_phase2g_a2.py" --config "$config" --weight "$checkpoint" --output "$result" 2>&1 | tee -a "$ROOT/logs/GPC最终LTCC评估_$RUN_TIME.log"
  done
done
status --set "阶段=完成" --set "状态=正式评估完成" --set "arm=B0,$passed" --set "seed=1234,4321" --set "epoch=60"
finish_manifest
echo "中文结论：GPC-v2 LTCC 流水线完成。"
