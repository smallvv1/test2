#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/DIFFER
PYTHON=/usr/local/bin/python3.10
TRAINER="$ROOT/scripts/gpc_a4_recheck_train.py"
EVAL="$ROOT/scripts/gpc_v2_expanded_dev_eval.py"
RUN_TIME="${1:?需要 RUN_TIME}"
OUT="$ROOT/outputs/gpc_a4_ltcc_${RUN_TIME}"
PROTOCOL="$ROOT/outputs/gpc_v2_protocol_audit_20261016_141306/expanded_dev_protocol.json"
STATUS="$OUT/状态_${RUN_TIME}.json"
LOG="$ROOT/logs/GPC身份均衡双种子复核_${RUN_TIME}.log"
TMUX_SESSION="${TMUX_SESSION:-gpc-a4-ltcc-${RUN_TIME}}"
PORT=29950
FAILED=0
HEARTBEAT_PID=""

mkdir -p "$ROOT/logs" "$ROOT/reports" "$OUT"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LIBRARY_PATH="${LIBRARY_PATH-}"
exec > >(tee -a "$LOG") 2>&1

status() {
  "$PYTHON" "$ROOT/scripts/gpc_v2_status.py" --path "$STATUS" --run-time "$RUN_TIME" --output-root "$OUT" \
    --phase "$1" --state "$2" --seed "$3" --arm "$4" --epoch "$5" --checkpoint "$6" \
    --pid "$$" --tmux-session "$TMUX_SESSION" --message "$7"
}

heartbeat_once() {
  "$PYTHON" - "$STATUS" <<'PY'
import csv, json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file(): raise SystemExit(0)
try:
    state = json.loads(p.read_text(encoding="utf-8"))
    root = Path(state["输出目录"]) / "training" / str(state["当前阶段"])
    found = []
    for metrics in root.glob("*_seed*/metrics_per_epoch.csv") if root.is_dir() else []:
        rows = list(csv.DictReader(metrics.open(encoding="utf-8")))
        if rows: found.append((metrics.stat().st_mtime, metrics, rows[-1]))
    if found:
        _, metrics, row = max(found)
        ckpt = metrics.parent / "model_final.pth"
        if not ckpt.is_file(): ckpt = metrics.parent / "model_best.pth"
        state["epoch"] = int(row.get("epoch", 0))
        state["最近checkpoint"] = str(ckpt) if ckpt.is_file() else "无"
        state["最近训练目录"] = str(metrics.parent)
    state["最后心跳时间"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
except Exception:
    pass
PY
}

heartbeat_loop() { while [[ ! -f "$OUT/.停止心跳" ]]; do heartbeat_once || true; sleep 20; done; }

run_arm() {
  local stage="$1" arm="$2" seed="$3" epochs="$4"
  local directory="$OUT/training/$stage/${arm}_seed${seed}"
  if [[ -f "$directory/RUN_COMPLETE.json" && -f "$directory/model_final.pth" ]]; then
    status "$stage" "已完成" "$seed" "$arm" "$epochs" "$directory/model_final.pth" "合法恢复：检测到完整 checkpoint。"
    return
  fi
  status "$stage" "运行中" "$seed" "$arm" 0 "无" "开始训练；不读取官方 query/gallery。"
  echo "中文说明：开始 $stage arm=$arm seed=$seed epochs=$epochs。"
  source /usr/local/PPU_SDK/envsetup.sh ppu
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 --master_port="$PORT" "$TRAINER" \
    --output "$OUT" --run-time "$RUN_TIME" --arm "$arm" --stage "$stage" --seed "$seed" --epochs "$epochs" \
    2>&1 | tee -a "$ROOT/logs/GPC身份均衡训练_${stage}_${arm}_seed${seed}_${RUN_TIME}.log"
  PORT=$((PORT + 1))
  status "$stage" "已完成" "$seed" "$arm" "$epochs" "$directory/model_final.pth" "训练完成；等待统一 PPU 评估。"
}

finish() {
  touch "$OUT/.停止心跳" 2>/dev/null || true
  [[ -n "$HEARTBEAT_PID" ]] && kill "$HEARTBEAT_PID" 2>/dev/null || true
  status "闭环归档" "$([[ "$FAILED" -eq 0 ]] && echo 已完成 || echo 失败或中断)" 1234 "B0,A4,R2" 0 "无" "历史结果未修改；官方 LTCC 测试未自动执行。" || true
  "$PYTHON" "$ROOT/scripts/gpc_a4_write_manifest.py" "$OUT" "$RUN_TIME" "$LOG" "$STATUS" "$PROTOCOL" || true
  if [[ -x "$ROOT/scripts/publish_run_to_github.sh" ]]; then
    "$ROOT/scripts/publish_run_to_github.sh" "$RUN_TIME" "$OUT" || echo "中文解释：GitHub 上传未成功；本地归档保留。"
  fi
}
trap 'rc=$?; [[ "$rc" -ne 0 ]] && FAILED=1; finish; exit "$rc"' EXIT

if [[ ! -f "$PROTOCOL" || ! -f "$OUT/implementation_audit/implementation_audit.json" || ! -f "$OUT/amp_preflight_recovery1/amp_preflight.json" || ! -f "$OUT/interface_preflight_B0.json" || ! -f "$OUT/interface_preflight_A4.json" ]]; then
  echo "中文错误：协议、实现审计、AMP 或 B0/A4 接口预检资产缺失，禁止训练。" >&2
  exit 2
fi
"$PYTHON" - "$OUT/implementation_audit/implementation_audit.json" "$OUT/amp_preflight_recovery1/amp_preflight.json" "$OUT/interface_preflight_B0.json" "$OUT/interface_preflight_A4.json" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1], encoding="utf-8")); amp = json.load(open(sys.argv[2], encoding="utf-8"))
interfaces = [json.load(open(path, encoding="utf-8")) for path in sys.argv[3:]]
if not audit["status"].startswith("PASS_SOURCE_AND_BUDGET"): raise SystemExit("实现审计未通过")
if amp.get("status") != "PASS": raise SystemExit("AMP 恢复预检未通过")
if any(item.get("status") != "PASS" for item in interfaces): raise SystemExit("B0/A4 relation_info 接口预检未通过")
print("中文门禁：源码、pair-budget、冻结划分、AMP 以及 B0/A4 双 PPU relation_info 接口预检通过。")
PY

status "E30诊断" "启动" 1234 "B0,A4,R2" 0 "无" "开始双种子 30 epoch 诊断。"
heartbeat_loop & HEARTBEAT_PID=$!
for seed in 1234 4321; do run_arm screen B0 "$seed" 30; run_arm screen A4 "$seed" 30; run_arm screen R2 "$seed" 30; done

status "E30开发评估" "运行中" 1234 "B0,A4,R2" 30 "无" "使用冻结扩展协议；模型 forward 使用 PPU。"
source /usr/local/PPU_SDK/envsetup.sh ppu
"$PYTHON" "$EVAL" --run-time "$RUN_TIME" --protocol "$PROTOCOL" --training-root "$OUT/training" --output "$OUT/evaluation_screen" \
  --specs screen_B0_seed1234 screen_A4_seed1234 screen_R2_seed1234 screen_B0_seed4321 screen_A4_seed4321 screen_R2_seed4321 \
  2>&1 | tee -a "$ROOT/logs/分析GPC扩展验证身份级可信度_${RUN_TIME}.log"
"$PYTHON" "$ROOT/scripts/gpc_a4_paired_bootstrap.py" --evaluation "$OUT/evaluation_screen/expanded_dev_evaluation.json" --protocol "$PROTOCOL" \
  --output "$OUT/bootstrap_screen" --report "$ROOT/reports/GPC_A4_PAIRED_BOOTSTRAP_REPORT.md" \
  --pairs screen_B0_seed1234:screen_A4_seed1234:E30_B0_vs_A4_seed1234 screen_B0_seed4321:screen_A4_seed4321:E30_B0_vs_A4_seed4321 \
  screen_B0_seed1234:screen_R2_seed1234:E30_B0_vs_R2_seed1234 screen_B0_seed4321:screen_R2_seed4321:E30_B0_vs_R2_seed4321 \
  2>&1 | tee -a "$ROOT/logs/分析GPC扩展验证身份级可信度_${RUN_TIME}.log"
if ! "$PYTHON" "$ROOT/scripts/gpc_a4_screen_decision.py" --evaluation "$OUT/evaluation_screen/expanded_dev_evaluation.json" \
  --output "$OUT/screen_decision.json" --report "$ROOT/reports/GPC_A4_SCREEN_DECISION.md" \
  2>&1 | tee -a "$ROOT/logs/分析GPC扩展验证身份级可信度_${RUN_TIME}.log"; then
  status "E30诊断" "A4_NOT_SUPPORTED" 1234 "A4,R2" 30 "无" "E30 双种子未通过继续 E60 的点估计门禁；停止新训练。"
  exit 0
fi

status "E60双种子复核" "运行中" 1234 "B0,A4" 0 "无" "E30 门禁通过，开始配对 B0/A4 60 epoch。"
for seed in 1234 4321; do run_arm two_seed B0 "$seed" 60; run_arm two_seed A4 "$seed" 60; done
status "E60开发评估" "运行中" 1234 "B0,A4" 60 "无" "统一使用 E60 checkpoint 和冻结扩展协议。"
source /usr/local/PPU_SDK/envsetup.sh ppu
"$PYTHON" "$EVAL" --run-time "$RUN_TIME" --protocol "$PROTOCOL" --training-root "$OUT/training" --output "$OUT/evaluation_two_seed" \
  --specs two_seed_B0_seed1234 two_seed_A4_seed1234 two_seed_B0_seed4321 two_seed_A4_seed4321 \
  2>&1 | tee -a "$ROOT/logs/GPC身份均衡双种子E60评估_${RUN_TIME}.log"
"$PYTHON" "$ROOT/scripts/gpc_a4_paired_bootstrap.py" --evaluation "$OUT/evaluation_two_seed/expanded_dev_evaluation.json" --protocol "$PROTOCOL" \
  --output "$OUT/bootstrap_two_seed" --report "$ROOT/reports/GPC_A4_TWO_SEED_CONFIRMATION.md" \
  --pairs two_seed_B0_seed1234:two_seed_A4_seed1234:E60_B0_vs_A4_seed1234 two_seed_B0_seed4321:two_seed_A4_seed4321:E60_B0_vs_A4_seed4321 \
  2>&1 | tee -a "$ROOT/logs/GPC身份均衡双种子E60评估_${RUN_TIME}.log"
status "E60闭环" "完成_待正式门禁" 1234 "B0,A4" 60 "无" "E60 开发复核完成；尚未执行官方 LTCC 测试。"
echo "中文结论：A4 E60 开发复核完成；正式 LTCC 评估需要独立门禁。"
exit 0
