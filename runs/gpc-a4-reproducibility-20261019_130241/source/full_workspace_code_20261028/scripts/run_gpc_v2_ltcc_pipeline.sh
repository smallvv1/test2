#!/usr/bin/env bash
set -u

# GPC-v2 LTCC 可恢复流水线。正式训练只在来源审计、数据泄漏审计和
# 双 PPU preflight 全部通过后进入；任何门禁失败都停止，不自动替换资产。

ROOT=/home/DIFFER
PYTHON=/usr/local/bin/python3
RUN_TIME=${RUN_TIME:-$(date +"%Y%m%d_%H%M%S")}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/outputs/gpc_v2_ltcc_${RUN_TIME}}
LOG=${ROOT}/logs/GPC第二版LTCC性能提升实验_${RUN_TIME}.log
STATUS=${OUTPUT_ROOT}/状态_${RUN_TIME}.json
PUBLISHER=${ROOT}/scripts/publish_run_to_github.sh
PUBLISH_ATTEMPTED=0
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${OUTPUT_ROOT}"

publish_on_exit() {
  if [[ "${PUBLISH_ATTEMPTED}" -eq 0 ]]; then
    PUBLISH_ATTEMPTED=1
    echo "中文说明：基础流水线结束，开始发布代码快照和本次日志。"
    if [[ -x "${PUBLISHER}" ]]; then
      "${PUBLISHER}" "${RUN_TIME}" "${OUTPUT_ROOT}" || echo "中文解释：GitHub 自动发布未完成；本地结果保留。"
    else
      echo "中文解释：发布器不存在：${PUBLISHER}；本次结果只保留在本地。"
    fi
  fi
}
trap publish_on_exit EXIT

write_status() {
  local phase="$1" state="$2" seed="${3:-1234}" arm="${4:-未开始}" epoch="${5:-0}" checkpoint="${6:-无}" message="${7:-}"
  STATUS_PATH="${STATUS}" RUN_TIME_VALUE="${RUN_TIME}" OUTPUT_ROOT_VALUE="${OUTPUT_ROOT}" PHASE_VALUE="${phase}" STATE_VALUE="${state}" SEED_VALUE="${seed}" ARM_VALUE="${arm}" EPOCH_VALUE="${epoch}" CHECKPOINT_VALUE="${checkpoint}" MESSAGE_VALUE="${message}" "${PYTHON}" - <<'PY'
import json
import os
import socket
import time

path = os.environ["STATUS_PATH"]
payload = {
    "run_time": os.environ["RUN_TIME_VALUE"],
    "主机": socket.gethostname(),
    "pid": os.getppid(),
    "tmux_session": None,
    "当前阶段": os.environ["PHASE_VALUE"],
    "状态": os.environ["STATE_VALUE"],
    "seed": int(os.environ["SEED_VALUE"]),
    "arm": os.environ["ARM_VALUE"],
    "epoch": int(os.environ["EPOCH_VALUE"]),
    "最近checkpoint": os.environ["CHECKPOINT_VALUE"],
    "最后心跳时间": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "输出目录": os.environ["OUTPUT_ROOT_VALUE"],
    "说明": os.environ["MESSAGE_VALUE"],
}
os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
os.replace(tmp, path)
PY
}

{
  echo "GPC-v2 LTCC 实验流水线；RUN_TIME=${RUN_TIME}"
  echo "中文说明：正式训练需要 Source Audit、Data Leakage Audit、双 PPU preflight 全部通过。"
  echo "主日志：${LOG}"
  write_status "来源审计与数据泄漏审计" "进行中" 1234 "未开始" 0 "无" "只读检查历史 GPC-v1、开发划分、caption 资产和测试边界。"
  if ! RUN_TIME="${RUN_TIME}" "${PYTHON}" "${ROOT}/scripts/gpc_v2_audit.py"; then
    write_status "来源审计与数据泄漏审计" "失败" 1234 "未开始" 0 "无" "审计脚本异常；未启动任何训练。"
    exit 2
  fi
  AUDIT_STATUS="${ROOT}/gpc_v2/audit_status_${RUN_TIME}.json"
  SOURCE_PASS=$(AUDIT_STATUS="${AUDIT_STATUS}" "${PYTHON}" - <<'PY'
import json, os
d = json.load(open(os.environ["AUDIT_STATUS"], encoding="utf-8"))
print("1" if d.get("source_audit_pass") else "0")
PY
)
  LEAKAGE_PASS=$(AUDIT_STATUS="${AUDIT_STATUS}" "${PYTHON}" - <<'PY'
import json, os
d = json.load(open(os.environ["AUDIT_STATUS"], encoding="utf-8"))
print("1" if d.get("leakage_audit_pass") else "0")
PY
)
  if [[ "${SOURCE_PASS}" != "1" || "${LEAKAGE_PASS}" != "1" ]]; then
    echo "中文结论：Source Audit 或 Data Leakage Audit 未通过，停止在正式训练门禁之前。"
    echo "中文原因：当前运行时缺少 LTCC Bio loss 所需的 train-only DIFFER_TextCaption caption 特征。"
    write_status "来源审计与数据泄漏审计" "BLOCKED" 1234 "未开始" 0 "无" "缺少 train-only caption 特征；没有启动候选训练、双 seed 或官方测试。"
    exit 3
  fi
  write_status "双 PPU preflight" "进行中" 1234 "未开始" 0 "无" "运行双卡 forward、backward、optimizer.step。"
  PREFLIGHT_LOG="${ROOT}/logs/GPC双PPU预检_${RUN_TIME}.log"
  source /usr/local/PPU_SDK/envsetup.sh ppu
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 "${ROOT}/gpc_v2_ppu_preflight.py" 2>&1 | tee "${PREFLIGHT_LOG}"
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
    write_status "双 PPU preflight" "失败" 1234 "未开始" 0 "无" "双 PPU preflight 失败；停止且不修改科学协议。"
    exit 4
  fi
  write_status "正式流水线待启动" "READY" 1234 "未开始" 0 "无" "所有门禁通过；当前脚本只在此处交给后续阶段。"
  echo "中文结论：门禁全部通过；候选阶段尚未在此脚本中自动改变参数。"
} 2>&1 | tee "${LOG}"
