#!/usr/bin/env bash
set -euo pipefail

# 运行结束后的安全发布器：只上传代码快照、本次日志和小型摘要。
# 数据集、caption、checkpoint、二进制大文件和密钥永远不进入发布目录。

ROOT=/home/DIFFER
PYTHON=/usr/local/bin/python3.10
REPO_URL="${GPC_GITHUB_REPO_URL:-git@github.com:smallvv1/test2.git}"
RUN_TIME="${1:-}"
OUTPUT_ROOT="${2:-${ROOT}/outputs/gpc_v2_ltcc_${RUN_TIME}}"
DRY_RUN=0
[[ "${3:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ ! "${RUN_TIME}" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "中文错误：RUN_TIME 必须符合 YYYYMMDD_HHMMSS，实际为：${RUN_TIME}" >&2
  exit 2
fi

PUBLISH_ROOT="${ROOT}/github_uploads"
CHECKOUT="${PUBLISH_ROOT}/test2_${RUN_TIME}"
BRANCH="${GPC_GITHUB_BRANCH:-runs/gpc-v2-ltcc-${RUN_TIME}}"
RUN_DIR="runs/gpc-v2-ltcc-${RUN_TIME}"
STAGE="${CHECKOUT}/${RUN_DIR}"
STATUS_FILE="${OUTPUT_ROOT}/github_upload_status_${RUN_TIME}.json"
mkdir -p "${PUBLISH_ROOT}"

write_status() {
  local state="$1" message="$2"
  STATUS_FILE_VALUE="${STATUS_FILE}" STATE_VALUE="${state}" MESSAGE_VALUE="${message}" RUN_TIME_VALUE="${RUN_TIME}" BRANCH_VALUE="${BRANCH}" CHECKOUT_VALUE="${CHECKOUT}" REPO_VALUE="${REPO_URL}" OUTPUT_VALUE="${OUTPUT_ROOT}" DRY_RUN_VALUE="${DRY_RUN}" "${PYTHON}" - <<'PY'
import json
import os
import pathlib
import time

path = pathlib.Path(os.environ["STATUS_FILE_VALUE"])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "run_time": os.environ["RUN_TIME_VALUE"],
    "状态": os.environ["STATE_VALUE"],
    "说明": os.environ["MESSAGE_VALUE"],
    "仓库": os.environ["REPO_VALUE"],
    "分支": os.environ["BRANCH_VALUE"],
    "本地发布目录": os.environ["CHECKOUT_VALUE"],
    "运行输出目录": os.environ["OUTPUT_VALUE"],
    "dry_run": bool(int(os.environ["DRY_RUN_VALUE"])),
    "时间": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

copy_one() {
  local source="$1" relative="$2"
  if [[ ! -f "${source}" ]]; then
    echo "中文说明：代码文件不存在，跳过：${source}"
    return 0
  fi
  case "${source}" in
    *.pth|*.pt|*.ckpt|*.npz|*.npy|*.bin|*.safetensors)
      echo "中文错误：拒绝复制模型或特征文件：${source}" >&2
      return 3
      ;;
  esac
  mkdir -p "${STAGE}/$(dirname "${relative}")"
  cp --dereference "${source}" "${STAGE}/${relative}"
}

collect_files() {
  mkdir -p "${STAGE}/code" "${STAGE}/logs" "${STAGE}/reports" "${STAGE}/metadata"

  local source
  local home_code=(
    "${ROOT}/scripts/run_gpc_v2_ltcc_pipeline.sh"
    "${ROOT}/scripts/run_gpc_v2_ltcc_restored_pipeline.sh"
    "${ROOT}/scripts/publish_run_to_github.sh"
    "${ROOT}/scripts/gpc_v2_audit.py"
    "${ROOT}/scripts/gpc_v2_status.py"
    "${ROOT}/scripts/prepare_gpc_v2_restored_protocol.py"
    "${ROOT}/scripts/freeze_gpc_v2_saturation_threshold.py"
    "${ROOT}/scripts/gpc_v2_dev_protocol_audit.py"
    "${ROOT}/scripts/gpc_v2_expanded_dev_eval.py"
    "${ROOT}/scripts/gpc_v2_paired_pid_bootstrap.py"
    "${ROOT}/scripts/gpc_gradient_compatibility_audit.py"
    "${ROOT}/scripts/gpc_v2_source_implementation_audit.py"
    "${ROOT}/scripts/gpc_v2_finalize_current_reports.py"
    "${ROOT}/scripts/gpc_a4_implementation_audit.py"
    "${ROOT}/scripts/gpc_a4_recheck_train.py"
    "${ROOT}/scripts/gpc_a4_amp_preflight.py"
    "${ROOT}/scripts/gpc_a4_interface_preflight.py"
    "${ROOT}/scripts/gpc_a4_paired_bootstrap.py"
    "${ROOT}/scripts/gpc_a4_screen_decision.py"
    "${ROOT}/scripts/gpc_a4_write_manifest.py"
    "${ROOT}/scripts/gpc_a4_result_audit.py"
    "${ROOT}/scripts/test_gpc_v2_expanded_dev_eval.py"
    "${ROOT}/scripts/run_gpc_a4_ltcc_pipeline.sh"
  )
  for source in "${home_code[@]}"; do
    [[ -f "${source}" ]] && copy_one "${source}" "code/home_DIFFER/scripts/$(basename "${source}")"
  done

  # 历史训练引擎若恢复，则自动归档；当前缺失时不伪造源码。
  local engine_code=(
    "/data/DIFFER/scripts/gpc_v2_train.py"
    "/data/DIFFER/scripts/gpc_v2_dev_eval.py"
    "/data/DIFFER/scripts/gpc_v2_select_candidates.py"
  )
  for source in "${engine_code[@]}"; do
    [[ -f "${source}" ]] && copy_one "${source}" "code/data_DIFFER/scripts/$(basename "${source}")"
  done

  # 只复制本次 RUN_TIME 的日志，不上传历史全量日志。
  while IFS= read -r source; do
    [[ -f "${source}" ]] && copy_one "${source}" "logs/$(basename "${source}")"
  done < <(find "${ROOT}/logs" -maxdepth 1 -type f -name "*_${RUN_TIME}.log" -print | sort)

  [[ -f "${OUTPUT_ROOT}/状态_${RUN_TIME}.json" ]] && copy_one "${OUTPUT_ROOT}/状态_${RUN_TIME}.json" "metadata/状态_${RUN_TIME}.json"
  [[ -f "${OUTPUT_ROOT}/MANIFEST.json" ]] && copy_one "${OUTPUT_ROOT}/MANIFEST.json" "metadata/MANIFEST_${RUN_TIME}.json"
  [[ -f "${OUTPUT_ROOT}/validation_closed_loop_manifest.json" ]] && copy_one "${OUTPUT_ROOT}/validation_closed_loop_manifest.json" "metadata/validation_closed_loop_manifest_${RUN_TIME}.json"
  [[ -f "${OUTPUT_ROOT}/gpc_a4_paired_pid_statistics.json" ]] && copy_one "${OUTPUT_ROOT}/gpc_a4_paired_pid_statistics.json" "metadata/gpc_a4_paired_pid_statistics_${RUN_TIME}.json"
  [[ -f "${OUTPUT_ROOT}/gpc_a4_exposure_statistics.json" ]] && copy_one "${OUTPUT_ROOT}/gpc_a4_exposure_statistics.json" "metadata/gpc_a4_exposure_statistics_${RUN_TIME}.json"
  [[ -f "${OUTPUT_ROOT}/gpc_a4_old_new_audit.json" ]] && copy_one "${OUTPUT_ROOT}/gpc_a4_old_new_audit.json" "metadata/gpc_a4_old_new_audit_${RUN_TIME}.json"
  local report
  for report in \
    "${ROOT}/reports/GPC_V2_EXPANDED_VALIDATION_REPORT.md" \
    "${ROOT}/reports/GPC_V2_PAIRED_PID_BOOTSTRAP.md" \
    "${ROOT}/reports/GPC_V2_SOURCE_IMPLEMENTATION_AUDIT.md" \
    "${ROOT}/reports/GPC_GRADIENT_COMPATIBILITY_AUDIT.md" \
    "${ROOT}/reports/GPC_V3_FEASIBILITY_DECISION.md" \
    "${ROOT}/reports/GPC_A4_IMPLEMENTATION_AUDIT.md" \
    "${ROOT}/reports/GPC_A4_PID_BALANCE_ANALYSIS.md" \
    "${ROOT}/reports/GPC_A4_PAIRED_BOOTSTRAP_REPORT.md" \
    "${ROOT}/reports/GPC_A4_TWO_SEED_CONFIRMATION.md" \
    "${ROOT}/reports/GPC_A4_MECHANISM_CONTROL_REPORT.md" \
    "${ROOT}/reports/GPC_A4_LTCC_FINAL_REPORT.md" \
    "${ROOT}/reports/GPC_A4_EVALUATOR_FIX.md" \
    "${ROOT}/reports/GPC_A4_OLD_NEW_REPRODUCIBILITY_AUDIT.md" \
    "${ROOT}/reports/GPC_A4_TWO_SEED_PAIRED_STATISTICS.md" \
    "${ROOT}/reports/GPC_A4_EXPOSURE_GRADIENT_AUDIT.md" \
    "${ROOT}/reports/GPC_A4_RANDOM_CONTROL_AUDIT.md" \
    "${ROOT}/reports/GPC_A4_NEXT_STAGE_DECISION.md"; do
    [[ -f "${report}" ]] && copy_one "${report}" "reports/$(basename "${report}")"
  done
  while IFS= read -r source; do
    [[ -f "${source}" ]] && copy_one "${source}" "reports/$(basename "${source}")"
  done < <(find "${ROOT}/reports" -maxdepth 1 -type f -name "*_${RUN_TIME}.md" -print | sort)
}

safety_check() {
  local path size
  while IFS= read -r path; do
    size=$(stat -c '%s' "${path}")
    (( size <= 5242880 )) || { echo "中文错误：单文件超过 5 MiB：${path}" >&2; return 4; }
    case "${path}" in
      *.pth|*.pt|*.ckpt|*.npz|*.npy|*.bin|*.safetensors|*.jpg|*.jpeg|*.png|*.tar|*.zip)
        echo "中文错误：发现禁止上传的二进制或数据文件：${path}" >&2
        return 5
        ;;
    esac
  done < <(find "${STAGE}" -type f -print)
  if rg -n -i 'BEGIN (RSA|OPENSSH) PRIVATE KEY|ghp_[A-Za-z0-9]|github_pat_[A-Za-z0-9]|AKIA[0-9A-Z]{16}|aws_secret_access_key[[:space:]]*=|api[_-]?key[[:space:]]*[:=]|password[[:space:]]*[:=]' "${STAGE}"; then
    echo "中文错误：安全扫描发现疑似凭据，拒绝上传。" >&2
    return 6
  fi
}

if [[ -e "${CHECKOUT}" && ! -d "${CHECKOUT}/.git" ]]; then
  echo "中文错误：发布目录存在但不是 Git checkout：${CHECKOUT}" >&2
  write_status "失败" "发布目录不是 Git checkout。"
  exit 7
fi
if [[ ! -e "${CHECKOUT}" ]]; then
  echo "中文说明：正在创建独立 GitHub checkout。"
  if ! git clone --depth 1 "${REPO_URL}" "${CHECKOUT}"; then
    echo "中文解释：GitHub checkout 创建失败，通常是 SSH/HTTPS 凭据或网络权限问题；科研结果保留在本地。" >&2
    write_status "BLOCKED_GITHUB_AUTH_OR_NETWORK" "无法 clone GitHub 仓库。"
    exit 20
  fi
fi
if [[ -e "${STAGE}" ]]; then
  echo "中文错误：同一 RUN_TIME 的发布目录已存在，拒绝覆盖：${STAGE}" >&2
  write_status "失败" "同一 RUN_TIME 已有发布目录。"
  exit 8
fi
if ! git -C "${CHECKOUT}" fetch --depth 1 origin main; then
  echo "中文解释：无法更新 origin/main；不会覆盖已有发布。" >&2
  write_status "BLOCKED_GITHUB_AUTH_OR_NETWORK" "无法 fetch GitHub main。"
  exit 21
fi
BASE_REF=FETCH_HEAD
if git -C "${CHECKOUT}" show-ref --verify --quiet refs/remotes/origin/main; then
  BASE_REF=origin/main
elif git -C "${CHECKOUT}" show-ref --verify --quiet refs/heads/main; then
  BASE_REF=main
fi
git -C "${CHECKOUT}" switch --detach "${BASE_REF}" >/dev/null
git -C "${CHECKOUT}" switch -c "${BRANCH}" >/dev/null

collect_files
[[ -f "${STAGE}/code/home_DIFFER/scripts/publish_run_to_github.sh" ]] || { echo "中文错误：发布器自身未进入代码快照。" >&2; exit 10; }
safety_check
cat > "${STAGE}/metadata/PUBLISH_MANIFEST_${RUN_TIME}.json" <<EOF
{
  "run_time": "${RUN_TIME}",
  "repository": "${REPO_URL}",
  "branch": "${BRANCH}",
  "contents": "代码快照、本次日志、小型状态与 MANIFEST 摘要",
  "excluded": ["dataset", "caption features", "checkpoint", "private key", "generated binary"],
  "source_checkout": "${ROOT}",
  "output_root": "${OUTPUT_ROOT}"
}
EOF

git -C "${CHECKOUT}" add "${RUN_DIR}"
# 原始运行日志允许保留 Python/PPU/Git 的原文错误；这些日志中的尾随空格
# 不应阻断代码与审计结果归档。代码、报告和 metadata 仍执行严格 whitespace 检查。
git -C "${CHECKOUT}" diff --cached --check -- . ":(exclude)${RUN_DIR}/logs/**"
if git -C "${CHECKOUT}" diff --cached --name-only | rg -n '\.(pth|pt|ckpt|npz|npy|bin|safetensors|jpg|jpeg|png|tar|zip)$'; then
  echo "中文错误：暂存区出现禁止文件，取消发布。" >&2
  write_status "失败" "暂存区安全检查失败。"
  exit 11
fi
if (( DRY_RUN == 1 )); then
  echo "中文结果：dry-run PASS；未执行 commit/push。分支=${BRANCH}，快照=${STAGE}"
  write_status "DRY_RUN_PASS" "代码、日志和安全检查通过；未 push。"
  exit 0
fi

git -C "${CHECKOUT}" -c user.name="${GPC_GITHUB_USER_NAME:-DIFFER Run Publisher}" -c user.email="${GPC_GITHUB_USER_EMAIL:-differ-run-publisher@users.noreply.github.com}" commit -m "Archive GPC-v2 LTCC run ${RUN_TIME}" >/dev/null
if ! git -C "${CHECKOUT}" push origin "${BRANCH}"; then
  echo "中文解释：commit 已保存在本地发布 checkout，但 GitHub push 因凭据/权限/网络失败；没有删除本地快照。" >&2
  write_status "BLOCKED_GITHUB_AUTH_OR_NETWORK" "本地 commit 成功，远端 push 失败。"
  exit 22
fi
echo "中文结果：代码与本次日志已上传 GitHub。分支=${BRANCH}，commit=$(git -C "${CHECKOUT}" rev-parse HEAD)"
write_status "PUBLISHED" "代码与本次日志已 push 到 GitHub。"
