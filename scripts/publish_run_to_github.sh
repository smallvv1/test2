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
git -C "${CHECKOUT}" diff --cached --check
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
