#!/usr/bin/env bash
set -euo pipefail

MODEL_TAG="llama-3.1-8b-instruct"
# MODEL_TAG="qwen2.5-7b-instruct"
declare -A MODEL_ZOO
MODEL_ZOO["llama-3.3-70b-instruct"]="meta-llama/Llama-3.3-70B-Instruct|8001"
MODEL_ZOO["llama-3.1-8b-instruct"]="meta-llama/Meta-Llama-3.1-8B-Instruct|8010"
MODEL_ZOO["qwen2.5-32b-instruct"]="Qwen/Qwen2.5-32B-Instruct|8006"
MODEL_ZOO["qwen2.5-7b-instruct"]="Qwen/Qwen2.5-7B-Instruct|8010"
MODEL_ZOO["qwen3-32b"]="Qwen/Qwen3-32B|8003"
MODEL_ZOO["qwen3-8b"]="Qwen/Qwen3-8B|8004"

if [[ -z "${MODEL_ZOO[$MODEL_TAG]:-}" ]]; then
  echo "Unknown MODEL_TAG: ${MODEL_TAG}" >&2
  exit 1
fi

IFS='|' read -r MODEL_NAME PORT <<<"${MODEL_ZOO[$MODEL_TAG]}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHONPATH_VALUE="${PYTHONPATH:-}"
if [[ -z "${PYTHONPATH_VALUE}" ]]; then
  PYTHONPATH_VALUE="${REPO_DIR}"
else
  PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${REPO_DIR}"
fi
export PYTHONPATH="${PYTHONPATH_VALUE}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <eval_domain> <policy_file_path> [output_suffix]" >&2
  exit 1
fi

DATA_PATH="${REPO_DIR}/data/openai_moderation"
ONLY_RUN_DOMAIN="$1"
POLICY_FILE_PATH="$2"
OUTPUT_SUFFIX="${3:-}"
POLICY_DOMAIN_DIR="$(dirname "${POLICY_FILE_PATH}")"
POLICY_DIR="$(dirname "${POLICY_DOMAIN_DIR}")"
POLICY_FILE_NAME="$(basename "${POLICY_FILE_PATH}")"
PROMPT_VERSION="20260107_qwen3-8b"
OUTPUT_DIR="${REPO_DIR}/reproduction/logs/202601_reproduction/eval_logs/${PROMPT_VERSION}/${MODEL_TAG}"
PROMPT_STRATEGY="policy_doc"
POLICY_TEXT_FILE=""
PRED_STRATEGY="label_only"
ENABLE_REASONING="false"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/run_eval_openai_moderation.py" \
  --data_path "${DATA_PATH}" \
  --only_run_domain "${ONLY_RUN_DOMAIN}" \
  --policy_dir "${POLICY_DIR}" \
  --policy_file_name "${POLICY_FILE_NAME}" \
  --model_name "${MODEL_NAME}" \
  --port "${PORT}" \
  --output_dir "${OUTPUT_DIR}" \
  $( [[ -n "${OUTPUT_SUFFIX}" ]] && echo "--output_suffix" "${OUTPUT_SUFFIX}" ) \
  --prompt_strategy "${PROMPT_STRATEGY}" \
  --policy_text_file "${POLICY_TEXT_FILE}" \
  --pred_strategy "${PRED_STRATEGY}" \
  $( [[ "${ENABLE_REASONING}" == "true" ]] && echo "--enable_reasoning" )
