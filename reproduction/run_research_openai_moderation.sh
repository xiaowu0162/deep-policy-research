#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHONPATH_VALUE="${PYTHONPATH:-}"
if [[ -z "${PYTHONPATH_VALUE}" ]]; then
  PYTHONPATH_VALUE="${REPO_DIR}"
else
  PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${REPO_DIR}"
fi
export PYTHONPATH="${PYTHONPATH_VALUE}"

export HF_HOME=/local3/diwu/selfrag_model_cache/
export CUDA_VISIBLE_DEVICES=3

MODEL_TAG="qwen3-8b"
DOMAIN="Harassment"
SEARCH_TOOL="serper"
LOG_LLM_OUTPUTS="false"
SERPER_KEY_FILE="$(dirname "$0")/serper_api_key"
declare -A MODEL_ZOO
MODEL_ZOO["llama-3.3-70b-instruct"]="meta-llama/Llama-3.3-70B-Instruct|8001"
MODEL_ZOO["qwen2.5-32b-instruct"]="Qwen/Qwen2.5-32B-Instruct|8006"
MODEL_ZOO["qwen3-32b"]="Qwen/Qwen3-32B|8003"
MODEL_ZOO["qwen3-8b"]="Qwen/Qwen3-8B|8004"

EMBEDDING="NovaSearch/stella_en_1.5B_v5"

if [[ -z "${MODEL_ZOO[$MODEL_TAG]:-}" ]]; then
  echo "Unknown MODEL_TAG: ${MODEL_TAG}" >&2
  exit 1
fi

IFS='|' read -r MODEL_NAME PORT <<<"${MODEL_ZOO[$MODEL_TAG]}"

if [[ -z "${SERPER_API_KEY:-}" && -f "${SERPER_KEY_FILE}" ]]; then
  export SERPER_API_KEY
  SERPER_API_KEY="$(tr -d '\r\n' <"${SERPER_KEY_FILE}")"
fi

python "${SCRIPT_DIR}/run_research_openai_moderation.py" \
  --exp_name "202601_reproduction" \
  --model_name "${MODEL_NAME}" \
  --port "${PORT}" \
  --model_tag "${MODEL_TAG}" \
  --domain "${DOMAIN}" \
  --search_tool "${SEARCH_TOOL}" \
  --embedding_model_name "${EMBEDDING}" \
  $( [[ "${LOG_LLM_OUTPUTS}" == "true" ]] && echo "--log_llm_outputs" ) \
  "$@"
