#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_DIR="${REPO_DIR}/reproduction/logs/202601_reproduction/qwen3-8b"
EVAL_SCRIPT="${SCRIPT_DIR}/run_eval_openai_moderation.sh"


declare -A DOMAIN_MAP=(
    ["sexual"]="Sexual"
    ["hate"]="Hate"
    ["violence"]="Violence"
    ["self-harm"]="Self-Harm"
    ["harassment"]="Harassment"
)

DOMAIN_SLUGS=(sexual hate violence harassment self-harm)

shopt -s nullglob

for domain_slug in "${DOMAIN_SLUGS[@]}"; do
  domain_dir="${BASE_DIR}/${domain_slug}"
  eval_domain="${DOMAIN_MAP[$domain_slug]}"

  if [[ ! -d "${domain_dir}" ]]; then
    echo "Missing domain directory: ${domain_dir}" >&2
    continue
  fi

  datastore_files=( "${domain_dir}"/datastore_iter_*_*.json )
  if [[ ${#datastore_files[@]} -eq 0 ]]; then
    echo "No datastore_iter files found for ${eval_domain} (${domain_dir})" >&2
    continue
  fi

  for datastore_file in "${datastore_files[@]}"; do
    filename="$(basename "${datastore_file}")"
    stem="${filename%.json}"
    iter_tag="${stem%_*}"
    timestamp="${stem##*_}"
    policy_file="${domain_dir}/indexed_rules_${timestamp}.json"

    if [[ ! -f "${policy_file}" ]]; then
      echo "Missing indexed rules for ${datastore_file}: ${policy_file}" >&2
      continue
    fi

    echo "Running ${eval_domain} eval with ${policy_file} (suffix ${iter_tag})"
    "${EVAL_SCRIPT}" "${eval_domain}" "${policy_file}" "${iter_tag}"
  done
done
