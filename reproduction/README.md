# reproduction

Policy research + evaluation for OpenAI moderation domains.

## Setup

Create and activate the conda environment:

```bash
conda create -y -n dpr-lite python=3.10
conda activate dpr-lite
```

Install PyTorch first (CUDA 12.8 example):

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining dependencies:

```bash
pip install -r reproduction/requirements.txt
```

## Research run

The policy research scripts expect the model to be served via a vLLM server (see `--model_name` + `--port`).

```bash
PYTHONPATH=. python reproduction/run_research_openai_moderation.py \
  --domain "Sexual/Minors" \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --port 8001 \
  --model_tag llama-3.3-70b-instruct \
  --search_tool google
```

Add `--no_cache` to disable cached search/scrape results.
Add `--log_llm_outputs` to log raw LLM outputs for each step (useful when rule extraction is empty).
Add `--enable_reasoning` to allow model thinking (default disables reasoning via extra_body).

Or use the shell wrapper (edit model/port inside the script):

```bash
./reproduction/run_research_openai_moderation.sh --domain "Sexual/Minors"
```

The wrapper defaults to `--search_tool serper`.

To store a Serper key locally, put it in `reproduction/serper_api_key` (one line). The shell wrapper will load it if `SERPER_API_KEY` is not already set.

Research retrieval uses `NovaSearch/stella_en_1.5B_v5` by default; override with `--embedding_model_name`.

For Serper search, set `SERPER_API_KEY` and switch tools:

```bash
export SERPER_API_KEY="..."
PYTHONPATH=. python reproduction/run_research_openai_moderation.py \
  --domain "Sexual/Minors" \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --port 8001 \
  --model_tag llama-3.3-70b-instruct \
  --search_tool serper \
  --serper_gl us \
  --serper_hl en
```

Outputs land under `./logs/{exp_name}/{model_tag}/{domain}/`.

## Evaluation run

```bash
PYTHONPATH=. python reproduction/run_eval_openai_moderation.py \
  --data_path data/openai_moderation \
  --only_run_domain "Sexual/Minors" \
  --policy_dir ./logs/20250414_datastore-with-index_openai/llama-3.3-70b-instruct \
  --policy_file_name indexed_rules_YYYYMMDD-HHMM.json \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --port 8001 \
  --output_dir ./logs/eval_results \
  --output_suffix prompt_v2 \
  --prompt_strategy policy_doc \
  --pred_strategy label_only
```

The `--policy_dir` should contain subfolders for each domain (e.g., `sexual_minors/`) produced by the research run.

`deep-policy-research-public/reproduction/run_eval_openai_moderation_batch.sh` helps find the indexed rules file and runs evaluation on all domains by default (requires starting a vLLM server first).

To use a standalone policy text file instead of a datastore/index output:

```bash
PYTHONPATH=. python reproduction/run_eval_openai_moderation.py \
  --data_path data/openai_moderation \
  --only_run_domain "Sexual/Minors" \
  --policy_text_file /path/to/policy.txt \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --port 8001 \
  --output_dir ./logs/eval_results \
  --prompt_strategy policy_doc \
  --pred_strategy label_only
```

`--model_tag` is used for logging paths; it defaults to a sanitized `--model_name`.

Shell wrapper for eval (edit model/port inside the script). Outputs land under `reproduction/logs/202601_reproduction/eval_logs/20260107_qwen3-8b/<model_tag>/` by default:

```bash
./reproduction/run_eval_openai_moderation.sh \
  "Sexual/Minors" \
  ./logs/20250414_datastore-with-index_openai/llama-3.3-70b-instruct/sexual_minors/indexed_rules_YYYYMMDD-HHMM.json \
  prompt_v2
```
