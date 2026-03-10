# Quickstart

## Install

From the repo root:

```bash
python -m pip install -e .
```

The CLI entry point is `dpr`.

## Model Endpoint

The checked-in example specs assume a local OpenAI-compatible endpoint:

- base URL: `http://localhost:8223/v1`
- model name: `Qwen/Qwen3.5-9B`

If you use a different model or endpoint, edit the three model blocks in the example spec before running. For local endpoints, the client automatically uses a placeholder API key, so you do not need to export a real key.

## Cached Demo Run

The fastest end-to-end path is the offline cached demo:

```bash
dpr run --spec examples/cached_demo/task_spec.json --output-dir runs
```

What this does:
- loads a tiny packaged dataset from `examples/cached_demo/data/`
- runs research against fixture search results from `examples/cached_demo/search_fixture.json`
- reads local source files from `examples/cached_demo/sources/`
- runs red-team, prompt optimization, and evaluation

This demo does not need `SERPER_API_KEY` or live web access.

Useful follow-ups:

```bash
dpr resume --run-dir runs/<run_id>
dpr inspect run runs/<run_id>
```

## Evaluate The Packaged Moderation Resplit Data

For a real packaged benchmark slice, use the harassment spec:

```bash
dpr eval --spec examples/openai_moderation_harassment_eval.json --output-dir runs
```

That spec points at:
- `data/openai_moderation_resplit/harassment/train.jsonl`
- `data/openai_moderation_resplit/harassment/validation.jsonl`
- `data/openai_moderation_resplit/harassment/test.jsonl`

To inspect the available packaged tasks:

```bash
dpr inspect openai-moderation
```

To inspect a single task:

```bash
dpr inspect openai-moderation --task-id openai_moderation__harassment
```

## Run The Full Pipeline On The Packaged Moderation Data

The checked-in evaluation spec is also a reasonable starting point for a full run:

```bash
dpr run --spec examples/openai_moderation_harassment_eval.json --output-dir runs
```

By default that spec still uses the fixture-backed search file at `examples/cached_demo/search_fixture.json`, so it stays reproducible and does not require Serper.

If you want live search instead, edit the `research.search` block:

```json
{
  "provider": "serper",
  "api_key_env": "SERPER_API_KEY",
  "num_results": 10,
  "country": "us",
  "language": "en"
}
```

Then export `SERPER_API_KEY` before running.

## Notes

- `dpr run` executes `research -> redteam -> optimize -> eval`.
- `dpr eval` only needs a policy and datasets; it does not use search.
- Run artifacts are written into a timestamped directory under the `--output-dir` you pass.
