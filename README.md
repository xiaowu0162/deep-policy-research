# deep-policy-research

`deep-policy-research` is a package for building, stress-testing, and evaluating moderation-style policies with OpenAI-compatible models.

The current repo includes:
- an installable `dpr` CLI
- schema and artifact management
- offline fixture-backed demo assets
- the packaged OpenAI moderation resplit data under `data/openai_moderation_resplit/`

Start with [docs/QUICKSTART.md](/home/diwu/ralm/deep-policy-research-public/docs/QUICKSTART.md).

Useful entry points:
- `dpr run --spec examples/cached_demo/task_spec.json --output-dir runs`
- `dpr eval --spec examples/openai_moderation_harassment_eval.json --output-dir runs`
- `dpr inspect openai-moderation`

The legacy paper reproduction path remains in [reproduction/README.md](/home/diwu/ralm/deep-policy-research-public/reproduction/README.md).
