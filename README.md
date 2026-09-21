# jevbench

Is TypeSafe's **JEV** a better text classifier than LLMs or BERT? This benchmark runs five
classifiers over three public datasets and reports accuracy, calibration, latency,
throughput and cost side by side.

| Classifier | What it is | Access |
|---|---|---|
| `jev` | TypeSafe `typesafe/jev-1.13`, System One model, one `choice` question per example | OpenRouter **Decisions API** (`/api/alpha/decisions`) |
| `llm-cheap` | small fast LLM (default `openai/gpt-5-mini`), JSON-schema enum output | OpenRouter chat completions |
| `llm-frontier` | frontier LLM (default `anthropic/claude-sonnet-5`), same prompt | OpenRouter chat completions |
| `bert-ft` | `distilbert-base-uncased` fine-tuned per dataset on ≤10k train examples | local, MPS/CPU |
| `bert-zs` | `facebook/bart-large-mnli` zero-shot NLI | local, MPS/CPU |

Datasets: `sst2` (2 classes), `agnews` (4), `banking77` (77 intents). A fixed seeded sample
of N=500 from each held-out split. The same label descriptions feed JEV criteria, the LLM
prompt and the NLI hypotheses.

## Setup

```bash
uv sync --extra dev
cp .env.example .env      # put your OpenRouter key in it
uv run pytest             # 26 offline tests
```

## Run

```bash
uv run jevbench run --smoke --classifiers jev llm-cheap bert-zs --datasets sst2   # n=5 sanity check
uv run jevbench run                                                             # everything, n=500
uv run jevbench run --classifiers jev bert-ft --datasets banking77 --n 200
uv run jevbench report results/<run-id>                                         # re-render summary.md
```

Output lands in `results/<timestamp>/`: one JSON per (classifier, dataset) with every
prediction and the raw provider response, plus `summary.md` with one table per dataset.

## What gets measured

- **Accuracy, macro-F1** over examples the classifier answered. **Err%** is the share it
  failed on (API error, unparseable or off-enum LLM output).
- **ECE** (expected calibration error, 10 bins) where per-class probabilities exist: JEV and
  both BERT variants. LLMs return no probabilities, so ECE is blank for them.
- **p50 / p95 latency** per example, requests strictly sequential.
- **ex/s throughput** on a second pass at concurrency 8 for APIs, batch size 32 for local models.
- **$/1k** from provider-reported `usage.cost`. Local models are $0 API cost; fine-tuning
  wall-clock is listed under the table.

API responses are cached in `cache.sqlite`, so re-running for accuracy is free. When a
latency pass is served entirely from cache the latency columns are blank. Use `--no-cache`
to force fresh timings.

## Caveats

- Fine-tuned DistilBERT has seen thousands of labelled examples; JEV, the LLMs and the
  zero-shot model see only label descriptions. Both comparisons are useful, they answer
  different questions.
- `temperature` is not sent to LLMs because the gpt-5 family rejects it; the JSON-schema
  enum plus low reasoning effort is the determinism lever. Effort is configurable per model
  in `config.toml`; if a model rejects the reasoning parameter it is dropped automatically.
- Local latency and throughput are for this Mac and are noted in the report header.
- Budget for a full N=500 run: JEV ≈ $0.02, cheap LLM ≈ $0.50, frontier LLM ≈ $3–5.
