# JEV Classifier Benchmark — Design

**Date:** 2026-09-22
**Goal:** Measure whether TypeSafe's JEV (System One model) is a better text classifier than LLMs and BERT-style encoders, on accuracy, calibration, latency, throughput, and cost.

## Context

- **JEV** (`typesafe/jev-1.13`) is accessed through OpenRouter's beta **Decisions API**:
  `POST https://openrouter.ai/api/alpha/decisions` with body `{model, state, questions}`.
  It is not a chat model. A `choice` question returns `{choice, probabilities, confidence}`;
  the response also carries `usage.{input_tokens, output_tokens, cost}`.
  Auth: `Authorization: Bearer $OPENROUTER_API_KEY`. Pricing $0.042/M input, output free.
- **LLMs** are accessed through OpenRouter chat completions (`/api/v1/chat/completions`)
  with JSON-schema structured output. Two tiers: cheap/fast and frontier.
- **BERT** runs locally via HuggingFace `transformers`:
  - fine-tuned: `distilbert-base-uncased` trained per dataset on the train split
  - zero-shot: `facebook/bart-large-mnli` NLI pipeline

## Datasets

| id | HF dataset | classes | text field | label field |
|---|---|---|---|---|
| `sst2` | `stanfordnlp/sst2` | 2 | `sentence` | `label` |
| `agnews` | `fancyzhx/ag_news` | 4 | `text` | `label` |
| `banking77` | `PolyAI/banking77` | 77 | `text` | `label` |

- Evaluation uses a fixed, seeded sample of **N=500** from each dataset's held-out split
  (`validation` for sst2 since its test labels are hidden; `test` otherwise).
- Fine-tuning uses the `train` split (capped at 10k examples for speed).
- Each dataset exposes a label → human-readable description map. The same descriptions
  feed JEV `criteria`, the LLM prompt, and the zero-shot NLI hypotheses, so every
  zero-shot model sees identical label semantics.

## Architecture

```
src/jevbench/
  datasets.py        # load_dataset(id, n, seed) -> Dataset(name, examples, labels, descriptions)
  classifiers/
    base.py          # Classifier protocol: predict_batch(texts, labels) -> list[Prediction]
    jev.py           # Decisions API adapter (httpx, async, concurrency-limited)
    llm.py           # OpenRouter chat adapter, JSON-schema output, invalid-output tracking
    bert_finetune.py # train-or-load DistilBERT per dataset, batched inference
    bert_zeroshot.py # bart-large-mnli zero-shot pipeline
  cache.py           # sqlite cache keyed on (classifier, model, dataset, text hash)
  metrics.py         # accuracy, macro-F1, ECE, latency p50/p95, throughput, cost/1k
  run.py             # CLI: jevbench run --datasets ... --classifiers ... --n 500
  report.py          # results/*.json -> results/summary.md table
tests/               # adapter tests with recorded responses; metrics unit tests
```

### Prediction record

```python
@dataclass
class Prediction:
    label: str                      # predicted label id
    probs: dict[str, float] | None  # per-label probabilities if available
    latency_ms: float               # wall-clock for this example's request
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    raw: Any                        # provider response, persisted for debugging
    error: str | None               # invalid output / API error; label becomes None
```

### Classifier adapters

- **JEV:** one Decisions request per example; a single `choice` question whose `criteria`
  is the label description map. Probabilities come straight from the response.
  Banking77 (77 criteria) tests the many-option case.
- **LLM:** system prompt lists labels with descriptions; `response_format` JSON schema
  with `label` as an enum. `temperature=0`. Reasoning disabled where the model allows.
  Unparseable or off-enum output counts as an error (invalid-output rate metric).
  No probabilities.
- **BERT fine-tuned:** trains once per dataset (3 epochs, lr 5e-5, max_len 128), saves to
  `models/distilbert-{dataset}/`, reloads on later runs. Softmax gives probabilities.
  Runs on MPS if available.
- **BERT zero-shot:** `pipeline("zero-shot-classification")` with label descriptions as
  candidate labels. Scores treated as probabilities.

### Latency and throughput

- **Latency:** measured per example, sequential requests (concurrency=1), p50 and p95.
- **Throughput:** separate pass at a fixed concurrency (default 8 for APIs, batch size 32
  for local models), reported as examples/second. Local models are measured on the
  same Mac, with hardware noted in the report.
- Both measurements are on the same N examples. Cache is bypassed for timing runs.

### Cost

- APIs: summed from provider `usage`, reported as USD per 1k examples.
- Local models: reported as $0 API cost plus wall-clock training time.

### Metrics

Per (classifier, dataset): accuracy, macro-F1, ECE (10 bins, only where probs exist),
invalid-output rate, latency p50/p95 ms, throughput ex/s, cost per 1k, tokens per example.

### Output

- `results/{run_id}/{classifier}__{dataset}.json` with all Prediction records and metrics.
- `results/{run_id}/summary.md`: one table per dataset, rows = classifiers.
- Every API response is cached in `cache.sqlite`; reruns for accuracy are free.

## Configuration

- `OPENROUTER_API_KEY` required for JEV and LLMs. Loaded from `.env` (gitignored).
- Model IDs live in `config.toml` so the LLM tier choices can change without code edits.
  Defaults: cheap = `openai/gpt-5-mini`, frontier = `anthropic/claude-sonnet-5`.

## Error handling

- API calls retry 3x with exponential backoff on 429/5xx. After that the example is
  recorded with `error` set and excluded from accuracy but counted in error rate.
- A classifier failing entirely (e.g. missing key) is skipped with a clear message,
  and the rest of the run continues.

## Testing

- `pytest` unit tests for metrics (known inputs → known ECE, F1).
- Adapter tests using recorded JSON fixtures; no network in tests.
- A `--smoke` flag runs N=5 against live APIs as a manual end-to-end check.

## Out of scope

- Score and noul question types (only `choice` is needed for classification).
- Custom Pype data (loader is pluggable via `datasets.py` but not built now).
- Direct TypeSafe SDK access.

## Budget estimate

500 examples × 3 datasets × 2 passes (latency + throughput):
JEV ≈ $0.02, cheap LLM ≈ $0.50, frontier LLM ≈ $3–5. Total well under $10.
