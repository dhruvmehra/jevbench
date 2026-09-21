# JEV Classifier Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI that runs JEV, two OpenRouter LLMs, fine-tuned DistilBERT and zero-shot BART-MNLI over three text-classification datasets and emits accuracy, calibration, latency, throughput and cost tables.

**Architecture:** One `Classifier` protocol with an async `predict_batch`; four adapters behind it. A `run` command does a sequential latency/accuracy pass and a concurrent throughput pass per (classifier, dataset), writes JSON, and `report` renders markdown. API responses are cached in sqlite.

**Tech Stack:** Python 3.12 via uv, httpx (async), HuggingFace `datasets` + `transformers` + `torch` (MPS), numpy, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-22-jev-classifier-benchmark-design.md`

## Global Constraints

- Python 3.12 (pinned in `.python-version`; torch wheels for 3.14 are not assumed).
- JEV endpoint: `POST https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`.
- LLM endpoint: `POST https://openrouter.ai/api/v1/chat/completions`, `usage: {"include": true}` on every request so `usage.cost` is returned.
- Auth header for both: `Authorization: Bearer $OPENROUTER_API_KEY`. Key comes from `.env` (gitignored) via python-dotenv.
- Datasets: `stanfordnlp/sst2` (validation split), `fancyzhx/ag_news` (test), `PolyAI/banking77` (test). N=500, seed=0. Train cap 10k.
- No network in unit tests. Use `httpx.MockTransport` and injected fakes.
- Every `Prediction` with `error` set is excluded from accuracy/F1/ECE and counted in error rate.

---

### Task 1: Project scaffold and metrics

**Files:**
- Create: `pyproject.toml`, `.python-version`, `config.toml`, `.env.example`, `src/jevbench/__init__.py`, `src/jevbench/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `accuracy(y_true: list[str], y_pred: list[str|None]) -> float` (over non-None preds; 0.0 if none)
  - `macro_f1(y_true, y_pred, labels: list[str]) -> float`
  - `ece(y_true: list[str], probs: list[dict[str,float]|None], n_bins=10) -> float|None` (None if no probs)
  - `percentile(values: list[float], p: float) -> float`
  - `error_rate(preds_errors: list[str|None]) -> float`

- [ ] **Step 1: Scaffold**

```bash
cd /Users/dhruvmehra/Desktop/Claude/Work/JEV
echo 3.12 > .python-version
```

`pyproject.toml`:
```toml
[project]
name = "jevbench"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "httpx>=0.27",
  "datasets>=3.0",
  "transformers>=4.45",
  "torch>=2.4",
  "numpy>=1.26",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.24"]

[project.scripts]
jevbench = "jevbench.run:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jevbench"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

`config.toml`:
```toml
[jev]
model = "typesafe/jev-1.13"

[llm.cheap]
model = "openai/gpt-5-mini"
reasoning_effort = "minimal"

[llm.frontier]
model = "anthropic/claude-sonnet-5"
reasoning_effort = "low"

[bert]
finetune_base = "distilbert-base-uncased"
zeroshot_model = "facebook/bart-large-mnli"
train_cap = 10000
epochs = 3
lr = 5e-5
max_len = 128
batch_size = 32

[run]
n = 500
seed = 0
api_concurrency = 8
local_batch_size = 32
```

`.env.example`: `OPENROUTER_API_KEY=sk-or-...`

Run: `uv sync --extra dev`

- [ ] **Step 2: Failing metrics tests** (`tests/test_metrics.py`)

```python
from jevbench.metrics import accuracy, macro_f1, ece, percentile, error_rate

def test_accuracy_ignores_none():
    assert accuracy(["a","b","a"], ["a","b",None]) == 1.0
    assert accuracy(["a","b"], ["a","a"]) == 0.5
    assert accuracy(["a"], [None]) == 0.0

def test_macro_f1_perfect_and_half():
    assert macro_f1(["a","b"], ["a","b"], ["a","b"]) == 1.0
    # a: tp=1 fp=1 fn=0 -> p=.5 r=1 f=.667 ; b: tp=0 fp=0 fn=1 -> f=0 ; macro=.333
    assert abs(macro_f1(["a","b"], ["a","a"], ["a","b"]) - 1/3) < 1e-9

def test_ece_calibrated_is_zero_and_none_without_probs():
    probs = [{"a":1.0,"b":0.0}, {"a":0.0,"b":1.0}]
    assert ece(["a","b"], probs) == 0.0
    assert ece(["a"], [None]) is None
    # confident but wrong -> ece = 1.0
    assert ece(["a"], [{"a":0.0,"b":1.0}]) == 1.0

def test_percentile():
    assert percentile([1,2,3,4,5], 50) == 3
    assert percentile([1,2,3,4,5], 95) == 4.8

def test_error_rate():
    assert error_rate([None, "boom", None, "x"]) == 0.5
```

- [ ] **Step 3: Run, expect ImportError**: `uv run pytest tests/test_metrics.py -v`

- [ ] **Step 4: Implement** `src/jevbench/metrics.py`

```python
from __future__ import annotations
import numpy as np

def accuracy(y_true, y_pred):
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not pairs:
        return 0.0
    return sum(t == p for t, p in pairs) / len(pairs)

def macro_f1(y_true, y_pred, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == lab and t == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == lab and t != lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != lab and t == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

def ece(y_true, probs, n_bins=10):
    rows = [(t, p) for t, p in zip(y_true, probs) if p]
    if not rows:
        return None
    conf = np.array([max(p.values()) for _, p in rows])
    correct = np.array([max(p, key=p.get) == t for t, p in rows], dtype=float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)

def percentile(values, p):
    return float(np.percentile(np.array(values, dtype=float), p))

def error_rate(errors):
    return sum(e is not None for e in errors) / len(errors) if errors else 0.0
```

- [ ] **Step 5: Run, expect PASS**; commit `feat: scaffold project and add metrics`

---

### Task 2: Datasets

**Files:**
- Create: `src/jevbench/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Produces:
  - `@dataclass Example(text: str, label: str)`
  - `@dataclass Dataset(name: str, labels: dict[str,str], examples: list[Example], train: list[Example])`  — `labels` maps label id → description.
  - `load(name: str, n: int, seed: int, train_cap: int, loader=None) -> Dataset` where `loader(hf_name, split) -> iterable of dict rows` (default uses HF `datasets.load_dataset`).
  - `DATASETS: dict[str, DatasetSpec]` with keys `sst2`, `agnews`, `banking77`.

- [ ] **Step 1: Failing test**

```python
from jevbench.datasets import load, DATASETS

def fake_loader(hf_name, split):
    # 20 rows, alternating labels 0/1, deterministic text
    return [{"sentence": f"s{i}", "text": f"t{i}", "label": i % 2} for i in range(20)]

def test_specs_have_full_label_maps():
    assert len(DATASETS["sst2"].labels) == 2
    assert len(DATASETS["agnews"].labels) == 4
    assert len(DATASETS["banking77"].labels) == 77

def test_load_samples_deterministically_and_maps_labels():
    d1 = load("sst2", n=5, seed=0, train_cap=3, loader=fake_loader)
    d2 = load("sst2", n=5, seed=0, train_cap=3, loader=fake_loader)
    assert [e.text for e in d1.examples] == [e.text for e in d2.examples]
    assert len(d1.examples) == 5 and len(d1.train) == 3
    assert set(e.label for e in d1.examples) <= {"negative", "positive"}
    assert d1.examples[0].text.startswith("s")  # sst2 uses `sentence` field
```

- [ ] **Step 2: Run, expect fail.** `uv run pytest tests/test_datasets.py -v`

- [ ] **Step 3: Implement** `src/jevbench/datasets.py`

```python
from __future__ import annotations
import random
from dataclasses import dataclass, field

@dataclass
class Example:
    text: str
    label: str

@dataclass
class DatasetSpec:
    hf_name: str
    eval_split: str
    text_field: str
    labels: dict[str, str]          # label id -> description, in HF label-index order
    train_split: str = "train"

@dataclass
class Dataset:
    name: str
    labels: dict[str, str]
    examples: list[Example]
    train: list[Example] = field(default_factory=list)

    @property
    def label_ids(self) -> list[str]:
        return list(self.labels)

BANKING77 = [ ... 77 label names from PolyAI/banking77 in index order ... ]

DATASETS = {
    "sst2": DatasetSpec("stanfordnlp/sst2", "validation", "sentence",
        {"negative": "The sentence expresses negative sentiment.",
         "positive": "The sentence expresses positive sentiment."}),
    "agnews": DatasetSpec("fancyzhx/ag_news", "test", "text",
        {"world": "World news: politics, international affairs, conflicts.",
         "sports": "Sports news: games, athletes, teams, results.",
         "business": "Business news: companies, markets, economy, finance.",
         "sci_tech": "Science and technology news: research, software, gadgets, internet."}),
    "banking77": DatasetSpec("PolyAI/banking77", "test", "text",
        {name: name.replace("_", " ") for name in BANKING77}),
}

def _hf_loader(hf_name, split):
    from datasets import load_dataset
    return load_dataset(hf_name, split=split)

def _to_examples(rows, spec):
    ids = list(spec.labels)
    return [Example(text=r[spec.text_field], label=ids[int(r["label"])]) for r in rows]

def load(name, n, seed, train_cap, loader=None) -> Dataset:
    spec = DATASETS[name]
    loader = loader or _hf_loader
    rng = random.Random(seed)
    ev = _to_examples(loader(spec.hf_name, spec.eval_split), spec)
    rng.shuffle(ev)
    tr = _to_examples(loader(spec.hf_name, spec.train_split), spec)
    rng.shuffle(tr)
    return Dataset(name, spec.labels, ev[:n], tr[:train_cap])
```

The BANKING77 list is the official 77 intents in index order (activate_my_card, age_limit, apple_pay_or_google_pay, atm_support, automatic_top_up, balance_not_updated_after_bank_transfer, ...). Copy from `https://huggingface.co/datasets/PolyAI/banking77` features.

- [ ] **Step 4: Run, expect PASS**; commit `feat: dataset loading with fixed label descriptions`

---

### Task 3: Classifier protocol, Prediction, and sqlite cache

**Files:**
- Create: `src/jevbench/classifiers/__init__.py`, `src/jevbench/classifiers/base.py`, `src/jevbench/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class Prediction:
      label: str | None
      probs: dict[str, float] | None
      latency_ms: float
      input_tokens: int | None = None
      output_tokens: int | None = None
      cost_usd: float | None = None
      raw: Any = None
      error: str | None = None
      def to_dict(self) -> dict ; @classmethod from_dict(d)

  class Classifier(Protocol):
      name: str          # e.g. "jev", "llm-cheap", "bert-ft", "bert-zs"
      model_id: str
      def prepare(self, dataset: Dataset) -> None: ...
      async def predict_batch(self, texts: list[str], labels: dict[str,str], parallelism: int) -> list[Prediction]: ...

  class Cache:
      def __init__(self, path: str = "cache.sqlite")
      def key(self, classifier: str, model_id: str, dataset: str, text: str, labels: dict) -> str
      def get(self, key) -> Prediction | None
      def put(self, key, pred: Prediction) -> None
  ```

- [ ] **Step 1: Failing test**

```python
from jevbench.cache import Cache
from jevbench.classifiers.base import Prediction

def test_cache_roundtrip(tmp_path):
    c = Cache(str(tmp_path / "c.sqlite"))
    k = c.key("jev", "typesafe/jev-1.13", "sst2", "hello", {"a": "x"})
    assert c.get(k) is None
    p = Prediction(label="a", probs={"a": 0.9, "b": 0.1}, latency_ms=12.5, cost_usd=1e-6, raw={"x": 1})
    c.put(k, p)
    got = c.get(k)
    assert got.label == "a" and got.probs["a"] == 0.9 and got.raw == {"x": 1}
    assert c.key("jev", "m", "sst2", "hello", {"a": "x"}) != c.key("jev", "m", "sst2", "hello", {"a": "y"})
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement** base.py (dataclass with `to_dict`/`from_dict` via `dataclasses.asdict`) and cache.py (sqlite3 table `cache(key TEXT PRIMARY KEY, value TEXT)`, key = sha256 of `json.dumps([classifier, model_id, dataset, text, labels], sort_keys=True)`).

- [ ] **Step 4: Run, expect PASS**; commit `feat: Prediction, Classifier protocol, sqlite cache`

---

### Task 4: JEV adapter (Decisions API)

**Files:**
- Create: `src/jevbench/classifiers/jev.py`, `src/jevbench/http.py`
- Test: `tests/test_jev.py`, `tests/fixtures/jev_choice.json`

**Interfaces:**
- Consumes: `Prediction`, `Classifier`
- Produces:
  - `http.py`: `async def post_json(client, url, body, headers, retries=3) -> dict` — retries on 429/5xx with backoff 1s,2s,4s; raises `httpx.HTTPStatusError` otherwise.
  - `JevClassifier(model_id: str, api_key: str, client: httpx.AsyncClient | None = None)`; `name = "jev"`.
  - `build_questions(labels: dict[str,str]) -> dict` → `{"label": {"type": "choice", "instructions": "...", "criteria": labels}}`

- [ ] **Step 1: Fixture** `tests/fixtures/jev_choice.json` (shape from the live example):

```json
{"answers": {"label": {"type": "choice", "choice": "positive",
  "probabilities": {"negative": 0.08, "positive": 0.92}, "confidence": 0.84}},
 "usage": {"input_tokens": 120, "output_tokens": 20, "cost": 0.00000504},
 "provider": "TypeSafe", "model": "typesafe/jev-1.13-20260917"}
```

- [ ] **Step 2: Failing test**

```python
import json, httpx, pytest
from pathlib import Path
from jevbench.classifiers.jev import JevClassifier, build_questions

FIX = json.loads(Path("tests/fixtures/jev_choice.json").read_text())

def make_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))

async def test_jev_parses_choice_and_usage():
    seen = {}
    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json=FIX)
    clf = JevClassifier("typesafe/jev-1.13", "KEY", client=make_client(handler))
    labels = {"negative": "neg", "positive": "pos"}
    [p] = await clf.predict_batch(["great movie"], labels, parallelism=1)
    assert p.label == "positive" and p.probs["positive"] == 0.92
    assert p.input_tokens == 120 and p.cost_usd == pytest.approx(5.04e-6)
    assert p.error is None and p.latency_ms >= 0
    assert seen["auth"] == "Bearer KEY"
    assert seen["body"]["model"] == "typesafe/jev-1.13"
    assert seen["body"]["state"] == "great movie"
    assert seen["body"]["questions"]["label"]["criteria"] == labels

async def test_jev_records_error_on_500_after_retries():
    def handler(req):
        return httpx.Response(500, json={"error": "boom"})
    clf = JevClassifier("m", "KEY", client=make_client(handler), backoff=0)
    [p] = await clf.predict_batch(["x"], {"a": "a"}, parallelism=1)
    assert p.label is None and "500" in p.error

def test_build_questions():
    q = build_questions({"a": "desc a"})
    assert q["label"]["type"] == "choice" and q["label"]["criteria"] == {"a": "desc a"}
```

- [ ] **Step 3: Run, expect fail.**

- [ ] **Step 4: Implement**

`src/jevbench/http.py`:
```python
import asyncio, httpx
RETRY = {429, 500, 502, 503, 504}
async def post_json(client, url, body, headers, retries=3, backoff=1.0):
    for attempt in range(retries + 1):
        r = await client.post(url, json=body, headers=headers, timeout=60)
        if r.status_code in RETRY and attempt < retries:
            await asyncio.sleep(backoff * (2 ** attempt))
            continue
        r.raise_for_status()
        return r.json()
```

`src/jevbench/classifiers/jev.py`:
```python
import asyncio, time, httpx
from .base import Prediction
from ..http import post_json

URL = "https://openrouter.ai/api/alpha/decisions"

def build_questions(labels):
    return {"label": {"type": "choice",
        "instructions": "Which category best describes the text? Pick exactly one.",
        "criteria": labels}}

class JevClassifier:
    name = "jev"
    def __init__(self, model_id, api_key, client=None, backoff=1.0):
        self.model_id, self.api_key, self.backoff = model_id, api_key, backoff
        self.client = client or httpx.AsyncClient()
    def prepare(self, dataset): pass
    async def predict_one(self, text, labels):
        body = {"model": self.model_id, "state": text, "questions": build_questions(labels)}
        t0 = time.perf_counter()
        try:
            data = await post_json(self.client, URL, body,
                {"Authorization": f"Bearer {self.api_key}"}, backoff=self.backoff)
        except Exception as e:
            return Prediction(None, None, (time.perf_counter()-t0)*1000, error=repr(e))
        ms = (time.perf_counter()-t0)*1000
        ans = data["answers"]["label"]; u = data.get("usage", {})
        return Prediction(ans["choice"], ans.get("probabilities"), ms,
            u.get("input_tokens"), u.get("output_tokens"), u.get("cost"), raw=data)
    async def predict_batch(self, texts, labels, parallelism):
        sem = asyncio.Semaphore(parallelism)
        async def one(t):
            async with sem: return await self.predict_one(t, labels)
        return await asyncio.gather(*(one(t) for t in texts))
```

- [ ] **Step 5: Run, expect PASS**; commit `feat: JEV Decisions API adapter`

---

### Task 5: LLM adapter (OpenRouter chat, structured output)

**Files:**
- Create: `src/jevbench/classifiers/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `LLMClassifier(name: str, model_id: str, api_key: str, reasoning_effort: str | None = "low", client=None, backoff=1.0)`; `build_messages(text, labels) -> list[dict]`; `build_response_format(labels) -> dict`.

- [ ] **Step 1: Failing test**

```python
import json, httpx, pytest
from jevbench.classifiers.llm import LLMClassifier

def resp(content, cost=0.0001):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 6, "cost": cost}}

def make_client(handler): return httpx.AsyncClient(transport=httpx.MockTransport(handler))

async def test_llm_parses_label_and_sends_schema():
    seen = {}
    def handler(req):
        seen["body"] = json.loads(req.content); return httpx.Response(200, json=resp('{"label":"sports"}'))
    clf = LLMClassifier("llm-cheap", "openai/gpt-5-mini", "KEY", client=make_client(handler))
    labels = {"sports": "sports desc", "world": "world desc"}
    [p] = await clf.predict_batch(["Lakers win"], labels, parallelism=1)
    assert p.label == "sports" and p.probs is None and p.error is None
    assert p.input_tokens == 80 and p.cost_usd == pytest.approx(1e-4)
    b = seen["body"]
    assert b["model"] == "openai/gpt-5-mini"
    assert b["usage"] == {"include": True}
    assert b["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"] == ["sports", "world"]
    assert b["reasoning"] == {"effort": "low"}
    assert "sports desc" in b["messages"][0]["content"]

async def test_llm_off_enum_output_is_error():
    def handler(req): return httpx.Response(200, json=resp('{"label":"cricket"}'))
    clf = LLMClassifier("llm-cheap", "m", "KEY", client=make_client(handler))
    [p] = await clf.predict_batch(["x"], {"sports": "s"}, parallelism=1)
    assert p.label is None and "invalid label" in p.error

async def test_llm_unparseable_is_error():
    def handler(req): return httpx.Response(200, json=resp('sports'))
    clf = LLMClassifier("llm-cheap", "m", "KEY", client=make_client(handler))
    [p] = await clf.predict_batch(["x"], {"sports": "s"}, parallelism=1)
    assert p.label is None and "parse" in p.error

async def test_llm_retries_without_reasoning_on_400():
    calls = []
    def handler(req):
        body = json.loads(req.content); calls.append("reasoning" in body)
        if "reasoning" in body: return httpx.Response(400, json={"error": {"message": "reasoning not supported"}})
        return httpx.Response(200, json=resp('{"label":"sports"}'))
    clf = LLMClassifier("llm-cheap", "m", "KEY", client=make_client(handler), backoff=0)
    [p] = await clf.predict_batch(["x"], {"sports": "s"}, parallelism=1)
    assert p.label == "sports" and calls == [True, False]
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement** `llm.py`

```python
import asyncio, json, time, httpx
from .base import Prediction
from ..http import post_json
URL = "https://openrouter.ai/api/v1/chat/completions"

def build_messages(text, labels):
    lines = "\n".join(f"- {k}: {v}" for k, v in labels.items())
    system = ("You are a text classifier. Classify the user's text into exactly one of these labels.\n"
              f"Labels:\n{lines}\n\nRespond with JSON only: {{\"label\": \"<label id>\"}}.")
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]

def build_response_format(labels):
    return {"type": "json_schema", "json_schema": {"name": "classification", "strict": True,
        "schema": {"type": "object", "properties": {"label": {"type": "string", "enum": list(labels)}},
                   "required": ["label"], "additionalProperties": False}}}

class LLMClassifier:
    def __init__(self, name, model_id, api_key, reasoning_effort="low", client=None, backoff=1.0):
        self.name, self.model_id, self.api_key = name, model_id, api_key
        self.reasoning_effort, self.backoff = reasoning_effort, backoff
        self.client = client or httpx.AsyncClient()
        self._reasoning_ok = True
    def prepare(self, dataset): pass
    def _body(self, text, labels):
        b = {"model": self.model_id, "messages": build_messages(text, labels),
             "response_format": build_response_format(labels), "usage": {"include": True},
             "max_tokens": 200}
        if self.reasoning_effort and self._reasoning_ok:
            b["reasoning"] = {"effort": self.reasoning_effort}
        return b
    async def predict_one(self, text, labels):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.perf_counter()
        try:
            try:
                data = await post_json(self.client, URL, self._body(text, labels), headers, backoff=self.backoff)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400 and self._reasoning_ok:
                    self._reasoning_ok = False
                    data = await post_json(self.client, URL, self._body(text, labels), headers, backoff=self.backoff)
                else: raise
        except Exception as e:
            return Prediction(None, None, (time.perf_counter()-t0)*1000, error=repr(e))
        ms = (time.perf_counter()-t0)*1000
        u = data.get("usage", {})
        base = dict(probs=None, latency_ms=ms, input_tokens=u.get("prompt_tokens"),
                    output_tokens=u.get("completion_tokens"), cost_usd=u.get("cost"), raw=data)
        content = data["choices"][0]["message"].get("content") or ""
        try:
            label = json.loads(content)["label"]
        except Exception:
            return Prediction(None, error=f"parse error: {content[:100]!r}", **base)
        if label not in labels:
            return Prediction(None, error=f"invalid label: {label!r}", **base)
        return Prediction(label, **base)
    async def predict_batch(self, texts, labels, parallelism):
        sem = asyncio.Semaphore(parallelism)
        async def one(t):
            async with sem: return await self.predict_one(t, labels)
        return await asyncio.gather(*(one(t) for t in texts))
```

- [ ] **Step 4: Run, expect PASS**; commit `feat: OpenRouter LLM adapter with JSON-schema output`

---

### Task 6: BERT zero-shot adapter

**Files:**
- Create: `src/jevbench/classifiers/bert_zeroshot.py`
- Test: `tests/test_bert_zeroshot.py`

**Interfaces:**
- Produces: `ZeroShotClassifier(model_id: str, pipeline_factory=None)`; `name = "bert-zs"`. `pipeline_factory()` returns a callable `pipe(texts: list[str], candidate_labels: list[str], batch_size: int) -> list[dict]` where each dict has `labels` (descriptions sorted by score desc) and `scores`.

- [ ] **Step 1: Failing test**

```python
from jevbench.classifiers.bert_zeroshot import ZeroShotClassifier

class FakePipe:
    def __call__(self, texts, candidate_labels, batch_size):
        # always prefer the first candidate
        return [{"labels": list(candidate_labels), "scores": [0.7, 0.3]} for _ in texts]

async def test_zero_shot_maps_descriptions_back_to_ids():
    clf = ZeroShotClassifier("facebook/bart-large-mnli", pipeline_factory=lambda: FakePipe())
    labels = {"neg": "negative sentiment", "pos": "positive sentiment"}
    preds = await clf.predict_batch(["a", "b"], labels, parallelism=2)
    assert [p.label for p in preds] == ["neg", "neg"]
    assert preds[0].probs == {"neg": 0.7, "pos": 0.3}
    assert preds[0].cost_usd == 0.0 and preds[0].latency_ms >= 0
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement**

```python
import time
from .base import Prediction

def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"

class ZeroShotClassifier:
    name = "bert-zs"
    def __init__(self, model_id, pipeline_factory=None):
        self.model_id = model_id
        self._factory = pipeline_factory or self._default_factory
        self._pipe = None
    def _default_factory(self):
        from transformers import pipeline
        return pipeline("zero-shot-classification", model=self.model_id, device=_device())
    def prepare(self, dataset):
        if self._pipe is None: self._pipe = self._factory()
    async def predict_batch(self, texts, labels, parallelism):
        if self._pipe is None: self._pipe = self._factory()
        desc_to_id = {v: k for k, v in labels.items()}
        cands = list(labels.values())
        out = []
        for i in range(0, len(texts), parallelism):
            chunk = texts[i:i+parallelism]
            t0 = time.perf_counter()
            res = self._pipe(chunk, candidate_labels=cands, batch_size=parallelism)
            per = (time.perf_counter()-t0)*1000 / len(chunk)
            if isinstance(res, dict): res = [res]
            for r in res:
                probs = {desc_to_id[l]: float(s) for l, s in zip(r["labels"], r["scores"])}
                out.append(Prediction(max(probs, key=probs.get), probs, per, 0, 0, 0.0))
        return out
```

Note: per-example latency in a batch is wall-clock / batch size; with `parallelism=1` it's true single-example latency.

- [ ] **Step 4: Run, expect PASS**; commit `feat: zero-shot NLI classifier adapter`

---

### Task 7: BERT fine-tuned adapter

**Files:**
- Create: `src/jevbench/classifiers/bert_finetune.py`
- Test: `tests/test_bert_finetune.py`

**Interfaces:**
- Produces: `FineTunedClassifier(base_model: str, models_dir: str, epochs, lr, max_len, batch_size, trainer=None, predictor=None)`; `name = "bert-ft"`. `prepare(dataset)` trains if `models_dir/distilbert-{dataset.name}` is missing, then loads. Injected `trainer(base_model, train_examples, label_ids, out_dir, cfg) -> None` and `predictor(model_dir) -> callable(texts, batch_size) -> list[list[float]]` (softmax rows in label-id order) so tests avoid torch.
- Records training wall-clock in `self.train_seconds`.

- [ ] **Step 1: Failing test**

```python
from jevbench.datasets import Dataset, Example
from jevbench.classifiers.bert_finetune import FineTunedClassifier

async def test_trains_once_then_loads(tmp_path):
    calls = []
    def trainer(base, train, ids, out, cfg):
        calls.append("train"); out.mkdir(parents=True); (out / "config.json").write_text("{}")
    def predictor(model_dir):
        return lambda texts, batch_size: [[0.2, 0.8] for _ in texts]
    ds = Dataset("sst2", {"neg": "n", "pos": "p"}, [Example("x", "pos")], [Example("t", "pos")])
    clf = FineTunedClassifier("distilbert-base-uncased", str(tmp_path), 1, 5e-5, 128, 8,
                              trainer=trainer, predictor=predictor)
    clf.prepare(ds); clf.prepare(ds)
    assert calls == ["train"]
    [p] = await clf.predict_batch(["x"], ds.labels, parallelism=1)
    assert p.label == "pos" and p.probs == {"neg": 0.2, "pos": 0.8}
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement** with a real default `trainer` (manual torch loop: AutoTokenizer + AutoModelForSequenceClassification, AdamW, linear schedule, MPS if available, saves with `save_pretrained`) and default `predictor` (loads model, tokenizes in batches, softmax). Label order = `dataset.label_ids`; store `id2label` in the model config.

- [ ] **Step 4: Run, expect PASS**; commit `feat: fine-tuned DistilBERT adapter with train-or-load`

---

### Task 8: Runner CLI

**Files:**
- Create: `src/jevbench/config.py`, `src/jevbench/runner.py`, `src/jevbench/run.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- `config.py`: `load_config(path="config.toml") -> dict` (tomllib) ; `api_key() -> str` (dotenv + env, raises `RuntimeError("OPENROUTER_API_KEY not set")`).
- `runner.py`:
  - `async def evaluate(clf, dataset, cache, parallelism_api: int, local_batch: int, skip_throughput: bool, use_cache: bool) -> dict` returning `{"classifier", "model_id", "dataset", "n", "predictions": [...], "metrics": {...}, "train_seconds", "timestamp", "hardware"}`.
  - Latency pass: `predict_batch(texts, labels, parallelism=1)`; if `use_cache` and every text is cached, load from cache and set `metrics["latency_from_cache"]=True`. Otherwise call, then `cache.put` each non-error result.
  - Throughput pass: `predict_batch(texts, labels, parallelism=P)` where P = `parallelism_api` for `jev`/`llm-*`, `local_batch` for `bert-*`; `throughput_ex_s = n / wall_seconds`. Skipped when `skip_throughput`.
  - Metrics dict keys: `accuracy, macro_f1, ece, error_rate, latency_p50_ms, latency_p95_ms, throughput_ex_s, cost_per_1k_usd, avg_input_tokens, avg_output_tokens`.
  - `build_classifiers(names: list[str], cfg) -> list[Classifier]` for names in `jev, llm-cheap, llm-frontier, bert-ft, bert-zs`.
- `run.py`: argparse with subcommands `run` (`--datasets`, `--classifiers`, `--n`, `--seed`, `--skip-throughput`, `--no-cache`, `--out results`) and `report` (`--run-dir`). Writes `results/<YYYYmmdd-HHMMSS>/<clf>__<ds>.json`, then calls `report.write_summary(run_dir)`.

- [ ] **Step 1: Failing test** with a `FakeClassifier` that returns fixed predictions and counts calls; assert metrics computed, throughput pass invoked once, cache populated, second `evaluate` with `use_cache=True` makes zero latency-pass calls and sets `latency_from_cache`.

```python
from jevbench.runner import evaluate
from jevbench.cache import Cache
from jevbench.datasets import Dataset, Example
from jevbench.classifiers.base import Prediction

class Fake:
    name = "jev"; model_id = "m"
    def __init__(self): self.calls = []
    def prepare(self, ds): pass
    async def predict_batch(self, texts, labels, parallelism):
        self.calls.append(parallelism)
        return [Prediction("a" if t == "x" else "b", {"a": .9, "b": .1}, 10.0, 5, 1, 1e-6) for t in texts]

async def test_evaluate_metrics_and_cache(tmp_path):
    ds = Dataset("sst2", {"a": "A", "b": "B"}, [Example("x", "a"), Example("y", "a")])
    cache = Cache(str(tmp_path / "c.sqlite")); clf = Fake()
    r = await evaluate(clf, ds, cache, parallelism_api=4, local_batch=8, skip_throughput=False, use_cache=True)
    assert r["metrics"]["accuracy"] == 0.5 and r["metrics"]["latency_p50_ms"] == 10.0
    assert r["metrics"]["cost_per_1k_usd"] == 0.001 and r["metrics"]["throughput_ex_s"] > 0
    assert clf.calls == [1, 4]
    clf2 = Fake()
    r2 = await evaluate(clf2, ds, cache, 4, 8, skip_throughput=True, use_cache=True)
    assert clf2.calls == [] and r2["metrics"]["latency_from_cache"] is True
```

- [ ] **Step 2: Run, expect fail.** — **Step 3: Implement.** — **Step 4: PASS; commit** `feat: benchmark runner and CLI`

---

### Task 9: Report

**Files:**
- Create: `src/jevbench/report.py`
- Test: `tests/test_report.py`

**Interfaces:** `write_summary(run_dir: Path) -> Path` reads every `*.json`, groups by dataset, writes `summary.md` with one table per dataset: columns `Classifier | Model | Acc | Macro-F1 | ECE | Err% | p50 ms | p95 ms | ex/s | $/1k`. Missing values render as `–`.

- [ ] **Step 1: Failing test**: write two result JSONs to `tmp_path`, call `write_summary`, assert the file contains `## sst2`, a row starting with `| jev |`, and `–` for a None ECE.

- [ ] **Step 2-4: fail → implement → pass; commit** `feat: markdown summary report`

---

### Task 10: README and live smoke run

**Files:**
- Create: `README.md`
- Modify: `run.py` — add `--smoke` (sets n=5, skip throughput)

- [ ] **Step 1:** README: what it is, setup (`uv sync`, `.env`), commands, how to read results, budget note, caveats (fine-tuned BERT sees training data; LLMs return no probabilities so ECE is blank; local timings are this Mac's MPS).
- [ ] **Step 2:** `uv run jevbench run --smoke --classifiers jev llm-cheap bert-zs --datasets sst2` and confirm a summary renders with real numbers. Report the output verbatim to the user.
- [ ] **Step 3:** Commit `docs: README and smoke flag`.

---

## Self-review

- Spec coverage: datasets ✔ (T2), five classifiers ✔ (T4–T7; llm-cheap/frontier are two `LLMClassifier` instances in T8), metrics ✔ (T1), latency/throughput/cost ✔ (T8), cache ✔ (T3), output JSON + summary.md ✔ (T8–T9), config.toml + .env ✔ (T1, T8), retries/error handling ✔ (T4 `http.py`, `error` field), tests without network ✔, `--smoke` ✔ (T10).
- Deviation from spec: `temperature=0` is not sent because the gpt-5 family rejects it on OpenRouter; JSON-schema enum plus low reasoning effort is the determinism lever instead. Noted in README.
- Type consistency: `predict_batch(texts, labels, parallelism)` everywhere; `Prediction` positional order `(label, probs, latency_ms, input_tokens, output_tokens, cost_usd, raw, error)`.
