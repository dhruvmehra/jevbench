"""Evaluate one classifier on one dataset: sequential latency/accuracy pass + concurrent throughput pass."""

from __future__ import annotations

import datetime as dt
import platform
import time

from . import metrics as M
from .cache import Cache
from .classifiers.base import Classifier, Prediction
from .datasets import Dataset

API_CLASSIFIERS = ("jev", "llm-cheap", "llm-frontier")


def is_api(clf: Classifier) -> bool:
    return clf.name.startswith(("jev", "llm"))


def group_classifiers(clfs: list[Classifier]) -> tuple[list[Classifier], list[Classifier]]:
    """API classifiers may run concurrently with each other; local ones must run one at a time."""
    api = [c for c in clfs if is_api(c)]
    local = [c for c in clfs if not is_api(c)]
    return api, local


def hardware() -> str:
    return f"{platform.system()} {platform.machine()}, {platform.processor() or 'unknown cpu'}"


def compute_metrics(dataset: Dataset, preds: list[Prediction]) -> dict:
    y_true = [e.label for e in dataset.examples]
    y_pred = [p.label for p in preds]
    ok = [p for p in preds if p.error is None]
    costs = [p.cost_usd for p in preds if p.cost_usd is not None]
    n = len(preds)
    return {
        "accuracy": M.accuracy(y_true, y_pred),
        "macro_f1": M.macro_f1(y_true, y_pred, dataset.label_ids),
        "ece": M.ece(y_true, [p.probs for p in preds]),
        "error_rate": M.error_rate([p.error for p in preds]),
        "latency_p50_ms": M.percentile([p.latency_ms for p in ok], 50) if ok else None,
        "latency_p95_ms": M.percentile([p.latency_ms for p in ok], 95) if ok else None,
        "cost_per_1k_usd": (sum(costs) / n * 1000) if costs and n else None,
        "avg_input_tokens": (sum(p.input_tokens or 0 for p in preds) / n) if n else None,
        "avg_output_tokens": (sum(p.output_tokens or 0 for p in preds) / n) if n else None,
        "throughput_ex_s": None,
        "latency_from_cache": False,
    }


async def evaluate(
    clf: Classifier,
    dataset: Dataset,
    cache: Cache,
    parallelism_api: int,
    local_batch: int,
    skip_throughput: bool,
    use_cache: bool,
) -> dict:
    texts = [e.text for e in dataset.examples]
    keys = [cache.key(clf.name, clf.model_id, dataset.name, t, dataset.labels) for t in texts]

    clf.prepare(dataset)
    if not is_api(clf) and texts:
        # warm up kernels / weights so the first timed example is not an outlier
        await clf.predict_batch(texts[:1], dataset.labels, parallelism=1)

    cached = [cache.get(k) for k in keys] if use_cache else [None] * len(keys)
    from_cache = use_cache and all(c is not None for c in cached)
    if from_cache:
        preds = cached  # type: ignore[assignment]
    else:
        preds = await clf.predict_batch(texts, dataset.labels, parallelism=1)
        for k, p in zip(keys, preds):
            if p.error is None:
                cache.put(k, p)

    m = compute_metrics(dataset, preds)
    m["latency_from_cache"] = from_cache
    if from_cache:
        m["latency_p50_ms"] = m["latency_p95_ms"] = None

    if not skip_throughput:
        par = parallelism_api if is_api(clf) else local_batch
        t0 = time.perf_counter()
        tp_preds = await clf.predict_batch(texts, dataset.labels, parallelism=par)
        wall = time.perf_counter() - t0
        m["throughput_ex_s"] = len(texts) / wall if wall > 0 else None
        m["throughput_parallelism"] = par
        m["throughput_error_rate"] = M.error_rate([p.error for p in tp_preds])
        m["throughput_cost_usd"] = sum(p.cost_usd or 0 for p in tp_preds)

    return {
        "classifier": clf.name,
        "model_id": clf.model_id,
        "dataset": dataset.name,
        "n": len(texts),
        "n_labels": len(dataset.labels),
        "predictions": [p.to_dict() for p in preds],
        "metrics": m,
        "train_seconds": getattr(clf, "train_seconds", None),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware": hardware(),
    }


def build_classifiers(names: list[str], cfg: dict, key: str | None) -> list[Classifier]:
    out: list[Classifier] = []
    for name in names:
        if name == "jev":
            from .classifiers.jev import JevClassifier

            out.append(JevClassifier(cfg["jev"]["model"], key or ""))
        elif name.startswith("llm-"):
            from .classifiers.llm import LLMClassifier

            tier = cfg["llm"][name.removeprefix("llm-")]
            out.append(LLMClassifier(name, tier["model"], key or "", tier.get("reasoning_effort")))
        elif name == "bert-zs":
            from .classifiers.bert_zeroshot import ZeroShotClassifier

            out.append(ZeroShotClassifier(cfg["bert"]["zeroshot_model"]))
        elif name == "bert-ft":
            from .classifiers.bert_finetune import FineTunedClassifier

            b = cfg["bert"]
            out.append(FineTunedClassifier(b["finetune_base"], "models", b["epochs"], b["lr"], b["max_len"], b["batch_size"]))
        else:
            raise ValueError(f"unknown classifier {name!r}")
    return out
