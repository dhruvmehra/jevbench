"""Zero-shot NLI classifier (default facebook/bart-large-mnli) via transformers pipeline."""

from __future__ import annotations

import time
from typing import Callable

from .base import Prediction


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class ZeroShotClassifier:
    name = "bert-zs"

    def __init__(self, model_id: str, pipeline_factory: Callable | None = None):
        self.model_id = model_id
        self._factory = pipeline_factory or self._default_factory
        self._pipe = None

    def _default_factory(self):
        from transformers import pipeline

        return pipeline("zero-shot-classification", model=self.model_id, device=device())

    def prepare(self, dataset) -> None:
        if self._pipe is None:
            self._pipe = self._factory()

    async def predict_batch(self, texts: list[str], labels: dict[str, str], parallelism: int) -> list[Prediction]:
        if self._pipe is None:
            self._pipe = self._factory()
        desc_to_id = {v: k for k, v in labels.items()}
        cands = list(labels.values())
        out: list[Prediction] = []
        # `parallelism` acts as the batch size for local models.
        for i in range(0, len(texts), parallelism):
            chunk = texts[i : i + parallelism]
            t0 = time.perf_counter()
            res = self._pipe(chunk, candidate_labels=cands, batch_size=parallelism)
            per_ms = (time.perf_counter() - t0) * 1000 / len(chunk)
            if isinstance(res, dict):
                res = [res]
            for r in res:
                probs = {desc_to_id[l]: float(s) for l, s in zip(r["labels"], r["scores"])}
                out.append(Prediction(max(probs, key=probs.get), probs, per_ms, 0, 0, 0.0))
        return out
