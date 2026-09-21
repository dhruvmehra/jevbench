"""Laya (Convai Innovations): open-weights System One model, run locally.

Same state + typed-questions interface as JEV, so we reuse JEV's question builder
to guarantee both models see identical criteria.
"""

from __future__ import annotations

import time
from typing import Callable

from .base import Prediction
from .jev import build_questions


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class LayaClassifier:
    name = "laya"
    supports_batch = False  # Agent.predict takes one state per call

    def __init__(self, model_id: str = "convaiinnovations/laya", loader: Callable | None = None):
        self.model_id = model_id
        self._loader = loader or self._default_loader
        self._agent = None
        self.load_seconds: float | None = None

    def _default_loader(self):
        import laya

        return laya.load(self.model_id, device=device())

    def prepare(self, dataset) -> None:
        if self._agent is None:
            t0 = time.perf_counter()
            self._agent = self._loader()
            self.load_seconds = time.perf_counter() - t0
            # warm up kernels so the first timed example is not an outlier
            self._agent.predict("warm up", build_questions({"a": "a", "b": "b"}))

    async def predict_batch(self, texts: list[str], labels: dict[str, str], parallelism: int) -> list[Prediction]:
        if self._agent is None:
            self.prepare(None)
        questions = build_questions(labels)
        out: list[Prediction] = []
        for text in texts:
            t0 = time.perf_counter()
            try:
                res = self._agent.predict(text, questions)
            except Exception as e:  # noqa: BLE001
                out.append(Prediction(None, None, (time.perf_counter() - t0) * 1000, error=repr(e)))
                continue
            ms = (time.perf_counter() - t0) * 1000
            try:
                ans = res["answers"]["label"]
                usage = res.get("usage") or {}
                probs = {k: float(v) for k, v in ans["probabilities"].items()}
                out.append(Prediction(ans["choice"], probs, ms, usage.get("input_tokens"), 0, 0.0, raw=res))
            except (KeyError, TypeError) as e:
                out.append(Prediction(None, None, ms, raw=res, error=f"unexpected response: {e!r}"))
        return out
