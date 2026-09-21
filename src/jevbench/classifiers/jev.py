"""TypeSafe JEV via OpenRouter's Decisions API (not chat completions)."""

from __future__ import annotations

import asyncio
import time

import httpx

from ..http import post_json
from .base import Prediction

URL = "https://openrouter.ai/api/alpha/decisions"


def build_questions(labels: dict[str, str]) -> dict:
    return {
        "label": {
            "type": "choice",
            "instructions": "Which category best describes the text? Pick exactly one.",
            "criteria": labels,
        }
    }


class JevClassifier:
    name = "jev"

    def __init__(self, model_id: str, api_key: str, client: httpx.AsyncClient | None = None, backoff: float = 1.0):
        self.model_id = model_id
        self.api_key = api_key
        self.backoff = backoff
        self.client = client or httpx.AsyncClient()

    def prepare(self, dataset) -> None:
        pass

    async def predict_one(self, text: str, labels: dict[str, str]) -> Prediction:
        body = {"model": self.model_id, "state": text, "questions": build_questions(labels)}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.perf_counter()
        try:
            data = await post_json(self.client, URL, body, headers, backoff=self.backoff)
        except Exception as e:  # noqa: BLE001 - recorded, not raised
            return Prediction(None, None, (time.perf_counter() - t0) * 1000, error=repr(e))
        ms = (time.perf_counter() - t0) * 1000
        try:
            ans = data["answers"]["label"]
            usage = data.get("usage", {})
            return Prediction(
                ans["choice"], ans.get("probabilities"), ms,
                usage.get("input_tokens"), usage.get("output_tokens"), usage.get("cost"),
                raw=data,
            )
        except (KeyError, TypeError) as e:
            return Prediction(None, None, ms, raw=data, error=f"unexpected response: {e!r}")

    async def predict_batch(self, texts: list[str], labels: dict[str, str], parallelism: int) -> list[Prediction]:
        sem = asyncio.Semaphore(parallelism)

        async def one(t):
            async with sem:
                return await self.predict_one(t, labels)

        return list(await asyncio.gather(*(one(t) for t in texts)))
