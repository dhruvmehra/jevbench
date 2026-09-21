"""LLM classifier via OpenRouter chat completions with JSON-schema structured output."""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from ..http import post_json
from .base import Prediction

URL = "https://openrouter.ai/api/v1/chat/completions"


def build_messages(text: str, labels: dict[str, str]) -> list[dict]:
    lines = "\n".join(f"- {k}: {v}" for k, v in labels.items())
    system = (
        "You are a text classifier. Classify the user's text into exactly one of these labels.\n"
        f"Labels:\n{lines}\n\n"
        'Respond with JSON only: {"label": "<label id>"}.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def build_response_format(labels: dict[str, str]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"label": {"type": "string", "enum": list(labels)}},
                "required": ["label"],
                "additionalProperties": False,
            },
        },
    }


class LLMClassifier:
    def __init__(
        self,
        name: str,
        model_id: str,
        api_key: str,
        reasoning_effort: str | None = "low",
        client: httpx.AsyncClient | None = None,
        backoff: float = 1.0,
    ):
        self.name = name
        self.model_id = model_id
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.backoff = backoff
        self.client = client or httpx.AsyncClient()
        self._reasoning_ok = True

    def prepare(self, dataset) -> None:
        pass

    def _body(self, text: str, labels: dict[str, str]) -> dict:
        b = {
            "model": self.model_id,
            "messages": build_messages(text, labels),
            "response_format": build_response_format(labels),
            "usage": {"include": True},
            "max_tokens": 200,
        }
        if self.reasoning_effort and self._reasoning_ok:
            b["reasoning"] = {"effort": self.reasoning_effort}
        return b

    async def predict_one(self, text: str, labels: dict[str, str]) -> Prediction:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.perf_counter()
        try:
            try:
                data = await post_json(self.client, URL, self._body(text, labels), headers, backoff=self.backoff)
            except httpx.HTTPStatusError as e:
                # Some models reject the reasoning parameter; drop it once and retry.
                if e.response.status_code == 400 and self._reasoning_ok and "reasoning" in self._body(text, labels):
                    self._reasoning_ok = False
                    data = await post_json(self.client, URL, self._body(text, labels), headers, backoff=self.backoff)
                else:
                    raise
        except Exception as e:  # noqa: BLE001
            return Prediction(None, None, (time.perf_counter() - t0) * 1000, error=repr(e))
        ms = (time.perf_counter() - t0) * 1000
        usage = data.get("usage") or {}
        base = dict(
            probs=None, latency_ms=ms,
            input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
            cost_usd=usage.get("cost"), raw=data,
        )
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return Prediction(None, error="unexpected response shape", **base)
        try:
            label = json.loads(content)["label"]
        except Exception:  # noqa: BLE001
            return Prediction(None, error=f"parse error: {content[:100]!r}", **base)
        if label not in labels:
            return Prediction(None, error=f"invalid label: {label!r}", **base)
        return Prediction(label, **base)

    async def predict_batch(self, texts: list[str], labels: dict[str, str], parallelism: int) -> list[Prediction]:
        sem = asyncio.Semaphore(parallelism)

        async def one(t):
            async with sem:
                return await self.predict_one(t, labels)

        return list(await asyncio.gather(*(one(t) for t in texts)))
