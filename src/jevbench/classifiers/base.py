from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ..datasets import Dataset


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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Prediction":
        return cls(**d)


class Classifier(Protocol):
    name: str
    model_id: str

    def prepare(self, dataset: Dataset) -> None: ...

    async def predict_batch(
        self, texts: list[str], labels: dict[str, str], parallelism: int
    ) -> list[Prediction]: ...
