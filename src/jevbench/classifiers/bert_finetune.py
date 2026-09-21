"""Fine-tuned encoder classifier (default distilbert-base-uncased), trained once per dataset.

The training and prediction routines are injectable so tests never touch torch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..datasets import Dataset, Example
from .base import Prediction


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    max_len: int
    batch_size: int


def _device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def default_trainer(base_model: str, train: list[Example], label_ids: list[str], out_dir: Path, cfg: TrainConfig) -> None:
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(0)
    dev = _device()
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label_ids),
        id2label={i: l for i, l in enumerate(label_ids)},
        label2id={l: i for i, l in enumerate(label_ids)},
    ).to(dev)
    idx = {l: i for i, l in enumerate(label_ids)}
    texts = [e.text for e in train]
    ys = torch.tensor([idx[e.label] for e in train])
    enc = tok(texts, truncation=True, max_length=cfg.max_len, padding=True, return_tensors="pt")
    n = len(texts)
    steps = cfg.epochs * ((n + cfg.batch_size - 1) // cfg.batch_size)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            b = perm[i : i + cfg.batch_size]
            out = model(
                input_ids=enc["input_ids"][b].to(dev),
                attention_mask=enc["attention_mask"][b].to(dev),
                labels=ys[b].to(dev),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += out.loss.item() * len(b)
        print(f"  [bert-ft] epoch {epoch + 1}/{cfg.epochs} loss={total / n:.4f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "label_ids.json").write_text(json.dumps(label_ids))


def default_predictor(model_dir: Path) -> Callable[[list[str], int], list[list[float]]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    max_len = 128

    def predict(texts: list[str], batch_size: int) -> list[list[float]]:
        rows: list[list[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i : i + batch_size], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
                logits = model(**enc).logits
                rows.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return rows

    return predict


class FineTunedClassifier:
    name = "bert-ft"

    def __init__(
        self,
        base_model: str,
        models_dir: str,
        epochs: int,
        lr: float,
        max_len: int,
        batch_size: int,
        trainer: Callable | None = None,
        predictor: Callable | None = None,
    ):
        self.model_id = base_model
        self.models_dir = Path(models_dir)
        self.cfg = TrainConfig(epochs, lr, max_len, batch_size)
        self._trainer = trainer or default_trainer
        self._predictor_factory = predictor or default_predictor
        self._predict = None
        self._label_ids: list[str] = []
        self.train_seconds: float | None = None

    def model_dir(self, dataset_name: str) -> Path:
        return self.models_dir / f"{self.model_id.replace('/', '__')}-{dataset_name}"

    def prepare(self, dataset: Dataset) -> None:
        out = self.model_dir(dataset.name)
        if not (out / "config.json").exists():
            print(f"  [bert-ft] training on {len(dataset.train)} examples -> {out}", flush=True)
            t0 = time.perf_counter()
            self._trainer(self.model_id, dataset.train, dataset.label_ids, out, self.cfg)
            self.train_seconds = time.perf_counter() - t0
            (out / "train_seconds.json").write_text(json.dumps(self.train_seconds))
        elif (out / "train_seconds.json").exists():
            self.train_seconds = json.loads((out / "train_seconds.json").read_text())
        self._label_ids = dataset.label_ids
        self._predict = self._predictor_factory(out)

    async def predict_batch(self, texts: list[str], labels: dict[str, str], parallelism: int) -> list[Prediction]:
        if self._predict is None:
            raise RuntimeError("call prepare(dataset) before predict_batch")
        ids = self._label_ids or list(labels)
        out: list[Prediction] = []
        for i in range(0, len(texts), parallelism):
            chunk = texts[i : i + parallelism]
            t0 = time.perf_counter()
            rows = self._predict(chunk, parallelism)
            per_ms = (time.perf_counter() - t0) * 1000 / len(chunk)
            for row in rows:
                probs = {l: float(p) for l, p in zip(ids, row)}
                out.append(Prediction(max(probs, key=probs.get), probs, per_ms, 0, 0, 0.0))
        return out
