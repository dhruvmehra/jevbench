"""sqlite cache of Prediction records keyed on classifier/model/dataset/text/labels."""

from __future__ import annotations

import hashlib
import json
import sqlite3

from .classifiers.base import Prediction


class Cache:
    def __init__(self, path: str = "cache.sqlite"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.commit()

    @staticmethod
    def key(classifier: str, model_id: str, dataset: str, text: str, labels: dict) -> str:
        blob = json.dumps([classifier, model_id, dataset, text, labels], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Prediction | None:
        row = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return Prediction.from_dict(json.loads(row[0])) if row else None

    def put(self, key: str, pred: Prediction) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, json.dumps(pred.to_dict(), default=str)),
        )
        self.conn.commit()
