import json
from pathlib import Path

import httpx
import pytest

from jevbench.classifiers.jev import JevClassifier, build_questions

FIX = json.loads(Path("tests/fixtures/jev_choice.json").read_text())


def make_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_jev_parses_choice_and_usage():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        seen["url"] = str(req.url)
        return httpx.Response(200, json=FIX)

    clf = JevClassifier("typesafe/jev-1.13", "KEY", client=make_client(handler))
    labels = {"negative": "neg", "positive": "pos"}
    [p] = await clf.predict_batch(["great movie"], labels, parallelism=1)
    assert p.label == "positive" and p.probs["positive"] == 0.92
    assert p.input_tokens == 120 and p.cost_usd == pytest.approx(5.04e-6)
    assert p.error is None and p.latency_ms >= 0
    assert seen["auth"] == "Bearer KEY"
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["body"]["model"] == "typesafe/jev-1.13"
    assert seen["body"]["state"] == "great movie"
    assert seen["body"]["questions"]["label"]["criteria"] == labels


async def test_jev_records_error_on_500_after_retries():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(500, json={"error": "boom"})

    clf = JevClassifier("m", "KEY", client=make_client(handler), backoff=0)
    [p] = await clf.predict_batch(["x"], {"a": "a"}, parallelism=1)
    assert p.label is None and "500" in p.error
    assert len(calls) == 4  # 1 + 3 retries


async def test_jev_respects_parallelism_and_order():
    def handler(req):
        body = json.loads(req.content)
        fix = json.loads(json.dumps(FIX))
        fix["answers"]["label"]["choice"] = body["state"]
        return httpx.Response(200, json=fix)

    clf = JevClassifier("m", "KEY", client=make_client(handler))
    preds = await clf.predict_batch(["a", "b", "c"], {"a": "", "b": "", "c": ""}, parallelism=2)
    assert [p.label for p in preds] == ["a", "b", "c"]


def test_build_questions():
    q = build_questions({"a": "desc a"})
    assert q["label"]["type"] == "choice" and q["label"]["criteria"] == {"a": "desc a"}
