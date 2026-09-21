import json

import httpx
import pytest

from jevbench.classifiers.llm import LLMClassifier


def resp(content, cost=0.0001):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 6, "cost": cost},
    }


def make_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_llm_parses_label_and_sends_schema():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=resp('{"label":"sports"}'))

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
    assert b["messages"][1] == {"role": "user", "content": "Lakers win"}


async def test_llm_off_enum_output_is_error():
    def handler(req):
        return httpx.Response(200, json=resp('{"label":"cricket"}'))

    clf = LLMClassifier("llm-cheap", "m", "KEY", client=make_client(handler))
    [p] = await clf.predict_batch(["x"], {"sports": "s"}, parallelism=1)
    assert p.label is None and "invalid label" in p.error
    assert p.cost_usd == pytest.approx(1e-4)  # cost is still recorded for failed outputs


async def test_llm_unparseable_is_error():
    def handler(req):
        return httpx.Response(200, json=resp("sports"))

    clf = LLMClassifier("llm-cheap", "m", "KEY", client=make_client(handler))
    [p] = await clf.predict_batch(["x"], {"sports": "s"}, parallelism=1)
    assert p.label is None and "parse" in p.error


async def test_llm_retries_without_reasoning_on_400():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append("reasoning" in body)
        if "reasoning" in body:
            return httpx.Response(400, json={"error": {"message": "reasoning not supported"}})
        return httpx.Response(200, json=resp('{"label":"sports"}'))

    clf = LLMClassifier("llm-cheap", "m", "KEY", client=make_client(handler), backoff=0)
    [p] = await clf.predict_batch(["x"], {"sports": "s"}, parallelism=1)
    assert p.label == "sports" and calls == [True, False]
    # subsequent calls skip reasoning entirely
    await clf.predict_batch(["y"], {"sports": "s"}, parallelism=1)
    assert calls == [True, False, False]


async def test_llm_no_reasoning_when_effort_none():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=resp('{"label":"sports"}'))

    clf = LLMClassifier("llm-cheap", "m", "KEY", reasoning_effort=None, client=make_client(handler))
    await clf.predict_batch(["x"], {"sports": "s"}, parallelism=1)
    assert "reasoning" not in seen["body"]
