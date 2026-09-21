from jevbench.cache import Cache
from jevbench.classifiers.base import Prediction


def test_cache_roundtrip(tmp_path):
    c = Cache(str(tmp_path / "c.sqlite"))
    k = c.key("jev", "typesafe/jev-1.13", "sst2", "hello", {"a": "x"})
    assert c.get(k) is None
    p = Prediction(label="a", probs={"a": 0.9, "b": 0.1}, latency_ms=12.5, cost_usd=1e-6, raw={"x": 1})
    c.put(k, p)
    got = c.get(k)
    assert got.label == "a" and got.probs["a"] == 0.9 and got.raw == {"x": 1}
    assert got.error is None
    assert c.key("jev", "m", "sst2", "hello", {"a": "x"}) != c.key("jev", "m", "sst2", "hello", {"a": "y"})
