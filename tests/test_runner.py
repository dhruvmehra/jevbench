from jevbench.cache import Cache
from jevbench.classifiers.base import Prediction
from jevbench.datasets import Dataset, Example
from jevbench.runner import build_classifiers, evaluate, group_classifiers


class Fake:
    name = "jev"
    model_id = "m"

    def __init__(self):
        self.calls = []

    def prepare(self, ds):
        pass

    async def predict_batch(self, texts, labels, parallelism):
        self.calls.append(parallelism)
        return [Prediction("a" if t == "x" else "b", {"a": 0.9, "b": 0.1}, 10.0, 5, 1, 1e-6) for t in texts]


async def test_evaluate_metrics_and_cache(tmp_path):
    ds = Dataset("sst2", {"a": "A", "b": "B"}, [Example("x", "a"), Example("y", "a")])
    cache = Cache(str(tmp_path / "c.sqlite"))
    clf = Fake()
    r = await evaluate(clf, ds, cache, parallelism_api=4, local_batch=8, skip_throughput=False, use_cache=True)
    m = r["metrics"]
    assert m["accuracy"] == 0.5 and m["latency_p50_ms"] == 10.0
    assert abs(m["cost_per_1k_usd"] - 0.001) < 1e-12
    assert m["throughput_ex_s"] > 0 and m["throughput_parallelism"] == 4
    assert m["avg_input_tokens"] == 5
    assert clf.calls == [1, 4]
    assert len(r["predictions"]) == 2 and r["dataset"] == "sst2"

    clf2 = Fake()
    r2 = await evaluate(clf2, ds, cache, 4, 8, skip_throughput=True, use_cache=True)
    assert clf2.calls == []
    assert r2["metrics"]["latency_from_cache"] is True and r2["metrics"]["latency_p50_ms"] is None
    assert r2["metrics"]["accuracy"] == 0.5


async def test_errors_are_not_cached(tmp_path):
    class Flaky(Fake):
        async def predict_batch(self, texts, labels, parallelism):
            self.calls.append(parallelism)
            return [Prediction(None, None, 1.0, error="boom") for _ in texts]

    ds = Dataset("sst2", {"a": "A"}, [Example("x", "a")])
    cache = Cache(str(tmp_path / "c.sqlite"))
    r = await evaluate(Flaky(), ds, cache, 4, 8, skip_throughput=True, use_cache=True)
    assert r["metrics"]["error_rate"] == 1.0 and r["metrics"]["accuracy"] == 0.0
    clf = Flaky()
    await evaluate(clf, ds, cache, 4, 8, skip_throughput=True, use_cache=True)
    assert clf.calls == [1]  # nothing was cached, so it re-ran


async def test_local_classifier_gets_warmup_call(tmp_path):
    class Local(Fake):
        name = "bert-zs"

    ds = Dataset("sst2", {"a": "A", "b": "B"}, [Example("x", "a"), Example("y", "a")])
    clf = Local()
    await evaluate(clf, ds, Cache(str(tmp_path / "c.sqlite")), 4, 8, skip_throughput=False, use_cache=False)
    assert clf.calls == [1, 1, 8]  # warmup, latency pass, throughput at local batch size


def test_build_classifiers_local_only_needs_no_key():
    cfg = {
        "jev": {"model": "typesafe/jev-1.13"},
        "laya": {"model": "convaiinnovations/laya"},
        "llm": {"cheap": {"model": "openai/gpt-5-mini", "reasoning_effort": "minimal"}},
        "bert": {"finetune_base": "distilbert-base-uncased", "zeroshot_model": "facebook/bart-large-mnli",
                 "epochs": 1, "lr": 1e-5, "max_len": 64, "batch_size": 8},
    }
    clfs = build_classifiers(["jev", "llm-cheap", "bert-zs", "bert-ft", "laya"], cfg, "KEY")
    assert [c.name for c in clfs] == ["jev", "llm-cheap", "bert-zs", "bert-ft", "laya"]
    assert clfs[1].model_id == "openai/gpt-5-mini" and clfs[1].reasoning_effort == "minimal"


def test_group_classifiers_splits_api_from_local():
    class C:
        def __init__(self, name):
            self.name = name
            self.model_id = "m"

    api, local = group_classifiers([C("jev"), C("bert-ft"), C("llm-cheap"), C("bert-zs"), C("llm-frontier")])
    assert [c.name for c in api] == ["jev", "llm-cheap", "llm-frontier"]
    assert [c.name for c in local] == ["bert-ft", "bert-zs"]


async def test_unbatchable_local_classifier_throughput_uses_parallelism_1(tmp_path):
    class Laya(Fake):
        name = "laya"
        supports_batch = False

    ds = Dataset("sst2", {"a": "A", "b": "B"}, [Example("x", "a")])
    clf = Laya()
    r = await evaluate(clf, ds, Cache(str(tmp_path / "c.sqlite")), 4, 32, skip_throughput=False, use_cache=False)
    assert clf.calls == [1, 1, 1] and r["metrics"]["throughput_parallelism"] == 1
