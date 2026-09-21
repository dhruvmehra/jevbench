import pytest

from jevbench.classifiers.bert_finetune import FineTunedClassifier
from jevbench.datasets import Dataset, Example


async def test_trains_once_then_loads(tmp_path):
    calls = []

    def trainer(base, train, ids, out, cfg):
        calls.append((base, len(train), ids, cfg.epochs))
        out.mkdir(parents=True)
        (out / "config.json").write_text("{}")

    def predictor(model_dir):
        assert (model_dir / "config.json").exists()
        return lambda texts, batch_size: [[0.2, 0.8] for _ in texts]

    ds = Dataset("sst2", {"neg": "n", "pos": "p"}, [Example("x", "pos")], [Example("t", "pos"), Example("u", "neg")])
    clf = FineTunedClassifier("distilbert-base-uncased", str(tmp_path), 1, 5e-5, 128, 8, trainer=trainer, predictor=predictor)
    clf.prepare(ds)
    clf.prepare(ds)
    assert calls == [("distilbert-base-uncased", 2, ["neg", "pos"], 1)]
    assert clf.train_seconds is not None and clf.train_seconds >= 0
    [p] = await clf.predict_batch(["x"], ds.labels, parallelism=1)
    assert p.label == "pos" and p.probs == {"neg": 0.2, "pos": 0.8}
    assert p.cost_usd == 0.0


async def test_predict_before_prepare_raises(tmp_path):
    clf = FineTunedClassifier("m", str(tmp_path), 1, 1e-5, 64, 4, trainer=lambda *a: None, predictor=lambda d: None)
    with pytest.raises(RuntimeError):
        await clf.predict_batch(["x"], {"a": "a"}, parallelism=1)
