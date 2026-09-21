from jevbench.classifiers.bert_zeroshot import ZeroShotClassifier


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, candidate_labels, batch_size):
        self.calls.append((len(texts), batch_size))
        # always prefer the second candidate
        return [{"labels": [candidate_labels[1], candidate_labels[0]], "scores": [0.7, 0.3]} for _ in texts]


async def test_zero_shot_maps_descriptions_back_to_ids():
    pipe = FakePipe()
    clf = ZeroShotClassifier("facebook/bart-large-mnli", pipeline_factory=lambda: pipe)
    labels = {"neg": "negative sentiment", "pos": "positive sentiment"}
    preds = await clf.predict_batch(["a", "b", "c"], labels, parallelism=2)
    assert [p.label for p in preds] == ["pos", "pos", "pos"]
    assert preds[0].probs == {"pos": 0.7, "neg": 0.3}
    assert preds[0].cost_usd == 0.0 and preds[0].latency_ms >= 0
    assert pipe.calls == [(2, 2), (1, 2)]


async def test_zero_shot_prepare_builds_pipeline_once():
    built = []
    clf = ZeroShotClassifier("m", pipeline_factory=lambda: built.append(1) or FakePipe())
    clf.prepare(None)
    clf.prepare(None)
    assert built == [1]
