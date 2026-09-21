from jevbench.classifiers.laya import LayaClassifier


class FakeAgent:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        crit = questions["label"]["criteria"]
        first = list(crit)[0]
        probs = {k: (0.7 if k == first else 0.3 / max(1, len(crit) - 1)) for k in crit}
        return {"model": "laya-rl-agent", "answers": {"label": {"type": "choice", "choice": first,
                "probabilities": probs, "confidence": 0.4, "action": "act"}},
                "usage": {"input_tokens": 94, "output_tokens": 0}}


async def test_laya_uses_jev_questions_and_parses_probs():
    agent = FakeAgent()
    clf = LayaClassifier(loader=lambda: agent)
    labels = {"neg": "negative", "pos": "positive"}
    clf.prepare(None)
    assert len(agent.calls) == 1  # warmup
    preds = await clf.predict_batch(["a", "b"], labels, parallelism=32)
    assert [p.label for p in preds] == ["neg", "neg"]
    assert preds[0].probs == {"neg": 0.7, "pos": 0.3}
    assert preds[0].input_tokens == 94 and preds[0].cost_usd == 0.0 and preds[0].error is None
    state, q = agent.calls[-1]
    assert state == "b" and q["label"]["type"] == "choice" and q["label"]["criteria"] == labels
    assert clf.load_seconds is not None


async def test_laya_records_error_per_example():
    class Boom(FakeAgent):
        def predict(self, state, questions):
            if state == "bad":
                raise RuntimeError("kaboom")
            return super().predict(state, questions)

    clf = LayaClassifier(loader=Boom)
    preds = await clf.predict_batch(["ok", "bad"], {"a": "a"}, parallelism=1)
    assert preds[0].label == "a" and preds[1].label is None and "kaboom" in preds[1].error
