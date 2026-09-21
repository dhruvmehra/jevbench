import json

from jevbench.report import write_summary


def result(clf, ds, acc, ece=None, tp=None):
    return {
        "classifier": clf, "model_id": f"model/{clf}", "dataset": ds, "n": 2, "n_labels": 2,
        "predictions": [], "train_seconds": 42.0 if clf == "bert-ft" else None,
        "timestamp": "t", "hardware": "Darwin arm64",
        "metrics": {"accuracy": acc, "macro_f1": acc, "ece": ece, "error_rate": 0.0,
                    "latency_p50_ms": 100.0, "latency_p95_ms": 200.0, "cost_per_1k_usd": 0.05,
                    "throughput_ex_s": tp, "latency_from_cache": False},
    }


def test_write_summary(tmp_path):
    (tmp_path / "jev__sst2.json").write_text(json.dumps(result("jev", "sst2", 0.9, ece=0.05, tp=12.3)))
    (tmp_path / "llm-cheap__sst2.json").write_text(json.dumps(result("llm-cheap", "sst2", 0.95)))
    (tmp_path / "bert-ft__agnews.json").write_text(json.dumps(result("bert-ft", "agnews", 0.8, ece=0.1)))
    out = write_summary(tmp_path)
    text = out.read_text()
    assert out.name == "summary.md"
    assert "## sst2" in text and "## agnews" in text
    assert "| jev | model/jev | 90.0 | 90.0 | 0.050 | 0.0 | 100 | 200 | 12.3 | $0.0500 |" in text
    assert "| llm-cheap | model/llm-cheap | 95.0 | 95.0 | – |" in text
    assert "fine-tuning took 42s" in text
    # best accuracy first within a dataset
    assert text.index("| llm-cheap |") < text.index("| jev |")
