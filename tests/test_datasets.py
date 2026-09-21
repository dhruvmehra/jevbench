from jevbench.datasets import DATASETS, load


def fake_loader(hf_name, split):
    return [{"sentence": f"s{i}", "text": f"t{i}", "label": i % 2} for i in range(20)]


def test_specs_have_full_label_maps():
    assert len(DATASETS["sst2"].labels) == 2
    assert len(DATASETS["agnews"].labels) == 4
    assert len(DATASETS["banking77"].labels) == 77
    # label ids must be safe identifiers for JEV criteria keys / JSON enums
    for spec in DATASETS.values():
        for k in spec.labels:
            assert k == k.lower() and "?" not in k and " " not in k


def test_load_samples_deterministically_and_maps_labels():
    d1 = load("sst2", n=5, seed=0, train_cap=3, loader=fake_loader)
    d2 = load("sst2", n=5, seed=0, train_cap=3, loader=fake_loader)
    assert [e.text for e in d1.examples] == [e.text for e in d2.examples]
    assert len(d1.examples) == 5 and len(d1.train) == 3
    assert set(e.label for e in d1.examples) <= {"negative", "positive"}
    assert d1.examples[0].text.startswith("s")
    assert d1.label_ids == ["negative", "positive"]


def test_agnews_uses_text_field():
    d = load("agnews", n=2, seed=1, train_cap=1, loader=fake_loader)
    assert d.examples[0].text.startswith("t")
