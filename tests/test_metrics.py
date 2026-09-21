from jevbench.metrics import accuracy, macro_f1, ece, percentile, error_rate


def test_accuracy_ignores_none():
    assert accuracy(["a", "b", "a"], ["a", "b", None]) == 1.0
    assert accuracy(["a", "b"], ["a", "a"]) == 0.5
    assert accuracy(["a"], [None]) == 0.0


def test_macro_f1_perfect_and_half():
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    # a: tp=1 fp=1 fn=0 -> p=.5 r=1 f=.667 ; b: tp=0 fp=0 fn=1 -> f=0 ; macro=.333
    assert abs(macro_f1(["a", "b"], ["a", "a"], ["a", "b"]) - 1 / 3) < 1e-9


def test_ece_calibrated_is_zero_and_none_without_probs():
    probs = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]
    assert ece(["a", "b"], probs) == 0.0
    assert ece(["a"], [None]) is None
    assert ece(["a"], [{"a": 0.0, "b": 1.0}]) == 1.0


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert abs(percentile([1, 2, 3, 4, 5], 95) - 4.8) < 1e-9


def test_error_rate():
    assert error_rate([None, "boom", None, "x"]) == 0.5
    assert error_rate([]) == 0.0
