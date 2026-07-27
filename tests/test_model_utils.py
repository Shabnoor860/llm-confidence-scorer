"""
tests/test_model_utils.py — run with: pytest
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model_utils as mu


def test_assign_domain_science():
    assert mu.assign_domain("Einstein developed the theory of relativity.") == "Science"


def test_assign_domain_geography():
    assert mu.assign_domain("The Nile is a river in Africa.") == "Geography"


def test_assign_domain_fallback_other():
    assert mu.assign_domain("Blorp is a fictional word with no keywords.") == "Other"


def test_load_data_returns_expected_columns():
    df, source = mu.load_data()
    assert {"claim", "label", "domain"}.issubset(df.columns)
    assert len(df) > 0
    assert set(df["label"].unique()).issubset({0, 1})
    assert isinstance(source, str)


def test_expected_calibration_error_perfect_calibration():
    # If confidence exactly matches accuracy in every bin, ECE should be ~0
    probs = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    ece, bins = mu.expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 0.15  # not exactly zero due to binning, but should be small


def test_expected_calibration_error_bounds():
    probs = np.random.uniform(0, 1, 50)
    labels = np.random.randint(0, 2, 50)
    ece, _ = mu.expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0


def test_fit_temperature_returns_positive_scalar():
    logits = np.random.randn(50) * 3
    labels = np.random.randint(0, 2, 50)
    T = mu.fit_temperature(logits, labels)
    assert T > 0


def test_train_model_and_baseline_run_end_to_end():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["test_probs"]) == len(result["y_test"])

    baseline = mu.train_baseline(
        result["X_train"], result["y_train"], result["X_test"], result["y_test"]
    )
    assert 0.0 <= baseline["accuracy"] <= 1.0


def test_classification_metrics_shape():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    metrics = mu.classification_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"precision", "recall", "f1", "confusion_matrix"}
    assert len(metrics["confusion_matrix"]) == 2


def test_bias_analysis_runs():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    bias = mu.bias_analysis(df, result["model"], result["vectorizer"])
    assert isinstance(bias, dict)
    for domain, (acc, count) in bias.items():
        assert 0.0 <= acc <= 100.0
        assert count > 0


def test_self_verification_cascade_length():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    scores = mu.self_verification_cascade(
        result["model"], result["vectorizer"], "The Eiffel Tower is in Paris.", rounds=4
    )
    assert len(scores) == 5  # original + 4 rounds
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_selective_prediction_curve_monotonic_coverage():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    curve = mu.selective_prediction_curve(result["test_probs"], result["y_test"])
    coverages = [c[1] for c in curve]
    # coverage should be non-increasing as threshold rises
    assert all(coverages[i] >= coverages[i + 1] - 1e-9 for i in range(len(coverages) - 1))


def test_paraphrase_consistency_keys():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    pc = mu.paraphrase_consistency(
        result["model"], result["vectorizer"],
        ["The Eiffel Tower is in Paris.", "Paris has the Eiffel Tower."]
    )
    assert set(pc.keys()) == {"scores", "mean", "std", "range", "label_flip"}
    assert len(pc["scores"]) == 2


def test_error_taxonomy_returns_dataframe():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    taxonomy = mu.error_taxonomy(df, result["model"], result["vectorizer"])
    assert "category" in taxonomy.columns or taxonomy.empty
    assert "count" in taxonomy.columns or taxonomy.empty


def test_bootstrap_accuracy_ci_bounds():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1])
    mean, lo, hi = mu.bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=200)
    assert 0.0 <= lo <= mean <= hi <= 1.0

