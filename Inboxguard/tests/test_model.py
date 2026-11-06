"""Model sanity checks. Skipped when no real (non-demo) model.pkl is present."""
import pytest

import analyzer

bundle = analyzer.get_model()
pytestmark = pytest.mark.skipif(not bundle or bundle.get("demo"), reason="no real trained model")


def test_obvious_phishing_scores_high_and_chat_scores_low():
    p_hi, _ = analyzer.ml_probability(analyzer.ml_text("Your account has been suspended. Verify your password now to claim your free gift or it will be closed."))
    p_lo, _ = analyzer.ml_probability(analyzer.ml_text("Thanks for the update. Can we move the meeting to Thursday afternoon? I attached the notes."))
    assert p_hi > 70 and p_lo < 30


def test_model_does_not_read_years():
    a, _ = analyzer.ml_probability(analyzer.ml_text("Meeting notes for the project review, circulated in 2001."))
    b, _ = analyzer.ml_probability(analyzer.ml_text("Meeting notes for the project review, circulated in 2025."))
    assert abs(a - b) < 1


def test_committed_metrics_are_from_real_data():
    import json, os
    m = json.load(open(os.path.join(os.path.dirname(analyzer.__file__), "metrics.json")))
    assert "demo" not in m["source"].lower(), "model.pkl/metrics.json were trained on demo data"
    assert "phishing_pot" in m["source"] or "kaggle" in m["source"].lower() or m["test_size"] > 1000
    assert m["test_size"] > 1000
