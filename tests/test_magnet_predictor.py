"""
Tests for operadic_consistency.magnet.predictor.

These tests cover the parts that don't require MAGNET or Together.ai installed:
- Token F1 normalization
- Linear calibration
- Scenario-state column extraction helpers
"""
import pytest

from operadic_consistency.magnet.predictor import (
    _normalize,
    _token_f1,
    _LinearCalibration,
    _extract_completion_text,
)


# ── Token F1 ──────────────────────────────────────────────────────────────────

def test_token_f1_exact():
    assert _token_f1("Harry Truman", "Harry Truman") == pytest.approx(1.0)


def test_token_f1_partial():
    score = _token_f1("Harry Truman", "Truman")
    assert 0.0 < score < 1.0


def test_token_f1_zero():
    assert _token_f1("Paris", "London") == 0.0


def test_token_f1_empty():
    assert _token_f1("", "") == 1.0
    assert _token_f1("", "something") == 0.0


def test_normalize_strips_articles():
    assert _normalize("The Quick Brown Fox") == "quick brown fox"
    assert _normalize("a dog and an elephant") == "dog and elephant"


# ── LinearCalibration ────────────────────────────────────────────────────────

def test_calibration_prior():
    cal = _LinearCalibration()
    # Prior: with consistency=0.55, accuracy should be around 0.60
    pred = cal.predict(0.55)
    assert 0.4 < pred < 0.8  # reasonable range


def test_calibration_fit_perfect():
    # If accuracy == consistency, slope should be ~1 and intercept ~0
    xs = [0.3, 0.5, 0.7, 0.9]
    cal = _LinearCalibration.fit(xs, xs)
    assert cal.predict(0.5) == pytest.approx(0.5, abs=0.05)


def test_calibration_fit_fallback_on_single_point():
    cal = _LinearCalibration.fit([0.6], [0.7])
    # Falls back to prior
    assert cal.n_points == 0


def test_calibration_clips_to_unit_interval():
    cal = _LinearCalibration(slope=10.0, intercept=-4.0)
    assert cal.predict(0.0) == pytest.approx(0.0)
    assert cal.predict(1.0) == pytest.approx(1.0)


# ── Completion extraction ─────────────────────────────────────────────────────

def test_extract_completion_text_string():
    assert _extract_completion_text("  hello  ") == "hello"


def test_extract_completion_text_list_of_dicts():
    val = [{"text": "Harry Truman"}, {"text": "other"}]
    assert _extract_completion_text(val) == "Harry Truman"


def test_extract_completion_text_empty_list():
    assert _extract_completion_text([]) == ""
