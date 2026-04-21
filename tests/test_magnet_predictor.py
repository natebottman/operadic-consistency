"""
Tests for operadic_consistency.magnet.predictor.

These tests cover the parts that don't require MAGNET or Together.ai installed:
- Token F1 normalization
- Linear calibration
- Scenario-state column extraction helpers
- LLMBackend protocol + consistency computation against a mock backend
"""
from typing import Optional, Sequence

import pytest

from operadic_consistency.magnet.backends import LLMBackend, TogetherBackend
from operadic_consistency.magnet.predictor import (
    _normalize,
    _token_f1,
    _LinearCalibration,
    _extract_completion_text,
    compute_consistency_for_run,
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


# ── LLMBackend protocol & mock backend ────────────────────────────────────────

class _MockBackend:
    """Test backend that returns pre-canned responses by prompt-keyword match.

    ``responses`` maps keyword → response text. The first keyword that
    appears in the incoming prompt wins. Unmatched prompts return "".
    """

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        self.calls.append(prompt)
        for keyword, resp in self.responses.items():
            if keyword in prompt:
                return resp
        return ""


def test_mock_backend_satisfies_protocol():
    """Protocol is structural — a mock with `complete` should be recognized."""
    backend = _MockBackend({})
    assert isinstance(backend, LLMBackend)


def test_together_backend_satisfies_protocol():
    backend = TogetherBackend(model="dummy/model", api_key="sk-dummy")
    assert isinstance(backend, LLMBackend)


def test_compute_consistency_all_consistent():
    """When decomposition and sub-answers reconstruct the direct answer, consistency=1."""
    decomposer = _MockBackend({
        "Where was the director of Inception born?":
            "Q1: Who directed Inception?\nQ2: Where was [A1] born?",
    })
    answerer = _MockBackend({
        "Who directed Inception?":   "Christopher Nolan",
        "Where was Christopher Nolan born?": "London",
    })
    consistency = compute_consistency_for_run(
        questions=["Where was the director of Inception born?"],
        direct_answers=["London"],
        answerer=answerer,
        decomposer=decomposer,
    )
    assert consistency == pytest.approx(1.0)


def test_compute_consistency_decomp_failure_counts_inconsistent():
    """If the decomposer returns garbage, that instance is inconsistent."""
    decomposer = _MockBackend({})  # returns "" for everything → no Q1/Q2 match
    answerer = _MockBackend({})
    consistency = compute_consistency_for_run(
        questions=["Some question"],
        direct_answers=["Some answer"],
        answerer=answerer,
        decomposer=decomposer,
    )
    assert consistency == pytest.approx(0.0)


def test_compute_consistency_mixed():
    """Two instances — one consistent, one where answerer's expansion disagrees."""
    decomposer = _MockBackend({
        "A?": "Q1: sub-a1?\nQ2: follow-up given [A1]?",
        "B?": "Q1: sub-b1?\nQ2: follow-up given [A1]?",
    })
    answerer = _MockBackend({
        "sub-a1?": "alpha",
        "follow-up given alpha?": "correct-A",
        "sub-b1?": "beta",
        "follow-up given beta?": "wrong-B",   # disagrees with direct
    })
    consistency = compute_consistency_for_run(
        questions=["A?", "B?"],
        direct_answers=["correct-A", "expected-B"],
        answerer=answerer,
        decomposer=decomposer,
    )
    assert consistency == pytest.approx(0.5)
