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
    InferenceRequest,
    OperadicConsistencyPredictor,
    _consistency_from_cache,
    _extract_completion_text,
    _LinearCalibration,
    _make_answer_request,
    _make_decompose_request,
    _normalize,
    _token_f1,
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


# ── Batched inference API ─────────────────────────────────────────────────────

def test_request_id_is_deterministic():
    """Same (role, model, prompt) triple → same request_id; different triple → different."""
    r1 = _make_decompose_request("Where was Nolan born?", "m1")
    r2 = _make_decompose_request("Where was Nolan born?", "m1")
    r3 = _make_decompose_request("Where was Nolan born?", "m2")  # different model
    r4 = _make_decompose_request("Where was Spielberg born?", "m1")  # different question
    assert r1.request_id == r2.request_id
    assert r1.request_id != r3.request_id
    assert r1.request_id != r4.request_id


def test_request_id_roles_distinct():
    """A decomposer request and an answerer request never collide."""
    d = _make_decompose_request("q", "m")
    a = _make_answer_request("q", "m")
    assert d.request_id != a.request_id
    assert d.role == "decomposer"
    assert a.role == "answerer"


def test_consistency_from_cache_matches_dynamic_path():
    """Filling the cache with what the dynamic path would get produces the same score."""
    decomposer_model = "dm"
    answerer_model = "am"

    question = "Where was the director of Inception born?"
    direct = "London"

    # Build the cache as if a harness had run the three phases
    cache = {}
    decomp_req = _make_decompose_request(question, decomposer_model)
    cache[decomp_req.request_id] = "Q1: Who directed Inception?\nQ2: Where was [A1] born?"

    q1_req = _make_answer_request("Who directed Inception?", answerer_model)
    cache[q1_req.request_id] = "Christopher Nolan"

    q2_req = _make_answer_request("Where was Christopher Nolan born?", answerer_model)
    cache[q2_req.request_id] = "London"

    consistency = _consistency_from_cache(
        questions=[question],
        direct_answers=[direct],
        answerer_model=answerer_model,
        decomposer_model=decomposer_model,
        cache=cache,
    )
    assert consistency == pytest.approx(1.0)


def test_consistency_from_cache_missing_entry_counts_inconsistent():
    """Missing cache entries are treated as failed steps → inconsistent."""
    consistency = _consistency_from_cache(
        questions=["Q?"],
        direct_answers=["A"],
        answerer_model="am",
        decomposer_model="dm",
        cache={},   # nothing precomputed
    )
    assert consistency == pytest.approx(0.0)


# ── plan_next_batch + predict_from_cache end-to-end ──────────────────────────

def _make_fake_splits(n_train_runs: int, n_test_runs: int, n_questions: int):
    """Build minimal TrainSplit/SequesteredTestSplit-shaped objects backed by
    pandas DataFrames. Good enough for the predictor's .groupby access pattern."""
    pd = pytest.importorskip("pandas")

    def _scenario_rows(run_name: str) -> list[dict]:
        return [
            {
                "run_spec.name": run_name,
                "scenario_state.request_states.instance.input.text": f"{run_name}-q{i}",
                "scenario_state.request_states.result.completions":
                    [{"text": f"{run_name}-direct-{i}"}],
            }
            for i in range(n_questions)
        ]

    train_rows, stats_rows = [], []
    for i in range(n_train_runs):
        name = f"train-run-{i}"
        train_rows += _scenario_rows(name)
        stats_rows.append({
            "run_spec.name": name,
            "stats.name.name": "exact_match",
            "stats.name.split": "valid",
            "stats.mean": 0.3 + 0.1 * i,   # distinct values so calibration has signal
        })

    test_rows = []
    for i in range(n_test_runs):
        test_rows += _scenario_rows(f"test-run-{i}")

    class _Split:
        def __init__(self, scenario_state, stats=None):
            self.scenario_state = scenario_state
            self.stats = stats

    train_split = _Split(pd.DataFrame(train_rows), pd.DataFrame(stats_rows))
    test_split = _Split(pd.DataFrame(test_rows))
    return train_split, test_split


def test_plan_next_batch_three_phases_then_empty():
    """plan_next_batch walks decompose → Q1 → Q2 → []."""
    pytest.importorskip("pandas")
    train, test = _make_fake_splits(n_train_runs=2, n_test_runs=1, n_questions=2)

    # Dummy backends — we only care about their .model attribute here.
    answerer = TogetherBackend(model="am", api_key="sk-dummy")
    decomposer = TogetherBackend(model="dm", api_key="sk-dummy")
    pred = OperadicConsistencyPredictor(
        answerer=answerer,
        decomposer=decomposer,
        num_example_runs=2,
        num_eval_samples=2,
        n_consistency_samples=2,
    )

    cache: dict = {}
    # Phase 1: decomposition requests only
    phase1 = pred.plan_next_batch(train, test, cache)
    assert phase1, "expected a non-empty decomposition batch"
    assert all(r.role == "decomposer" for r in phase1)
    assert all(r.model == "dm" for r in phase1)
    # Fulfill with well-formed decompositions
    for req in phase1:
        q_text = req.prompt.split("Question: ", 1)[1].strip()
        cache[req.request_id] = f"Q1: sub of {q_text}\nQ2: follow-up given [A1]?"

    # Phase 2: Q1 answerer requests
    phase2 = pred.plan_next_batch(train, test, cache)
    assert phase2, "expected a non-empty Q1 batch"
    assert all(r.role == "answerer" for r in phase2)
    assert all(r.model == "am" for r in phase2)
    for req in phase2:
        cache[req.request_id] = "A1-answer"

    # Phase 3: Q2 answerer requests (rendered with A1)
    phase3 = pred.plan_next_batch(train, test, cache)
    assert phase3, "expected a non-empty Q2 batch"
    assert all(r.role == "answerer" for r in phase3)
    # Each Q2 prompt should contain the resolved A1 text
    assert all("A1-answer" in r.prompt for r in phase3)
    for req in phase3:
        cache[req.request_id] = "expansion"

    # Phase 4: done
    assert pred.plan_next_batch(train, test, cache) == []


def test_predict_from_cache_produces_predictions():
    """After filling the cache via plan_next_batch, predict_from_cache returns one
    RunPrediction per sequestered test run."""
    pytest.importorskip("pandas")
    train, test = _make_fake_splits(n_train_runs=2, n_test_runs=2, n_questions=2)

    answerer = TogetherBackend(model="am", api_key="sk-dummy")
    decomposer = TogetherBackend(model="dm", api_key="sk-dummy")
    pred = OperadicConsistencyPredictor(
        answerer=answerer,
        decomposer=decomposer,
        num_example_runs=2,
        num_eval_samples=2,
        n_consistency_samples=2,
    )

    # Walk phases, filling the cache
    cache: dict = {}
    while True:
        batch = pred.plan_next_batch(train, test, cache)
        if not batch:
            break
        for req in batch:
            if req.role == "decomposer":
                q = req.prompt.split("Question: ", 1)[1].strip()
                cache[req.request_id] = f"Q1: s-{q}\nQ2: follow-up given [A1]?"
            else:
                cache[req.request_id] = "placeholder-answer"

    predictions = pred.predict_from_cache(train, test, cache)
    # One prediction per test run, and each prediction's mean is in [0, 1]
    test_run_names = {f"test-run-{i}" for i in range(2)}
    assert {p.run_spec_name for p in predictions} == test_run_names
    for p in predictions:
        assert 0.0 <= p.mean <= 1.0
        assert p.stat_name == "exact_match"
