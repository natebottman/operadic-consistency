"""
OperadicConsistencyPredictor
============================

A MAGNET RunPredictor that uses operadic consistency as a signal to predict
model accuracy without access to ground truth labels.

Algorithm
---------
1. **Decompose** each question into a 2-step Tree of Questions (ToQ) using an
   LLM decomposer.
2. **Evaluate** the ToQ in two ways:
   - *Direct*: The model's answer that was already recorded in HELM's
     scenario_state (no extra LLM call needed).
   - *Expansion*: Answer sub-questions sequentially with the answerer LLM, then
     compose to get the root answer.
3. **Consistency score**: Fraction of instances where direct ≈ expansion
   (token-F1 ≥ threshold).
4. **Calibration**: Fit a consistency → accuracy mapping using training runs,
   which have both consistency scores (computed here) and ground-truth accuracy
   (from MAGNET's ``train_split.stats``).  With 2+ runs: affine OLS.
   With 1 run: linear through the origin.  With 0 runs: identity.
5. **Predict**: Apply the calibration to each test run's consistency score.

Two groups of models
--------------------
The MAGNET interface cleanly separates the two groups the caller provides:

- **Training runs** (``train_split``): models for which ground-truth accuracy
  is available.  These are used to fit the consistency → accuracy mapping.
- **Test runs** (``sequestered_test_split``): models for which only raw
  outputs are available.  Consistency is computed and mapped to a predicted
  accuracy via the fitted calibration.

Usage
-----
The predictor is driven by swappable ``LLMBackend`` objects. The reference
backend calls Together.ai; the TA2 harness can plug in a vllm server or any
other completion endpoint without touching this module (see
``operadic_consistency.magnet.backends``).

::

    from operadic_consistency.magnet import (
        OperadicConsistencyPredictor, TogetherBackend,
    )

    answerer = TogetherBackend(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key="YOUR_KEY",
    )
    predictor = OperadicConsistencyPredictor(
        answerer=answerer,              # decomposer defaults to answerer
        num_eval_samples=20,
        n_consistency_samples=20,
    )
    predictor(helm_suites="path/to/benchmark_output/runs/suite_name")

Legacy kwargs (``answerer_model`` + ``together_api_key``) are still accepted
for backward compatibility and will auto-construct a ``TogetherBackend``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from operadic_consistency.magnet.backends import LLMBackend, TogetherBackend

log = logging.getLogger(__name__)


# ── Token F1 (SQuAD normalization) ───────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    precision = sum(p_toks.count(t) for t in common) / len(p_toks)
    recall = sum(g_toks.count(t) for t in common) / len(g_toks)
    return 2 * precision * recall / (precision + recall)


# ── LLM helpers ───────────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = (
    "Break the following multi-hop question into exactly two sub-questions "
    "where the answer to the first fills a blank in the second.\n"
    "Format your answer ONLY as:\n"
    "Q1: <first sub-question>\n"
    "Q2: <second sub-question, using [A1] as placeholder for the answer to Q1>\n\n"
    "Question: {question}\n"
)

ANSWER_PROMPT = (
    "Answer the following question with a short phrase (a few words only).\n\n"
    "Question: {question}\n"
    "Answer:"
)


def _parse_decomposition(raw: str) -> Optional[tuple[str, str]]:
    """Parse a decomposer's ``'Q1: ...\\nQ2: ...'`` output into ``(q1, q2_template)``.

    Returns ``None`` if the format can't be parsed. If Q2 does not contain the
    ``[A1]`` placeholder, one is appended so downstream rendering still works.
    This function is pure — it's reused by both the dynamic and batched paths.
    """
    q1_m = re.search(r"Q1:\s*(.+?)(?:\n|$)", raw)
    q2_m = re.search(r"Q2:\s*(.+?)(?:\n|$)", raw)
    if not q1_m or not q2_m:
        return None
    q1 = q1_m.group(1).strip()
    q2 = q2_m.group(1).strip()
    if "[A1]" not in q2:
        q2 = q2 + " (given that [A1])"
    return q1, q2


def _clean_answer(raw: str) -> str:
    """Normalize an answerer's raw completion down to a single short phrase."""
    return raw.strip().split("\n")[0].strip()


def _decompose(question: str, backend: LLMBackend) -> Optional[tuple[str, str]]:
    """Decompose a question into (q1, q2_template) via ``backend``; None on failure."""
    prompt = DECOMPOSE_PROMPT.format(question=question)
    raw = backend.complete(prompt, max_tokens=128)
    return _parse_decomposition(raw)


def _answer(question: str, backend: LLMBackend) -> str:
    """Get a short answer to a question via ``backend``."""
    prompt = ANSWER_PROMPT.format(question=question)
    raw = backend.complete(prompt, max_tokens=32)
    return _clean_answer(raw)


# ── Consistency computation ───────────────────────────────────────────────────

def compute_consistency_for_run(
    questions: Sequence[str],
    direct_answers: Sequence[str],
    answerer: LLMBackend,
    decomposer: LLMBackend,
    f1_threshold: float = 0.5,
) -> float:
    """
    For each (question, direct_answer) pair:
      1. Decompose question via ``decomposer`` → (q1, q2_template)
      2. Answer q1 via ``answerer`` → a1
      3. Render q2 with a1 → answer q2 via ``answerer`` → expansion_answer
      4. consistent_i = token_f1(direct_answer, expansion_answer) >= threshold

    Returns fraction of consistent instances (0.0–1.0).
    Instances that fail to decompose are counted as inconsistent.
    """
    n_consistent = 0
    n_total = 0

    for question, direct in zip(questions, direct_answers):
        n_total += 1
        try:
            decomp = _decompose(question, decomposer)
            if decomp is None:
                log.debug("Decompose failed for: %s", question[:60])
                continue

            q1, q2_tmpl = decomp

            # Answer q1
            a1 = _answer(q1, answerer)
            if not a1:
                continue

            # Render q2 with a1
            q2 = q2_tmpl.replace("[A1]", a1)
            expansion = _answer(q2, answerer)
            if not expansion:
                continue

            if _token_f1(direct, expansion) >= f1_threshold:
                n_consistent += 1

        except Exception as e:
            log.warning("Consistency check error: %s", e)

    if n_total == 0:
        return 0.0
    return n_consistent / n_total


# ── Batched inference API (for TA2 harness integration) ──────────────────────

# The batched path lets the TA2 harness precompute all LLM calls the predictor
# needs in a small number of sequential batches (decompose → answer-Q1 →
# answer-Q2). The predictor then computes consistency from the filled cache
# with no further inference. This matches Kitware's preferred integration:
# LLM calls are owned by the harness, not by the predictor.
#
# Flow:
#     cache: dict[str, str] = {}
#     while True:
#         batch = predictor.plan_next_batch(train, test, cache)
#         if not batch:
#             break
#         results = harness.run_batch(batch)          # {request_id: text}
#         cache.update(results)
#     predictions = predictor.predict_from_cache(train, test, cache)


@dataclass(frozen=True)
class InferenceRequest:
    """A single LLM call declared by the predictor.

    The harness is expected to execute a batch of these requests and return
    results keyed by ``request_id`` so the predictor can look them up.
    ``request_id`` is a deterministic hash of ``(role, model, prompt)``, which
    means two requests with the same triple are naturally deduplicated.
    """
    request_id: str
    role: str            # "decomposer" or "answerer"
    model: str           # model ID (harness uses this for routing)
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


def _request_id(role: str, model: str, prompt: str) -> str:
    """Deterministic request ID: ``<role>:<16-char sha256(role|model|prompt)>``."""
    key = f"{role}|{model}|{prompt}".encode("utf-8")
    return f"{role}:" + hashlib.sha256(key).hexdigest()[:16]


def _make_decompose_request(question: str, model: str) -> InferenceRequest:
    prompt = DECOMPOSE_PROMPT.format(question=question)
    return InferenceRequest(
        request_id=_request_id("decomposer", model, prompt),
        role="decomposer",
        model=model,
        prompt=prompt,
        max_tokens=128,
    )


def _make_answer_request(question: str, model: str) -> InferenceRequest:
    prompt = ANSWER_PROMPT.format(question=question)
    return InferenceRequest(
        request_id=_request_id("answerer", model, prompt),
        role="answerer",
        model=model,
        prompt=prompt,
        max_tokens=32,
    )


def _consistency_from_cache(
    questions: Sequence[str],
    direct_answers: Sequence[str],
    answerer_model: str,
    decomposer_model: str,
    cache: Mapping[str, str],
    f1_threshold: float = 0.5,
) -> float:
    """Pure-computation version of ``compute_consistency_for_run``.

    Reads decomposition/Q1/Q2 completions from ``cache`` instead of calling an
    LLM. A missing cache entry is treated as a failed step, which makes the
    instance count as inconsistent — same policy as the dynamic path.
    """
    n_consistent = 0
    n_total = 0

    for question, direct in zip(questions, direct_answers):
        n_total += 1

        decomp_req = _make_decompose_request(question, decomposer_model)
        raw_decomp = cache.get(decomp_req.request_id, "")
        decomp = _parse_decomposition(raw_decomp)
        if decomp is None:
            continue

        q1, q2_tmpl = decomp

        q1_req = _make_answer_request(q1, answerer_model)
        a1 = _clean_answer(cache.get(q1_req.request_id, ""))
        if not a1:
            continue

        q2 = q2_tmpl.replace("[A1]", a1)
        q2_req = _make_answer_request(q2, answerer_model)
        expansion = _clean_answer(cache.get(q2_req.request_id, ""))
        if not expansion:
            continue

        if _token_f1(direct, expansion) >= f1_threshold:
            n_consistent += 1

    if n_total == 0:
        return 0.0
    return n_consistent / n_total


# ── Scenario-state helpers ────────────────────────────────────────────────────

# Candidate column names for question text and model completion in HELM's
# flattened scenario_state DataFrame.  We try each in order and use the first
# that exists.
_QUESTION_COLS = [
    "scenario_state.request_states.instance.input.text",
    "instance.input.text",
    "input.text",
]
_COMPLETION_COLS = [
    "scenario_state.request_states.result.completions",
    "result.completions",
    "completions",
]
_PROMPT_COLS = [
    "scenario_state.request_states.request.prompt",
    "request.prompt",
    "prompt",
]


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _extract_completion_text(val) -> str:
    """Extract text from a HELM completions value (list of dicts or string)."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list) and val:
        first = val[0]
        if isinstance(first, dict):
            return first.get("text", "").strip()
        return str(first).strip()
    return ""


def _extract_questions_and_answers(df) -> tuple[list[str], list[str]]:
    """Extract (questions, direct_answers) from a scenario_state DataFrame."""
    question_col = _pick_col(df, _QUESTION_COLS)
    completion_col = _pick_col(df, _COMPLETION_COLS)

    questions = []
    direct_answers = []

    if question_col is None:
        log.warning("Could not find question column in scenario_state. "
                    "Available: %s", list(df.columns)[:10])
        return [], []

    for _, row in df.iterrows():
        q = str(row[question_col]).strip() if question_col else ""
        if not q:
            continue

        if completion_col is not None:
            ans = _extract_completion_text(row[completion_col])
        else:
            ans = ""

        questions.append(q)
        direct_answers.append(ans)

    return questions, direct_answers


# ── Linear calibration ────────────────────────────────────────────────────────

class _LinearCalibration:
    """
    Calibration from (consistency, accuracy) pairs drawn from training runs:

    - 0 points: identity (accuracy = consistency).
    - 1 point:  linear through the origin -- slope = accuracy / consistency,
                intercept = 0.
    - 2+ points: affine OLS fit (slope and intercept both free).
    """

    def __init__(self, slope: float = 1.0, intercept: float = 0.0):
        self.slope = slope
        self.intercept = intercept
        self.n_points = 0

    @classmethod
    def fit(cls, consistencies: list[float], accuracies: list[float]) -> "_LinearCalibration":
        """Fit from (consistency, accuracy) pairs; see class docstring for fallback rules."""
        n = len(consistencies)

        if n == 0:
            log.warning("No training runs available; falling back to identity (accuracy = consistency).")
            return cls(slope=1.0, intercept=0.0)

        if n == 1:
            c, a = consistencies[0], accuracies[0]
            slope = a / c if c != 0.0 else 1.0
            log.warning("Only 1 training run; using slope=%.3f, intercept=0.", slope)
            cal = cls(slope=float(slope), intercept=0.0)
            cal.n_points = 1
            return cal

        x = np.array(consistencies, dtype=float)
        y = np.array(accuracies, dtype=float)

        # Affine OLS: minimise sum((y - slope*x - intercept)^2)
        A = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        cal = cls(slope=float(slope), intercept=float(intercept))
        cal.n_points = n
        return cal

    def predict(self, consistency: float) -> float:
        pred = self.slope * consistency + self.intercept
        return float(np.clip(pred, 0.0, 1.0))


# ── Main predictor ────────────────────────────────────────────────────────────

try:
    from magnet.predictor import RunPredictor, RunPrediction
    from magnet.data_splits import TrainSplit, SequesteredTestSplit
    _MAGNET_AVAILABLE = True
except ImportError:
    _MAGNET_AVAILABLE = False
    # Define stubs so the module can still be imported (and tested) without
    # magnet installed. Both base classes accept arbitrary kwargs so the
    # predictor's constructor/predict paths still function in a fake harness.
    class RunPredictor:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class RunPrediction:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class TrainSplit:  # type: ignore[no-redef]
        pass
    class SequesteredTestSplit:  # type: ignore[no-redef]
        pass


class OperadicConsistencyPredictor(RunPredictor):
    """
    Predict HELM run accuracy using operadic consistency as a signal.

    The predictor performs all LLM calls through swappable ``LLMBackend``
    objects, so the TA2 harness can substitute its own inference setup
    (e.g. a vllm server) without touching this class.

    Per-run vs fixed answerer
    -------------------------
    Consistency of a model X is, strictly speaking, a within-X property:
    "does X's factored answer match X's direct answer?" That means the
    answerer backend should match the model that produced each run's direct
    answers. Pass ``answerer_factory`` to enable per-run answerer resolution;
    the factory is called with the model ID extracted from each run's
    ``run_spec.name`` and should return an ``LLMBackend`` bound to that model.

    If no factory is given, the predictor falls back to the single
    ``answerer`` backend for every run (backward compatible — but note this
    mixes models, so the resulting signal is really "does X's direct answer
    match some reference model's factored answer," not pure self-consistency).

    Parameters
    ----------
    answerer:
        Default ``LLMBackend`` used to answer sub-questions when no
        ``answerer_factory`` is provided. Can be ``None`` if a factory is
        supplied.
    decomposer:
        ``LLMBackend`` used to decompose questions into sub-questions.
        Defaults to ``answerer`` if omitted. Always fixed across runs —
        the decomposer is part of the evaluation methodology, not the thing
        being evaluated.
    answerer_factory:
        Optional ``Callable[[str], LLMBackend]``. If given, called once per
        run with the model ID extracted from ``run_spec.name`` and expected
        to return a backend bound to that model. Enables per-run
        self-consistency evaluation.
    answerer_model, together_api_key, decomposer_model:
        Legacy kwargs. Provide these to auto-construct ``TogetherBackend``
        instances. Ignored if ``answerer`` is passed explicitly.
    stat_name:
        The HELM metric to predict (default: ``"exact_match"``).
    stat_split:
        The HELM split to predict (default: ``"valid"``).
    f1_threshold:
        Token F1 threshold above which an instance is considered consistent
        (default: ``0.5``).
    n_consistency_samples:
        Max number of instances per run to use for consistency estimation.
        Capped by ``num_eval_samples`` from the parent class.
    num_example_runs:
        Number of HELM runs to use for calibration (passed to base class).
    num_eval_samples:
        Number of instances sampled from each run (passed to base class).
    random_seed:
        RNG seed for sampling (passed to base class).
    """

    def __init__(
        self,
        answerer: Optional[LLMBackend] = None,
        decomposer: Optional[LLMBackend] = None,
        *,
        answerer_factory: Optional[Callable[[str], LLMBackend]] = None,
        answerer_model: Optional[str] = None,
        together_api_key: Optional[str] = None,
        decomposer_model: Optional[str] = None,
        stat_name: str = "exact_match",
        stat_split: str = "valid",
        f1_threshold: float = 0.5,
        n_consistency_samples: int = 20,
        num_example_runs: int = 3,
        num_eval_samples: int = 20,
        random_seed: int = 1,
    ):
        super().__init__(
            num_example_runs=num_example_runs,
            num_eval_samples=num_eval_samples,
            random_seed=random_seed,
        )

        # Resolve default answerer. Prefer explicit backend; fall back to
        # legacy (model, api_key) kwargs. If neither is given AND a factory
        # is provided, the default answerer stays None and every run must
        # route through the factory.
        if answerer is None and answerer_model is not None and together_api_key is not None:
            answerer = TogetherBackend(model=answerer_model, api_key=together_api_key)

        if answerer is None and answerer_factory is None:
            raise ValueError(
                "Must provide at least one of: `answerer`, `answerer_factory`, "
                "or `(answerer_model, together_api_key)`."
            )

        # Resolve decomposer. It stays fixed across runs.
        if decomposer is None:
            if decomposer_model is not None and together_api_key is not None:
                decomposer = TogetherBackend(
                    model=decomposer_model, api_key=together_api_key
                )
            elif answerer is not None:
                decomposer = answerer  # default: reuse default answerer
            else:
                raise ValueError(
                    "Cannot default decomposer to answerer when only "
                    "`answerer_factory` is provided — pass `decomposer` "
                    "(or `decomposer_model` + `together_api_key`) explicitly."
                )

        self.answerer: Optional[LLMBackend] = answerer
        self.answerer_factory: Optional[Callable[[str], LLMBackend]] = answerer_factory
        self.decomposer: LLMBackend = decomposer
        self.stat_name = stat_name
        self.stat_split = stat_split
        self.f1_threshold = f1_threshold
        self.n_consistency_samples = n_consistency_samples
        # Also set random_seed on self so helpers work even when MAGNET isn't
        # installed (in which case the parent is a stub that ignores kwargs).
        self.random_seed = random_seed

    # ── Internal helpers ──────────────────────────────────────────────────────

    # ── Per-run model resolution ──────────────────────────────────────────────

    _MODEL_RE = re.compile(r"model=([^,\s]+)")

    def _extract_model_id(self, run_spec_name: str) -> str:
        """Extract the model ID from a HELM ``run_spec.name``.

        HELM convention: ``<scenario>:model=<model_id>,key2=value2,...``.
        Override this in a subclass for non-HELM naming conventions.
        """
        m = self._MODEL_RE.search(run_spec_name)
        return m.group(1).strip() if m else "unknown-model"

    def _answerer_for_run(self, run_spec_name: str) -> tuple[LLMBackend, str]:
        """Resolve (answerer_backend, answerer_model_id) for a given run.

        - If ``answerer_factory`` is set, calls ``factory(model_id)`` where
          ``model_id`` comes from :meth:`_extract_model_id`. This is the
          per-run self-consistency path.
        - Otherwise falls back to the default ``self.answerer`` and reports
          its own ``.model`` attribute (the legacy fixed-answerer path).
        """
        if self.answerer_factory is not None:
            model_id = self._extract_model_id(run_spec_name)
            return self.answerer_factory(model_id), model_id
        if self.answerer is None:
            raise RuntimeError(
                f"No answerer available for run '{run_spec_name}': "
                "neither an `answerer` nor an `answerer_factory` is configured."
            )
        return self.answerer, getattr(self.answerer, "model", "unknown-model")

    def _answerer_model_for_run(self, run_spec_name: str) -> str:
        """Return just the answerer model ID for a run (no backend construction).

        Used by the batched path to populate ``InferenceRequest.model``.
        """
        if self.answerer_factory is not None:
            return self._extract_model_id(run_spec_name)
        if self.answerer is not None:
            return getattr(self.answerer, "model", "unknown-model")
        raise RuntimeError(
            f"No answerer configured for run '{run_spec_name}'."
        )

    def _sampled_questions(self, scenario_df) -> tuple[list[str], list[str]]:
        """Deterministically sample ``n_consistency_samples`` questions from a run.

        Uses ``np.random.default_rng(self.random_seed)`` so the same indices are
        picked every time the method is called with the same DataFrame. This is
        essential for the batched path — planning and prediction must agree on
        which questions they're looking at so the request IDs line up.
        """
        questions, directs = _extract_questions_and_answers(scenario_df)
        if not questions:
            return [], []

        n = min(self.n_consistency_samples, len(questions))
        rng = np.random.default_rng(self.random_seed)
        idx = rng.choice(len(questions), size=n, replace=False)
        return [questions[i] for i in idx], [directs[i] for i in idx]

    def _consistency_for_run(self, run_spec_name: str, scenario_df) -> float:
        """Compute consistency rate for one run (dynamic path).

        Uses :meth:`_answerer_for_run` so a per-run answerer backend is picked
        up when ``answerer_factory`` is configured. Otherwise falls back to
        the default fixed answerer.
        """
        questions, directs = self._sampled_questions(scenario_df)
        if not questions:
            log.warning("No questions extracted from scenario_state for '%s'.", run_spec_name)
            return 0.0

        answerer, _ = self._answerer_for_run(run_spec_name)

        return compute_consistency_for_run(
            questions=questions,
            direct_answers=directs,
            answerer=answerer,
            decomposer=self.decomposer,
            f1_threshold=self.f1_threshold,
        )

    # ── Batched inference path ────────────────────────────────────────────────

    def plan_next_batch(
        self,
        train_split,
        sequestered_test_split,
        cache: Mapping[str, str],
    ) -> list[InferenceRequest]:
        """Return the next batch of unfulfilled LLM requests, or ``[]`` when done.

        The batched flow has up to three sequential phases:

        1. **Decompose** each sampled question (decomposer model — fixed).
        2. **Answer Q1** for each successful decomposition, using the
           *per-run* answerer model (whichever model produced that run's
           direct answers).
        3. **Answer Q2** with ``[A1]`` filled from the cached Q1 result,
           also using the per-run answerer model.

        The caller is expected to execute each returned batch, merge the
        results into ``cache`` (``{request_id: completion_text}``), and call
        this method again until it returns ``[]``.

        Requests are deduplicated by ``request_id`` (a hash of
        ``(role, model, prompt)``) so asking the same triple twice — e.g.
        two runs that happen to use the same answerer model — only costs
        one inference call.
        """
        decomposer_model = getattr(self.decomposer, "model", "unknown-decomposer")

        # Enumerate (run_name, questions, answerer_model) once so we can
        # resolve per-run answerer models without repeating work.
        runs_with_models = []
        for run_name, grp in train_split.scenario_state.groupby("run_spec.name"):
            qs, _ = self._sampled_questions(grp)
            runs_with_models.append((run_name, qs, self._answerer_model_for_run(run_name)))
        for run_name, grp in sequestered_test_split.scenario_state.groupby("run_spec.name"):
            qs, _ = self._sampled_questions(grp)
            runs_with_models.append((run_name, qs, self._answerer_model_for_run(run_name)))

        # ── Phase 1: decomposition (fixed decomposer; dedup by question) ──────
        decompose_reqs: dict[str, InferenceRequest] = {}
        for _, questions, _ in runs_with_models:
            for q in questions:
                req = _make_decompose_request(q, decomposer_model)
                if req.request_id not in cache:
                    decompose_reqs[req.request_id] = req
        if decompose_reqs:
            return list(decompose_reqs.values())

        # ── Phase 2: answer Q1 (per-run answerer model) ───────────────────────
        q1_reqs: dict[str, InferenceRequest] = {}
        for _, questions, answerer_model in runs_with_models:
            for q in questions:
                decomp_req = _make_decompose_request(q, decomposer_model)
                decomp = _parse_decomposition(cache.get(decomp_req.request_id, ""))
                if decomp is None:
                    continue
                q1, _ = decomp
                req = _make_answer_request(q1, answerer_model)
                if req.request_id not in cache:
                    q1_reqs[req.request_id] = req
        if q1_reqs:
            return list(q1_reqs.values())

        # ── Phase 3: answer Q2 with [A1] substituted (per-run answerer) ───────
        q2_reqs: dict[str, InferenceRequest] = {}
        for _, questions, answerer_model in runs_with_models:
            for q in questions:
                decomp_req = _make_decompose_request(q, decomposer_model)
                decomp = _parse_decomposition(cache.get(decomp_req.request_id, ""))
                if decomp is None:
                    continue
                q1, q2_tmpl = decomp
                a1_req = _make_answer_request(q1, answerer_model)
                a1 = _clean_answer(cache.get(a1_req.request_id, ""))
                if not a1:
                    continue
                q2 = q2_tmpl.replace("[A1]", a1)
                req = _make_answer_request(q2, answerer_model)
                if req.request_id not in cache:
                    q2_reqs[req.request_id] = req
        if q2_reqs:
            return list(q2_reqs.values())

        return []

    def predict_from_cache(
        self,
        train_split,
        sequestered_test_split,
        cache: Mapping[str, str],
    ) -> list:
        """Batched counterpart to ``predict``: compute predictions from the
        filled-in ``cache``, making no LLM calls.

        Assumes ``plan_next_batch`` has been run to completion and all its
        returned requests have been executed by the harness.
        """
        decomposer_model = getattr(self.decomposer, "model", "unknown-decomposer")

        # ── Step 1: calibrate from training runs ─────────────────────────────
        consistencies: list[float] = []
        accuracies: list[float] = []

        for run_spec_name, grp in train_split.scenario_state.groupby("run_spec.name"):
            log.info("Reading cached consistency for training run: %s", run_spec_name)
            questions, directs = self._sampled_questions(grp)
            answerer_model = self._answerer_model_for_run(run_spec_name)
            consistency = _consistency_from_cache(
                questions=questions,
                direct_answers=directs,
                answerer_model=answerer_model,
                decomposer_model=decomposer_model,
                cache=cache,
                f1_threshold=self.f1_threshold,
            )
            accuracy = self._accuracy_from_stats(train_split.stats, run_spec_name)
            if accuracy is not None:
                consistencies.append(consistency)
                accuracies.append(accuracy)
                log.info("  consistency=%.3f  accuracy=%.3f", consistency, accuracy)

        calibration = _LinearCalibration.fit(consistencies, accuracies)
        log.info(
            "Calibration: accuracy ≈ %.3f * consistency + %.3f  (n=%d)",
            calibration.slope, calibration.intercept, calibration.n_points,
        )

        # ── Step 2: predict for sequestered test runs ────────────────────────
        predictions = []
        for run_spec_name, grp in sequestered_test_split.scenario_state.groupby(
            "run_spec.name"
        ):
            questions, directs = self._sampled_questions(grp)
            answerer_model = self._answerer_model_for_run(run_spec_name)
            consistency = _consistency_from_cache(
                questions=questions,
                direct_answers=directs,
                answerer_model=answerer_model,
                decomposer_model=decomposer_model,
                cache=cache,
                f1_threshold=self.f1_threshold,
            )
            predicted_accuracy = calibration.predict(consistency)
            log.info(
                "  %s: consistency=%.3f  predicted_accuracy=%.3f",
                run_spec_name, consistency, predicted_accuracy,
            )
            predictions.append(
                RunPrediction(
                    run_spec_name=run_spec_name,
                    split=self.stat_split,
                    stat_name=self.stat_name,
                    mean=predicted_accuracy,
                )
            )

        return predictions

    def _accuracy_from_stats(self, stats_df, run_spec_name: str) -> Optional[float]:
        """Extract the target accuracy metric from a stats DataFrame."""
        mask = (
            (stats_df["run_spec.name"] == run_spec_name)
            & (stats_df["stats.name.name"] == self.stat_name)
            & (stats_df["stats.name.split"] == self.stat_split)
        )
        subset = stats_df[mask]
        if subset.empty:
            # Try without split filter
            mask2 = (
                (stats_df["run_spec.name"] == run_spec_name)
                & (stats_df["stats.name.name"] == self.stat_name)
            )
            subset = stats_df[mask2]

        if subset.empty:
            log.warning("No '%s' stat found for run '%s'.", self.stat_name, run_spec_name)
            return None

        return float(subset["stats.mean"].iloc[0])

    # ── MAGNET interface ──────────────────────────────────────────────────────

    def predict(
        self,
        train_split: TrainSplit,
        sequestered_test_split: SequesteredTestSplit,
    ) -> list:
        """
        Compute operadic consistency for training runs, calibrate, then
        predict accuracy for the test run.

        Returns
        -------
        list[RunPrediction]
        """
        # ── Step 1: Build calibration from training runs ──────────────────────
        consistencies = []
        accuracies = []

        for run_spec_name, grp in train_split.scenario_state.groupby("run_spec.name"):
            log.info("Computing consistency for training run: %s", run_spec_name)
            consistency = self._consistency_for_run(run_spec_name, grp)
            accuracy = self._accuracy_from_stats(train_split.stats, run_spec_name)

            if accuracy is not None:
                consistencies.append(consistency)
                accuracies.append(accuracy)
                log.info("  consistency=%.3f  accuracy=%.3f", consistency, accuracy)

        calibration = _LinearCalibration.fit(consistencies, accuracies)
        log.info(
            "Calibration: accuracy ≈ %.3f * consistency + %.3f  (n=%d)",
            calibration.slope, calibration.intercept, calibration.n_points,
        )

        # ── Step 2: Predict for test runs ─────────────────────────────────────
        predictions = []

        for run_spec_name, grp in sequestered_test_split.scenario_state.groupby(
            "run_spec.name"
        ):
            log.info("Computing consistency for test run: %s", run_spec_name)
            consistency = self._consistency_for_run(run_spec_name, grp)
            predicted_accuracy = calibration.predict(consistency)
            log.info(
                "  consistency=%.3f  predicted_accuracy=%.3f",
                consistency, predicted_accuracy,
            )

            predictions.append(
                RunPrediction(
                    run_spec_name=run_spec_name,
                    split=self.stat_split,
                    stat_name=self.stat_name,
                    mean=predicted_accuracy,
                )
            )

        return predictions
