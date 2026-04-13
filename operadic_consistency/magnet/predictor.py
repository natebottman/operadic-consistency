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
4. **Calibration**: Fit a linear model (consistency → accuracy) using the
   training runs, which have both consistency scores (computed here) and
   ground-truth accuracy (provided by MAGNET's ``train_split.stats``).
   If fewer than 2 training runs are available, falls back to the identity
   mapping (accuracy ≈ consistency).
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
::

    from operadic_consistency.magnet import OperadicConsistencyPredictor

    predictor = OperadicConsistencyPredictor(
        answerer_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        together_api_key="YOUR_KEY",
        decomposer_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        num_eval_samples=20,
        n_consistency_samples=20,
    )
    predictor(helm_suites="path/to/benchmark_output/runs/suite_name")
"""
from __future__ import annotations

import re
import json
import logging
from typing import Optional, Sequence

import numpy as np

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

def _together_complete(
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """Call Together.ai completion API and return the generated text."""
    try:
        import together
        client = together.Together(api_key=api_key)
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n", "###"],
        )
        return resp.choices[0].text.strip()
    except Exception as e:
        log.warning("Together.ai call failed: %s", e)
        return ""


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


def _decompose(question: str, model: str, api_key: str) -> Optional[tuple[str, str]]:
    """Decompose a question into (q1, q2_template) or None on failure."""
    prompt = DECOMPOSE_PROMPT.format(question=question)
    raw = _together_complete(prompt, model, api_key, max_tokens=128)

    q1_m = re.search(r"Q1:\s*(.+?)(?:\n|$)", raw)
    q2_m = re.search(r"Q2:\s*(.+?)(?:\n|$)", raw)

    if not q1_m or not q2_m:
        return None

    q1 = q1_m.group(1).strip()
    q2 = q2_m.group(1).strip()

    # Validate that Q2 references [A1]
    if "[A1]" not in q2:
        q2 = q2 + " (given that [A1])"

    return q1, q2


def _answer(question: str, model: str, api_key: str) -> str:
    """Get a short answer to a question."""
    prompt = ANSWER_PROMPT.format(question=question)
    raw = _together_complete(prompt, model, api_key, max_tokens=32)
    return raw.strip().split("\n")[0].strip()


# ── Consistency computation ───────────────────────────────────────────────────

def compute_consistency_for_run(
    questions: Sequence[str],
    direct_answers: Sequence[str],
    answerer_model: str,
    api_key: str,
    decomposer_model: str,
    f1_threshold: float = 0.5,
) -> float:
    """
    For each (question, direct_answer) pair:
      1. Decompose question → (q1, q2_template)
      2. Answer q1 → a1
      3. Render q2 with a1 → answer q2 → expansion_answer
      4. consistent_i = token_f1(direct_answer, expansion_answer) >= threshold

    Returns fraction of consistent instances (0.0–1.0).
    Instances that fail to decompose are counted as inconsistent.
    """
    n_consistent = 0
    n_total = 0

    for question, direct in zip(questions, direct_answers):
        n_total += 1
        try:
            decomp = _decompose(question, decomposer_model, api_key)
            if decomp is None:
                log.debug("Decompose failed for: %s", question[:60])
                continue

            q1, q2_tmpl = decomp

            # Answer q1
            a1 = _answer(q1, answerer_model, api_key)
            if not a1:
                continue

            # Render q2 with a1
            q2 = q2_tmpl.replace("[A1]", a1)
            expansion = _answer(q2, answerer_model, api_key)
            if not expansion:
                continue

            if _token_f1(direct, expansion) >= f1_threshold:
                n_consistent += 1

        except Exception as e:
            log.warning("Consistency check error: %s", e)

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
    Linear map: accuracy ≈ slope * consistency + intercept.

    Fitted from (consistency, accuracy) pairs drawn from training runs:

    - 0 points: identity (accuracy = consistency).
    - 1 point:  slope=1, intercept chosen so the line passes through the
                single training point.  This honors the one anchor we have
                without trying to estimate the slope from a single observation.
    - 2+ points: OLS fit with intercept.
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
            # Slope=1, intercept passes through the single point.
            intercept = accuracies[0] - consistencies[0]
            log.warning(
                "Only 1 training run; using slope=1, intercept=%.3f "
                "(line through single point).",
                intercept,
            )
            cal = cls(slope=1.0, intercept=float(intercept))
            cal.n_points = 1
            return cal

        x = np.array(consistencies)
        y = np.array(accuracies)

        # OLS with intercept
        A = np.column_stack([x, np.ones_like(x)])
        result = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = result[0]

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
    # Define stubs so the module can still be imported without magnet installed
    class RunPredictor:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            pass
    class RunPrediction:  # type: ignore[no-redef]
        pass
    class TrainSplit:  # type: ignore[no-redef]
        pass
    class SequesteredTestSplit:  # type: ignore[no-redef]
        pass


class OperadicConsistencyPredictor(RunPredictor):
    """
    Predict HELM run accuracy using operadic consistency as a signal.

    Parameters
    ----------
    answerer_model:
        Together.ai model ID used to answer sub-questions during the
        consistency expansion.
    together_api_key:
        Together.ai API key.
    decomposer_model:
        Together.ai model ID used to decompose questions into sub-questions.
        Defaults to the same model as ``answerer_model``.
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
        answerer_model: str,
        together_api_key: str,
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
        self.answerer_model = answerer_model
        self.together_api_key = together_api_key
        self.decomposer_model = decomposer_model or answerer_model
        self.stat_name = stat_name
        self.stat_split = stat_split
        self.f1_threshold = f1_threshold
        self.n_consistency_samples = n_consistency_samples

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _consistency_for_df(self, scenario_df) -> float:
        """Compute consistency rate for a scenario_state DataFrame."""
        questions, directs = _extract_questions_and_answers(scenario_df)

        if not questions:
            log.warning("No questions extracted from scenario_state.")
            return 0.0

        # Subsample
        n = min(self.n_consistency_samples, len(questions))
        rng = np.random.default_rng(self.random_seed)
        idx = rng.choice(len(questions), size=n, replace=False)
        questions = [questions[i] for i in idx]
        directs = [directs[i] for i in idx]

        return compute_consistency_for_run(
            questions=questions,
            direct_answers=directs,
            answerer_model=self.answerer_model,
            api_key=self.together_api_key,
            decomposer_model=self.decomposer_model,
            f1_threshold=self.f1_threshold,
        )

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
            consistency = self._consistency_for_df(grp)
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
            consistency = self._consistency_for_df(grp)
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
