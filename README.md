# Operadic Consistency

A MAGNET-compatible accuracy predictor for the DARPA AIQ evaluation.

`operadic_consistency` provides `OperadicConsistencyPredictor`, a
`magnet.RunPredictor` that estimates a language model's benchmark accuracy
without ground-truth labels by using the model's *operadic consistency* as a
reliability signal. All inference is routed through a swappable backend
protocol, so the TA2 harness can supply its own LLM endpoint (vLLM, Together,
mock) and either drive the predictor in a single batched precompute pass or
let it make calls dynamically.

## The signal

Given a multi-hop question Q and a model M that has already produced a direct
answer to Q, we:

1. Decompose Q into two sub-questions Q1 and Q2 (where Q2 depends on the
   answer to Q1 via an `[A1]` placeholder).
2. Ask M to answer Q1, getting A1.
3. Substitute A1 into Q2 and ask M to answer the resulting question, getting
   an *expansion* answer.
4. Compare the expansion answer to M's direct answer (token F1 ≥ threshold).

The fraction of sampled instances where expansion and direct agree is M's
*consistency* on that benchmark. Empirically, consistency and accuracy are
strongly linearly related across models and datasets; the predictor fits
`accuracy ≈ slope * consistency + intercept` from calibration runs with known
accuracy and applies it to held-out runs.

## Install

```bash
pip install -e ".[magnet]"
```

This installs `numpy`, `together`, and `magnet` (from the AIQ-Kitware repo).
Dev extras (`pip install -e ".[dev]"`) give you `pytest` for the test suite.

## MAGNET integration

### Construct the predictor

The predictor needs an **answerer** backend (or a factory that produces one
per run) and a **decomposer** backend. Backends are `LLMBackend`-typed
objects with a single `complete(prompt, *, max_tokens, temperature, stop)`
method — Kitware can implement their own by defining that method on any
class.

```python
from operadic_consistency.magnet import (
    OperadicConsistencyPredictor,
    TogetherBackend,
)

# Fixed answerer for all runs (simplest case)
predictor = OperadicConsistencyPredictor(
    answerer=TogetherBackend(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                             api_key=TOGETHER_KEY),
    decomposer=TogetherBackend(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                               api_key=TOGETHER_KEY),
    n_consistency_samples=20,
    num_example_runs=5,
    num_eval_samples=20,
)
```

### Per-run answerer (self-consistency)

Consistency of a model X is really a within-X property: "does X's factored
answer match X's direct answer?" To measure this properly, the answerer must
match the model that produced each run's direct answers. Pass
`answerer_factory` to resolve a backend per run:

```python
def factory(model_id: str) -> LLMBackend:
    # Called with the model ID parsed from run_spec.name
    return TogetherBackend(model=model_id, api_key=TOGETHER_KEY)

predictor = OperadicConsistencyPredictor(
    answerer_factory=factory,
    decomposer=TogetherBackend(model="reference-decomposer",
                               api_key=TOGETHER_KEY),
)
```

The decomposer stays fixed across runs — it is part of the evaluation
methodology, not the system being evaluated.

### Batched harness flow (recommended)

The predictor's preferred integration is batched precompute: the harness
owns all LLM calls and runs them in a small number of large batches. Between
batches, the predictor tells the harness exactly which prompts it needs next.

```python
cache: dict[str, str] = {}   # request_id → completion text

while True:
    batch = predictor.plan_next_batch(train_split, test_split, cache)
    if not batch:
        break
    # harness.run_batch runs every InferenceRequest in the batch however it
    # wants (vLLM, Together, etc.) and returns {request_id: text}
    cache.update(harness.run_batch(batch))

predictions = predictor.predict_from_cache(train_split, test_split, cache)
```

`plan_next_batch` walks up to three sequential phases:

1. **Decompose** each sampled question via the (fixed) decomposer.
2. **Answer Q1** for every successful decomposition, using the per-run
   answerer model.
3. **Answer Q2** with `[A1]` substituted from the cached Q1 answers,
   using the per-run answerer model.

Each `InferenceRequest` exposes `request_id`, `role` (`"decomposer"` or
`"answerer"`), `model`, `prompt`, and standard inference knobs
(`max_tokens`, `temperature`). `request_id` is a deterministic hash of
`(role, model, prompt)`, so asking the same triple from two runs costs only
one inference call.

### Dynamic flow (fallback)

If batching is impractical, the predictor can drive the LLM calls itself
through the configured backends:

```python
predictions = predictor.predict(train_split, sequestered_test_split)
```

Semantically identical to the batched path, but all inference happens
serially inside the predictor.

### Driver script

There is no registration system. Write a small driver and point it at a
HELM output directory — `RunPredictor.__call__` (inherited from MAGNET)
handles loading runs, splitting into training and test, and invoking
`predict`.

```python
# run_predictor.py
import argparse
from operadic_consistency.magnet import OperadicConsistencyPredictor, TogetherBackend

predictor = OperadicConsistencyPredictor(
    answerer=TogetherBackend(model="...", api_key="..."),
    decomposer=TogetherBackend(model="...", api_key="..."),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_path",
                        help="Path to HELM benchmark_output/runs/<suite>")
    args = parser.parse_args()
    predictor(args.suite_path)
```

```bash
python run_predictor.py path/to/benchmark_output/runs/my_suite
```

### Calibration fallback behavior

- **2+ training runs**: affine OLS fit (slope and intercept both free).
- **1 training run**: linear through the origin (slope = accuracy /
  consistency, intercept = 0).
- **0 training runs**: identity (predicted accuracy = consistency).

## Core framework

Under the MAGNET predictor sits a small research framework for evaluating
reasoning robustness via structured question decompositions. The central
object is a **Tree of Questions (ToQ)** — a tree whose nodes are
sub-questions with `[A_i]` placeholders referring to child answers. The
framework lets you systematically collapse subtrees into single questions,
re-evaluate, and compare answers. The MAGNET predictor is a specific
instantiation that uses a 2-node chain (Q1 → Q2) and token F1 agreement as
the consistency metric.

### Minimal ToQ example

```python
from operadic_consistency import run_consistency_check
from operadic_consistency.core.toq_types import ToQ, ToQNode
from operadic_consistency.core.interfaces import Answer


class TinyAnswerer:
    def __call__(self, question: str, *, context=None) -> Answer:
        q = question.lower()
        if "when did ww2 end" in q:
            return Answer("1945")
        if "president" in q:
            return Answer("Harry Truman")
        return Answer("UNKNOWN")


class TinyCollapser:
    def __call__(self, open_toq, *, context=None) -> str:
        return open_toq.toq.nodes[open_toq.root_id].text


nodes = {
    1: ToQNode(1, "When did WW2 end?", parent=2),
    2: ToQNode(2, "Who was President at time [A1]?", parent=None),
}
toq = ToQ(nodes=nodes, root_id=2)

report = run_consistency_check(
    toq,
    answerer=TinyAnswerer(),
    collapser=TinyCollapser(),
)
print("Baseline root answer:", report.base_root_answer.text)
```

See [docs/](docs/) for the full core API (`ToQ`, `OpenToQ`, `Collapser`,
`QuestionDecomposer`, `run_consistency_check`,
`run_consistency_check_from_question`).

## Project layout

```text
.
├── operadic_consistency/
│   ├── core/               # ToQ types, evaluation, metrics, serialization
│   ├── magnet/             # MAGNET predictor, LLMBackend, TogetherBackend
│   └── model_interface/    # (reserved for future non-MAGNET integrations)
├── tests/                  # pytest suite
├── docs/                   # usage docs
├── examples/               # runnable examples
├── pyproject.toml
└── README.md
```

## Quickstart

```bash
pip install -e ".[dev]"   # or ".[magnet]" for the predictor + deps
pytest                    # run tests
python examples/minimal_consistency.py
```

## Docs

- [docs/index.md](docs/index.md)
- [docs/getting_started.md](docs/getting_started.md)
- [docs/api_usage.md](docs/api_usage.md)

## Status

Research prototype under active development for the DARPA AIQ May smoketest.
The MAGNET-facing API (`LLMBackend`, `plan_next_batch`, `predict_from_cache`,
`answerer_factory`) is stable enough to build against; the core ToQ
framework API may still evolve.
