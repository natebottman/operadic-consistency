# Operadic Consistency

Operadic Consistency is a lightweight research framework for evaluating
reasoning robustness of language models via **structured question
decompositions**.

The central object is a **Tree of Questions (ToQ)**.
We evaluate whether collapsing subtrees into single questions preserves
the final answer.

## Project Layout

```text
.
├── operadic_consistency/
│   ├── core/               # ToQ types, evaluation, metrics, serialization
│   ├── magnet/             # DARPA AIQ MAGNET predictor (optional)
│   └── model_interface/    # LLM decomposer/answerer wrappers
├── tests/                  # pytest suite
├── docs/                   # usage docs
├── examples/               # runnable examples
├── pyproject.toml
└── README.md
```

## Core Idea

A complex question is decomposed into smaller subquestions arranged as a tree.

We then:

1. Evaluate the full tree (baseline).
2. Systematically collapse subtrees into single questions.
3. Re-evaluate.
4. Compare answers.

If the answer changes under valid collapses, reasoning may be brittle.

## Visual Example

Consider:

> Who was President when WW2 ended?

### Original ToQ

```text
      (2) Who was President at time [A1]?
           |
      (1) When did WW2 end?
```

Evaluation proceeds bottom-up:

-   Node 1 → "1945"
-   Node 2 → "Harry Truman"

Baseline answer: **Harry Truman**

### Collapse Run 1 (no cuts)

We collapse the entire tree into a single question:

```text
(2) Who was President when WW2 ended?
```

Answer: **Harry Truman**

### Collapse Run 2 (cut edge 1)

Keep the leaf separate and collapse the root component:

```text
      (2) Who was President at time [A1]?
           |
      (1) When did WW2 end?
```

This produces the same structure as the original tree.

Answer: **Harry Truman**

If collapsing changes the root answer, reasoning may be inconsistent.

## Core Concepts

### ToQ (Tree of Questions)

A `ToQ` represents a structured decomposition of a question.

Each node:

- Has a question string
- May contain placeholders like `[A1]`, `[A2]`
- Refers to answers of child nodes

Example:

```text
    When did WW2 end?        (node 1)
    Who was President at time [A1]?   (node 2, root)
```

### OpenToQ

An `OpenToQ` is a ToQ with explicit external inputs.
It represents a component extracted during partial collapse.

### Collapser

A `Collapser` maps an `OpenToQ` to a single question.

### Answerer

An `Answerer` maps a fully-instantiated question string to an `Answer`.

### Decomposer

A `QuestionDecomposer` maps a raw question string to a `ToQ`.

## Public API

### `run_consistency_check`

```python
run_consistency_check(
    toq: ToQ,
    *,
    answerer: Answerer,
    collapser: Collapser,
    normalizer=None,
    substituter=None,
    context=None,
    plan_opts=None,
    cache=None,
) -> ConsistencyReport
```

Run the operadic consistency check from a manually constructed `ToQ`.

### `run_consistency_check_from_question`

```python
run_consistency_check_from_question(
    question: str,
    *,
    decomposer: QuestionDecomposer,
    answerer: Answerer,
    collapser: Collapser,
    normalizer=None,
    substituter=None,
    context=None,
    plan_opts=None,
    cache=None,
) -> ConsistencyReport
```

Run consistency checking from a raw question.
The provided decomposer constructs the initial `ToQ`.

## Minimal Usage Example

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

## Quickstart

Install package:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

Run minimal example:

```bash
python examples/minimal_consistency.py
```

## MAGNET Integration (DARPA AIQ)

`operadic_consistency.magnet` provides an `OperadicConsistencyPredictor` that
implements the MAGNET `RunPredictor` interface for the DARPA AIQ TA2 evaluation.

### Idea

Operadic consistency — whether a model's direct answer to a question agrees
with the answer it reaches by first answering sub-questions — correlates
strongly with accuracy.  The predictor exploits this to estimate accuracy on
models whose ground-truth labels are held out.

The caller specifies two groups of models:

- **Training runs** (`train_split`): models for which ground-truth accuracy is
  available.  The predictor computes operadic consistency for each, then fits a
  linear consistency → accuracy calibration from these pairs.
- **Test runs** (`sequestered_test_split`): models for which only raw outputs
  are available.  The predictor computes consistency and maps it to a predicted
  accuracy via the fitted calibration.

If fewer than 2 training runs are available, the predictor falls back to the
identity mapping (predicted accuracy = consistency rate).

### Install

```bash
pip install -e ".[magnet]"
```

This installs `together` (for LLM calls) and `magnet` from the AIQ-Kitware
repository.

### Usage

```python
from operadic_consistency.magnet import OperadicConsistencyPredictor

predictor = OperadicConsistencyPredictor(
    answerer_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    together_api_key="YOUR_KEY",
    n_consistency_samples=20,
)
# predictor.predict(train_split, sequestered_test_split) is called by the
# MAGNET harness automatically.
```

## Docs

- [docs/index.md](docs/index.md)
- [docs/getting_started.md](docs/getting_started.md)
- [docs/api_usage.md](docs/api_usage.md)

## Status

Research prototype. API may evolve.
