# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (miniforge)
#     language: python
#     name: miniforge-base
# ---

# %%
# Dev setup
# %load_ext autoreload
# %autoreload 2

# %%
from operadic_consistency.core.interfaces import Answer
from operadic_consistency.core.metrics import (
    answer_distribution,
    mode_answer,
    agreement_rate,
    shannon_entropy,
    inconsistency_witnesses,
    summarize_report,
)


# %%
# ---- tests for core/metrics.py ----

def expect_ok(fn, msg=""):
    try:
        fn()
        print("passed", msg or fn.__name__)
    except Exception as e:
        print("FAILED", msg or fn.__name__, "->", type(e).__name__, e)

# We’ll fabricate a tiny ConsistencyReport with known answers.
# (This avoids needing to run the whole pipeline.)

def test_distribution_and_mode_and_agreement():
    # Fake objects (minimal fields accessed by metrics.py)
    class FakePlan:
        def __init__(self, cut_edges): self.cut_edges = tuple(cut_edges)

    class FakeRun:
        def __init__(self, raw, norm, cut_edges):
            self.root_answer = Answer(text=raw)
            self.normalized_root = norm
            self.plan = FakePlan(cut_edges)

    class FakeReport:
        def __init__(self, runs):
            self.runs = runs

    runs = [
        FakeRun("YES", "yes", ()),
        FakeRun("Yes ", "yes", (1,)),
        FakeRun("NO",  "no",  (2,)),
        FakeRun("YES", "yes", (1,2)),
    ]
    rep = FakeReport(runs)

    # normalized distribution
    dist = answer_distribution(rep, use_normalized=True)
    assert dist == {"yes": 3, "no": 1}

    m = mode_answer(rep, use_normalized=True)
    assert m == ("yes", 3)

    ar = agreement_rate(rep, use_normalized=True)
    assert abs(ar - 0.75) < 1e-9

    # raw distribution (no normalization)
    dist_raw = answer_distribution(rep, use_normalized=False)
    assert dist_raw["YES"] == 2
    assert dist_raw["NO"] == 1
    assert dist_raw["Yes "] == 1


def test_entropy_and_witnesses_and_summary():
    class FakePlan:
        def __init__(self, cut_edges): self.cut_edges = tuple(cut_edges)

    class FakeRun:
        def __init__(self, raw, norm, cut_edges):
            self.root_answer = Answer(text=raw)
            self.normalized_root = norm
            self.plan = FakePlan(cut_edges)

    class FakeReport:
        def __init__(self, runs):
            self.runs = runs

    runs = [
        FakeRun("A", "a", ()),
        FakeRun("A", "a", (1,)),
        FakeRun("B", "b", (2,)),
        FakeRun("B", "b", (3,)),
    ]
    rep = FakeReport(runs)

    dist = answer_distribution(rep, use_normalized=True)
    ent = shannon_entropy(dist)
    # For a 50/50 split, entropy should be log(2) in natural units
    import math
    assert abs(ent - math.log(2)) < 1e-9

    wit = inconsistency_witnesses(rep, use_normalized=True, max_per_answer=1)
    assert set(wit.keys()) == {"a", "b"}
    assert len(wit["a"]) == 1
    assert len(wit["b"]) == 1

    summ = summarize_report(rep, use_normalized=True, top_k=5, max_witnesses_per_answer=2)
    assert summ["num_runs"] == 4
    assert summ["num_unique_answers"] == 2
    assert summ["mode_fraction"] == 0.5
    assert summ["mode_answer"] in ("a", "b")
    assert isinstance(summ["top_answers"], list)
    assert "witness_plans" in summ


expect_ok(test_distribution_and_mode_and_agreement, "distribution/mode/agreement")
expect_ok(test_entropy_and_witnesses_and_summary, "entropy/witnesses/summary")
print("metrics.py tests done")

# %%
