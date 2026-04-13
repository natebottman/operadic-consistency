import math

from operadic_consistency.core.interfaces import Answer
from operadic_consistency.core.metrics import (
    answer_distribution,
    mode_answer,
    agreement_rate,
    shannon_entropy,
    inconsistency_witnesses,
    summarize_report,
)


# ── Minimal fakes ─────────────────────────────────────────────────────────────

class _Plan:
    def __init__(self, cut_edges):
        self.cut_edges = tuple(cut_edges)


class _Run:
    def __init__(self, raw, norm, cut_edges):
        self.root_answer = Answer(text=raw)
        self.normalized_root = norm
        self.plan = _Plan(cut_edges)


class _Report:
    def __init__(self, runs):
        self.runs = runs


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_distribution_mode_agreement():
    runs = [
        _Run("YES", "yes", ()),
        _Run("Yes ", "yes", (1,)),
        _Run("NO",  "no",  (2,)),
        _Run("YES", "yes", (1, 2)),
    ]
    rep = _Report(runs)

    dist = answer_distribution(rep, use_normalized=True)
    assert dist == {"yes": 3, "no": 1}

    assert mode_answer(rep, use_normalized=True) == ("yes", 3)
    assert abs(agreement_rate(rep, use_normalized=True) - 0.75) < 1e-9

    dist_raw = answer_distribution(rep, use_normalized=False)
    assert dist_raw["YES"] == 2
    assert dist_raw["NO"] == 1
    assert dist_raw["Yes "] == 1


def test_entropy_witnesses_summary():
    runs = [
        _Run("A", "a", ()),
        _Run("A", "a", (1,)),
        _Run("B", "b", (2,)),
        _Run("B", "b", (3,)),
    ]
    rep = _Report(runs)

    dist = answer_distribution(rep, use_normalized=True)
    ent = shannon_entropy(dist)
    # 50/50 split → entropy = log(2) nats
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
