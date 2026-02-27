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
from typing import Optional

from operadic_consistency.core.toq_types import ToQ, ToQNode, OpenToQ
from operadic_consistency.core.interfaces import Answer

from operadic_consistency.core.consistency import run_consistency_check



# %%
# ---- tests for core/consistency.py ----

def expect_ok(fn, msg=""):
    try:
        fn()
        print("passed", msg or fn.__name__)
    except Exception as e:
        print("FAILED", msg or fn.__name__, "->", type(e).__name__, e)


class RecordingCollapser:
    """
    Collapser that records how many times it's called for each OpenToQ boundary.
    Cache keys should be determined by (root_id, inputs).
    """
    def __init__(self):
        self.calls = []   # list of (root_id, inputs_tuple, context)
        self.counts = {}  # (root_id, inputs_tuple) -> count

    def __call__(self, open_toq: OpenToQ, *, context: Optional[str] = None) -> str:
        key = (open_toq.root_id, tuple(open_toq.inputs))
        self.calls.append((open_toq.root_id, tuple(open_toq.inputs), context))
        self.counts[key] = self.counts.get(key, 0) + 1
        return f"COLLAPSED({open_toq.root_id}|inputs={list(open_toq.inputs)})"


class ToyAnswerer:
    def __init__(self):
        self.calls = []
    def __call__(self, question: str, *, context: Optional[str] = None) -> Answer:
        self.calls.append((question, context))
        return Answer(text=f"ANS({question})")


def test_runs_count_and_shapes():
    # Tree: 5 nodes, 4 edges
    #
    #        5 (root)
    #       / \
    #      3   4
    #     / \
    #    1   2
    #
    nodes = {
        1: ToQNode(1, "Q1?", parent=3),
        2: ToQNode(2, "Q2?", parent=3),
        3: ToQNode(3, "Q3([A1],[A2])", parent=5),
        4: ToQNode(4, "Q4?", parent=5),
        5: ToQNode(5, "Q5([A3],[A4])", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=5)
    toq.validate()

    answerer = ToyAnswerer()
    collapser = RecordingCollapser()

    rep = run_consistency_check(
        toq,
        answerer=answerer,
        collapser=collapser,
        plan_opts={"include_empty": True},
        cache={},  # explicit cache for determinism
    )

    # There are 4 edges => 2^4 = 16 plans => 16 runs
    assert len(rep.runs) == 2 ** (len(nodes) - 1)

    # Baseline root answer exists
    assert rep.base_root_answer.text.startswith("ANS(")

    # Every run should evaluate a collapsed ToQ whose nodes are {root} unioned with the cut_edges
    for run in rep.runs:
        expected_nodes = set(run.plan.cut_edges) | {toq.root_id}
        assert set(run.collapsed.toq.nodes.keys()) == expected_nodes
        assert run.root_answer.text == run.trace.answer[toq.root_id].text

def test_frontier_caching_reduces_collapser_calls():
    # Same tree as above
    nodes = {
        1: ToQNode(1, "Q1?", parent=3),
        2: ToQNode(2, "Q2?", parent=3),
        3: ToQNode(3, "Q3([A1],[A2])", parent=5),
        4: ToQNode(4, "Q4?", parent=5),
        5: ToQNode(5, "Q5([A3],[A4])", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=5)
    toq.validate()

    # Collapser call count is what we care about
    collapser = RecordingCollapser()

    # Answerer can be trivial
    answerer = ToyAnswerer()

    cache = {}
    rep = run_consistency_check(
        toq,
        answerer=answerer,
        collapser=collapser,
        plan_opts={"include_empty": True},
        cache=cache,
    )

    # Naively, without caching, collapser would be called once per (plan, component root).
    # With 16 plans and average >1 component per plan, you'd see many more calls than nodes.
    # With frontier caching, calls should be "reasonably small"—certainly less than (plans * nodes).
    naive_upper = len(rep.runs) * len(toq.nodes)  # very loose upper bound
    assert len(collapser.calls) < naive_upper

    # Also, cache should have stored many collapsed questions
    assert len(cache) > 0


expect_ok(test_runs_count_and_shapes, "run count + collapsed node sets match")
expect_ok(test_frontier_caching_reduces_collapser_calls, "frontier caching reduces collapser calls")
print("consistency.py tests done")

# %%
