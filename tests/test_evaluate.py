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
from typing import Optional, Mapping

from operadic_consistency.core.toq_types import NodeId, ToQ, ToQNode
from operadic_consistency.core.interfaces import Answer
from operadic_consistency.core.evaluate import evaluate_toq

# %%
# ---- tests for core/evaluate.py ----

def expect_ok(fn, msg=""):
    try:
        fn()
        print("passed", msg or fn.__name__)
    except Exception as e:
        print("FAILED", msg or fn.__name__, "->", type(e).__name__, e)

# A toy answerer that is deterministic and records calls
class RecordingAnswerer:
    def __init__(self):
        self.calls = []  # list of (question, context)

    def __call__(self, question: str, *, context: Optional[str] = None) -> Answer:
        self.calls.append((question, context))
        return Answer(text=f"ANS({question})", meta={"context": context})

def test_leaf_only():
    nodes = {1: ToQNode(1, "Root?", parent=None)}
    toq = ToQ(nodes=nodes, root_id=1)

    ans = RecordingAnswerer()
    tr = evaluate_toq(toq, answerer=ans, context="ctx")

    assert tr.rendered_question[1] == "Root?"
    assert tr.answer[1].text == "ANS(Root?)"
    assert ans.calls == [("Root?", "ctx")]

def test_two_leaves_then_root_substitution():
    nodes = {
        1: ToQNode(1, "How old is Michael Jordan?", parent=3),
        2: ToQNode(2, "How old is Larry Bird?", parent=3),
        3: ToQNode(3, "If Michael Jordan is [A1] and Larry Bird is [A2], who is older?", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=3)

    ans = RecordingAnswerer()
    tr = evaluate_toq(toq, answerer=ans, context=None)

    # Leaves rendered as-is
    assert tr.rendered_question[1] == "How old is Michael Jordan?"
    assert tr.rendered_question[2] == "How old is Larry Bird?"
    # Root should substitute leaf answers
    expected_root_q = "If Michael Jordan is ANS(How old is Michael Jordan?) and Larry Bird is ANS(How old is Larry Bird?), who is older?"
    assert tr.rendered_question[3] == expected_root_q

    # Order of calls should be leaves before root (postorder)
    assert ans.calls[0][0] in ("How old is Michael Jordan?", "How old is Larry Bird?")
    assert ans.calls[1][0] in ("How old is Michael Jordan?", "How old is Larry Bird?")
    assert ans.calls[0][0] != ans.calls[1][0]
    assert ans.calls[2][0] == expected_root_q

def test_custom_substituter():
    def subst(template: str, child_answers: Mapping[NodeId, str]) -> str:
        # Ignore placeholders; append a structured summary
        parts = "; ".join(f"{cid}={child_answers[cid]}" for cid in sorted(child_answers))
        return f"{template}\nCHILDREN: {parts}"

    nodes = {
        1: ToQNode(1, "Q1?", parent=2),
        2: ToQNode(2, "Q2?", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=2)

    ans = RecordingAnswerer()
    tr = evaluate_toq(toq, answerer=ans, substituter=subst, context="CTX")
    
    assert tr.rendered_question[1] == "Q1?"
    assert tr.rendered_question[2] == "Q2?\nCHILDREN: 1=ANS(Q1?)"
    assert ans.calls == [("Q1?", "CTX"), ("Q2?\nCHILDREN: 1=ANS(Q1?)", "CTX")]

expect_ok(test_leaf_only, "single-node ToQ evaluates")
expect_ok(test_two_leaves_then_root_substitution, "substitution + postorder works")
expect_ok(test_custom_substituter, "custom substituter is respected")
print("evaluate.py tests done")

# %%
