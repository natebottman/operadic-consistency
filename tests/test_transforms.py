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
import pytest
from operadic_consistency.core.toq_types import ToQ, ToQNode
from operadic_consistency.core.transforms import (
    CollapsePlan,
    enumerate_collapse_plans,
    component_roots,
    apply_collapse_plan,
)



# %%
# ---- tests for core/transforms.py ----

def expect_ok(fn, msg=""):
    try:
        fn()
        print("passed", msg or fn.__name__)
    except Exception as e:
        print("failed", msg or fn.__name__, "->", type(e).__name__, e)

def expect_fail(fn, msg=""):
    try:
        fn()
        print("expected failure but passed:", msg or fn.__name__)
    except Exception as e:
        print("expected failure:", msg or fn.__name__)
        print("  ", type(e).__name__, e)


# A small ToQ: root(3) with children (1,2)
nodes = {
    1: ToQNode(1, "Q1?", parent=3),
    2: ToQNode(2, "Q2?", parent=3),
    3: ToQNode(3, "Q3 uses [A1],[A2]?", parent=None),
}
toq = ToQ(nodes=nodes, root_id=3)
toq.validate()

# --- 1) enumeration count ---
plans = enumerate_collapse_plans(toq, include_empty=True)
print("num edges =", len(nodes) - 1)
print("num plans =", len(plans))
assert len(plans) == 2 ** (len(nodes) - 1)  # 2^k, k = #edges

# check empty and full cuts exist
cut_sets = {p.cut_edges for p in plans}
assert () in cut_sets
assert tuple(sorted([1, 2])) in cut_sets
print("passed; enumeration includes empty and full cut")

# --- 2) component_roots correctness ---
p_empty = CollapsePlan(())
assert set(component_roots(toq, p_empty)) == {3}

p_cut1 = CollapsePlan((1,))
assert set(component_roots(toq, p_cut1)) == {3, 1}

p_cut12 = CollapsePlan((1, 2))
assert set(component_roots(toq, p_cut12)) == {3, 1, 2}
print("passed; component_roots behaves")

# --- 3) apply_collapse_plan structure ---
def test_apply_empty():
    cq = {3: "C(3)"}  # only root component
    ct = apply_collapse_plan(toq, p_empty, cq)
    ct.toq.validate()
    assert set(ct.toq.nodes.keys()) == {3}
    assert ct.toq.nodes[3].parent is None
    assert ct.toq.nodes[3].text == "C(3)"
    assert ct.removed_nodes == frozenset({1, 2})

expect_ok(test_apply_empty, "apply empty cut collapses everything to root")

def test_apply_cut1():
    cq = {3: "C(3)", 1: "C(1)"}  # components: root and node1
    ct = apply_collapse_plan(toq, p_cut1, cq)
    ct.toq.validate()
    assert set(ct.toq.nodes.keys()) == {3, 1}
    assert ct.toq.nodes[3].parent is None
    assert ct.toq.nodes[1].parent == 3
    assert ct.toq.nodes[1].text == "C(1)"
    assert ct.toq.nodes[3].text == "C(3)"
    assert ct.removed_nodes == frozenset({2})

expect_ok(test_apply_cut1, "apply cut {1} yields quotient nodes {3,1}")

def test_apply_cut12():
    cq = {3: "C(3)", 1: "C(1)", 2: "C(2)"}
    ct = apply_collapse_plan(toq, p_cut12, cq)
    ct.toq.validate()
    assert set(ct.toq.nodes.keys()) == {3, 1, 2}
    assert ct.toq.nodes[1].parent == 3
    assert ct.toq.nodes[2].parent == 3
    assert ct.removed_nodes == frozenset()  # all nodes are component roots

expect_ok(test_apply_cut12, "apply cut {1,2} keeps all nodes but replaces texts")

# --- 4) failure modes ---
def test_missing_collapsed_question():
    cq = {3: "C(3)"}  # missing 1
    with pytest.raises(ValueError, match="Missing collapsed question for component root 1"):
        apply_collapse_plan(toq, p_cut1, cq)

def test_invalid_cut_root():
    bad_plan = CollapsePlan((3,))
    cq = {3: "C(3)"}
    with pytest.raises(ValueError, match="root_id cannot be a cut edge"):
        apply_collapse_plan(toq, bad_plan, cq)

def test_invalid_cut_nonexistent():
    bad_plan = CollapsePlan((99,))
    cq = {3: "C(3)", 99: "C(99)"}
    with pytest.raises(ValueError, match="node 99 not in ToQ"):
        apply_collapse_plan(toq, bad_plan, cq)

print("transforms.py tests done")

# %%
