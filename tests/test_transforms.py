import pytest

from operadic_consistency.core.toq_types import ToQ, ToQNode
from operadic_consistency.core.transforms import (
    CollapsePlan,
    enumerate_collapse_plans,
    component_roots,
    apply_collapse_plan,
)


def _three_node_toq():
    nodes = {
        1: ToQNode(1, "Q1?", parent=3),
        2: ToQNode(2, "Q2?", parent=3),
        3: ToQNode(3, "Q3 uses [A1],[A2]?", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=3)
    toq.validate()
    return toq


def test_enumerate_plan_count():
    toq = _three_node_toq()
    plans = enumerate_collapse_plans(toq, include_empty=True)
    # 2 edges => 2^2 = 4 plans
    assert len(plans) == 2 ** (len(toq.nodes) - 1)

    cut_sets = {p.cut_edges for p in plans}
    assert () in cut_sets
    assert (1, 2) in cut_sets


def test_component_roots():
    toq = _three_node_toq()

    assert set(component_roots(toq, CollapsePlan(()))) == {3}
    assert set(component_roots(toq, CollapsePlan((1,)))) == {3, 1}
    assert set(component_roots(toq, CollapsePlan((1, 2)))) == {3, 1, 2}


def test_apply_empty_cut():
    toq = _three_node_toq()
    p_empty = CollapsePlan(())
    ct = apply_collapse_plan(toq, p_empty, {3: "C(3)"})
    ct.toq.validate()
    assert set(ct.toq.nodes.keys()) == {3}
    assert ct.toq.nodes[3].parent is None
    assert ct.toq.nodes[3].text == "C(3)"
    assert ct.removed_nodes == frozenset({1, 2})


def test_apply_cut_one():
    toq = _three_node_toq()
    p_cut1 = CollapsePlan((1,))
    ct = apply_collapse_plan(toq, p_cut1, {3: "C(3)", 1: "C(1)"})
    ct.toq.validate()
    assert set(ct.toq.nodes.keys()) == {3, 1}
    assert ct.toq.nodes[3].parent is None
    assert ct.toq.nodes[1].parent == 3
    assert ct.toq.nodes[1].text == "C(1)"
    assert ct.toq.nodes[3].text == "C(3)"
    assert ct.removed_nodes == frozenset({2})


def test_apply_cut_both():
    toq = _three_node_toq()
    p_cut12 = CollapsePlan((1, 2))
    ct = apply_collapse_plan(toq, p_cut12, {3: "C(3)", 1: "C(1)", 2: "C(2)"})
    ct.toq.validate()
    assert set(ct.toq.nodes.keys()) == {3, 1, 2}
    assert ct.toq.nodes[1].parent == 3
    assert ct.toq.nodes[2].parent == 3
    assert ct.removed_nodes == frozenset()


def test_missing_collapsed_question_raises():
    toq = _three_node_toq()
    with pytest.raises(ValueError, match="Missing collapsed question for component root 1"):
        apply_collapse_plan(toq, CollapsePlan((1,)), {3: "C(3)"})


def test_invalid_cut_root_raises():
    toq = _three_node_toq()
    with pytest.raises(ValueError, match="root_id cannot be a cut edge"):
        apply_collapse_plan(toq, CollapsePlan((3,)), {3: "C(3)"})


def test_invalid_cut_nonexistent_raises():
    toq = _three_node_toq()
    with pytest.raises(ValueError, match="node 99 not in ToQ"):
        apply_collapse_plan(toq, CollapsePlan((99,)), {3: "C(3)", 99: "C(99)"})
