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
from operadic_consistency.core.toq_types import ToQ, ToQNode
from operadic_consistency.core.serialization import toq_to_json, toq_from_json

# %%
# ---- tests for serialization.py ----

def expect_ok(fn, msg=""):
    try:
        fn()
        print("passed", msg or fn.__name__)
    except Exception as e:
        print("FAILED", msg or fn.__name__, "->", type(e).__name__, e)


def expect_fail(fn, msg=""):
    try:
        fn()
        print("FAILED (expected failure)", msg or fn.__name__)
    except Exception as e:
        print("passed (expected failure)", msg or fn.__name__)
        print(" ", type(e).__name__, e)


def test_roundtrip_simple():
    nodes = {
        1: ToQNode(1, "Root?", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=1)
    toq.validate()

    j = toq_to_json(toq)
    toq2 = toq_from_json(j)

    assert toq2.root_id == 1
    assert set(toq2.nodes.keys()) == {1}
    assert toq2.nodes[1].text == "Root?"
    assert toq2.nodes[1].parent is None


def test_roundtrip_nontrivial_tree():
    nodes = {
        1: ToQNode(1, "Q1?", parent=3),
        2: ToQNode(2, "Q2?", parent=3),
        3: ToQNode(3, "Q3([A1],[A2])", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=3)
    toq.validate()

    j = toq_to_json(toq)
    toq2 = toq_from_json(j)

    assert toq2.root_id == 3
    assert set(toq2.nodes.keys()) == {1, 2, 3}
    for nid in nodes:
        assert toq2.nodes[nid].text == nodes[nid].text
        assert toq2.nodes[nid].parent == nodes[nid].parent


def test_json_keys_are_strings():
    nodes = {
        1: ToQNode(1, "Root?", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=1)
    j = toq_to_json(toq)

    # JSON object keys must be strings
    assert list(j["nodes"].keys()) == ["1"]


def test_invalid_missing_fields():
    bad = {
        "nodes": {
            "1": {"id": 1, "text": "Q?", "parent": None}
        }
        # missing root_id
    }

    expect_fail(lambda: toq_from_json(bad), "missing root_id")


def test_invalid_parent_reference():
    bad = {
        "root_id": 1,
        "nodes": {
            "1": {"id": 1, "text": "Root?", "parent": None},
            "2": {"id": 2, "text": "Child?", "parent": 99},  # invalid parent
        },
    }

    expect_fail(lambda: toq_from_json(bad), "invalid parent id")


def test_node_id_mismatch():
    bad = {
        "root_id": 1,
        "nodes": {
            "1": {"id": 99, "text": "Q?", "parent": None},
        },
    }

    expect_fail(lambda: toq_from_json(bad), "node key != node.id")


expect_ok(test_roundtrip_simple, "roundtrip single-node")
expect_ok(test_roundtrip_nontrivial_tree, "roundtrip multi-node")
expect_ok(test_json_keys_are_strings, "json keys are strings")
test_invalid_missing_fields()
test_invalid_parent_reference()
test_node_id_mismatch()

print("serialization.py tests done")

# %%
