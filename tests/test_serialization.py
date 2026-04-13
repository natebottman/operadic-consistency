import pytest

from operadic_consistency.core.toq_types import ToQ, ToQNode
from operadic_consistency.core.serialization import toq_to_json, toq_from_json


def test_roundtrip_single_node():
    toq = ToQ(nodes={1: ToQNode(1, "Root?", parent=None)}, root_id=1)
    toq.validate()

    j = toq_to_json(toq)
    toq2 = toq_from_json(j)

    assert toq2.root_id == 1
    assert set(toq2.nodes.keys()) == {1}
    assert toq2.nodes[1].text == "Root?"
    assert toq2.nodes[1].parent is None


def test_roundtrip_multi_node():
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


def test_json_node_keys_are_strings():
    toq = ToQ(nodes={1: ToQNode(1, "Root?", parent=None)}, root_id=1)
    j = toq_to_json(toq)
    assert list(j["nodes"].keys()) == ["1"]


def test_missing_root_id_raises():
    bad = {"nodes": {"1": {"id": 1, "text": "Q?", "parent": None}}}
    with pytest.raises(ValueError, match="missing root_id"):
        toq_from_json(bad)


def test_invalid_parent_reference_raises():
    bad = {
        "root_id": 1,
        "nodes": {
            "1": {"id": 1, "text": "Root?", "parent": None},
            "2": {"id": 2, "text": "Child?", "parent": 99},
        },
    }
    with pytest.raises(ValueError):
        toq_from_json(bad)


def test_node_id_mismatch_raises():
    bad = {
        "root_id": 1,
        "nodes": {"1": {"id": 99, "text": "Q?", "parent": None}},
    }
    with pytest.raises(ValueError):
        toq_from_json(bad)
