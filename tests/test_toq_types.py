import pytest

from operadic_consistency.core.toq_types import ToQ, ToQNode


def test_single_node_valid():
    toq = ToQ(nodes={1: ToQNode(1, "Root question?", parent=None)}, root_id=1)
    toq.validate()
    assert list(toq.leaves()) == [1]
    assert toq.children() == {1: []}


def test_three_node_tree_valid():
    nodes = {
        1: ToQNode(1, "Subquestion A?", parent=3),
        2: ToQNode(2, "Subquestion B?", parent=3),
        3: ToQNode(3, "Main question using [A1], [A2]?", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=3)
    toq.validate()
    assert sorted(toq.leaves()) == [1, 2]
    ch = toq.children()
    assert sorted(ch[3]) == [1, 2]


def test_multiple_roots_invalid():
    nodes = {
        1: ToQNode(1, "Q1?", parent=None),
        2: ToQNode(2, "Q2?", parent=None),
    }
    with pytest.raises(ValueError, match="root"):
        ToQ(nodes=nodes, root_id=1).validate()


def test_missing_parent_invalid():
    with pytest.raises(ValueError, match="missing parent"):
        ToQ(nodes={1: ToQNode(1, "Q1?", parent=99)}, root_id=1).validate()


def test_cycle_invalid():
    nodes = {
        1: ToQNode(1, "Q1?", parent=2),
        2: ToQNode(2, "Q2?", parent=1),
    }
    with pytest.raises(ValueError, match="[Cc]ycle|root"):
        ToQ(nodes=nodes, root_id=1).validate()


def test_orphan_node_invalid():
    nodes = {
        1: ToQNode(1, "Root?", parent=None),
        2: ToQNode(2, "Child?", parent=1),
        3: ToQNode(3, "Orphan?", parent=None),
    }
    with pytest.raises(ValueError):
        ToQ(nodes=nodes, root_id=1).validate()


def test_node_id_mismatch_invalid():
    with pytest.raises(ValueError, match="Node key"):
        ToQ(nodes={1: ToQNode(99, "Bad id?", parent=None)}, root_id=1).validate()
