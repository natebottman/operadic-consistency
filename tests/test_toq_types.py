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


# %%
# ---- tests for core/toq_types.py ----

def expect_ok(toq: ToQ):
    try:
        toq.validate()
        print("validate OK")
    except Exception as e:
        print("unexpected error:", e)

def expect_fail(toq: ToQ, msg: str):
    try:
        toq.validate()
        print("expected failure but validate() passed:", msg)
    except Exception as e:
        print("expected failure:", msg)
        print("  ", type(e).__name__, e)


# 1) Minimal valid ToQ (single node)
nodes1 = {
    1: ToQNode(1, "Root question?", parent=None),
}
toq1 = ToQ(nodes=nodes1, root_id=1)
expect_ok(toq1)

# children / leaves sanity
print("children:", toq1.children())
print("leaves:", toq1.leaves())


# 2) Simple 3-node tree
nodes2 = {
    1: ToQNode(1, "Subquestion A?", parent=3),
    2: ToQNode(2, "Subquestion B?", parent=3),
    3: ToQNode(3, "Main question using [A1], [A2]?", parent=None),
}
toq2 = ToQ(nodes=nodes2, root_id=3)
expect_ok(toq2)

print("children:", toq2.children())
print("leaves:", sorted(toq2.leaves()))


# 3) Multiple roots (invalid)
nodes_bad_roots = {
    1: ToQNode(1, "Q1?", parent=None),
    2: ToQNode(2, "Q2?", parent=None),
}
toq_bad_roots = ToQ(nodes=nodes_bad_roots, root_id=1)
expect_fail(toq_bad_roots, "multiple roots")


# 4) Missing parent reference (invalid)
nodes_missing_parent = {
    1: ToQNode(1, "Q1?", parent=99),
}
toq_missing_parent = ToQ(nodes=nodes_missing_parent, root_id=1)
expect_fail(toq_missing_parent, "missing parent id")


# 5) Cycle (invalid)
nodes_cycle = {
    1: ToQNode(1, "Q1?", parent=2),
    2: ToQNode(2, "Q2?", parent=1),
}
toq_cycle = ToQ(nodes=nodes_cycle, root_id=1)
expect_fail(toq_cycle, "cycle in ToQ")


# 6) Orphan subtree (invalid)
nodes_orphan = {
    1: ToQNode(1, "Root?", parent=None),
    2: ToQNode(2, "Child?", parent=1),
    3: ToQNode(3, "Orphan?", parent=None),
}
toq_orphan = ToQ(nodes=nodes_orphan, root_id=1)
expect_fail(toq_orphan, "unreachable orphan node")


# 7) Node id mismatch (invalid)
nodes_id_mismatch = {
    1: ToQNode(99, "Bad id?", parent=None),
}
toq_id_mismatch = ToQ(nodes=nodes_id_mismatch, root_id=1)
expect_fail(toq_id_mismatch, "node key != node.id")

# %%
