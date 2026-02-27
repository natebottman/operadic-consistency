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
# core/serialization.py

# %%
# Dev setup
# %load_ext autoreload
# %autoreload 2

# %%
from typing import Any, Dict, Mapping

from operadic_consistency.core.toq_types import ToQ, ToQNode, NodeId


# %%
def toq_to_json(toq: ToQ) -> Dict[str, Any]:
    """
    Convert a ToQ to a JSON-serializable dict.
    NodeId keys are converted to strings (JSON requirement).

    Example:
        >>> toq = ToQ(
        ...     nodes={
        ...         1: ToQNode(1, "When did WW2 end?", parent=2),
        ...         2: ToQNode(2, "Who was President at time [A1]?", parent=None),
        ...     },
        ...     root_id=2,
        ... )
        >>> toq_to_json(toq)
        {'root_id': 2, 'nodes': {'1': {'id': 1, 'text': 'When did WW2 end?', 'parent': 2}, '2': {'id': 2, 'text': 'Who was President at time [A1]?', 'parent': None}}}
    """
    return {
        "root_id": toq.root_id,
        "nodes": {
            str(nid): {
                "id": node.id,
                "text": node.text,
                "parent": node.parent,
            }
            for nid, node in toq.nodes.items()
        },
    }


def toq_from_json(obj: Mapping[str, Any]) -> ToQ:
    """
    Parse a ToQ from a JSON-like dict and validate it.

    Example:
        >>> raw = {
        ...     "root_id": 2,
        ...     "nodes": {
        ...         "1": {"id": 1, "text": "When did WW2 end?", "parent": 2},
        ...         "2": {"id": 2, "text": "Who was President at time [A1]?", "parent": None},
        ...     },
        ... }
        >>> toq = toq_from_json(raw)
        >>> toq.root_id
        2
        >>> sorted(toq.nodes)
        [1, 2]
        >>> toq.nodes[1].parent
        2

    Invalid structures raise a ``ValueError``:
        >>> toq_from_json({"nodes": {}})
        Traceback (most recent call last):
        ...
        ValueError: Invalid ToQ JSON: missing root_id or nodes
    """
    if "root_id" not in obj or "nodes" not in obj:
        raise ValueError("Invalid ToQ JSON: missing root_id or nodes")

    nodes_raw = obj["nodes"]
    if not isinstance(nodes_raw, Mapping):
        raise ValueError("Invalid ToQ JSON: nodes must be a mapping")

    nodes: Dict[NodeId, ToQNode] = {}

    for k, v in nodes_raw.items():
        try:
            nid = int(k)
        except Exception:
            raise ValueError(f"Invalid node id key: {k!r}")

        if not isinstance(v, Mapping):
            raise ValueError(f"Invalid node entry for id {nid}")

        try:
            node = ToQNode(
                id=v["id"],
                text=v["text"],
                parent=v.get("parent"),
            )
        except KeyError as e:
            raise ValueError(f"Missing field {e} in node {nid}")

        nodes[nid] = node

    toq = ToQ(nodes=nodes, root_id=obj["root_id"])
    toq.validate()
    return toq

# %%
