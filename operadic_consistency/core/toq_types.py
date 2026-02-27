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
# core/toq_types.py

# %%
# Dev setup
# %load_ext autoreload
# %autoreload 2

# %%
from typing import Dict, List, Set, Optional, Mapping, Sequence
from dataclasses import dataclass

# %%
NodeId = int  # unique identifier for a question node

@dataclass(frozen=True)
class ToQNode:
    id: NodeId
    text: str                  # question text at this node
    parent: Optional[NodeId]   # parent node id; None iff root

@dataclass(frozen=True)
class ToQ:
    nodes: Mapping[NodeId, ToQNode]
    root_id: NodeId

    def children(self) -> Mapping[NodeId, Sequence[NodeId]]:
        # Return adjacency: node -> list of child node ids
        ch: Dict[NodeId, List[NodeId]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            p = node.parent
            if p is not None:
                # If validate() is called first, p is guaranteed to exist.
                # But we still guard to avoid KeyError in casual usage.
                if p in ch:
                    ch[p].append(nid)
        return ch

    def leaves(self) -> Sequence[NodeId]:
        # Return all leaf nodes (nodes with no children)
        ch = self.children()
        return [nid for nid in self.nodes if len(ch.get(nid, [])) == 0]

    def validate(self) -> None:
        # Check tree well-formedness (unique root, no cycles, valid parents)

        if self.root_id not in self.nodes:
            raise ValueError(f"root_id {self.root_id} not in nodes")

        # Check parent pointers refer to existing nodes and ids are consistent
        for nid, node in self.nodes.items():
            if node.id != nid:
                raise ValueError(f"Node key {nid} != node.id {node.id}")
            if node.parent is not None and node.parent not in self.nodes:
                raise ValueError(f"Node {nid} has missing parent {node.parent}")
            if node.parent == nid:
                raise ValueError(f"Node {nid} cannot be its own parent")

        # Exactly one root: root_id must have parent None, and no other node may have parent None
        if self.nodes[self.root_id].parent is not None:
            raise ValueError(f"root_id {self.root_id} must have parent=None")

        roots = [nid for nid, node in self.nodes.items() if node.parent is None]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly 1 root, found {len(roots)}: {roots}")
        if roots[0] != self.root_id:
            raise ValueError(f"root_id {self.root_id} != the unique root {roots[0]}")

        # Detect cycles + ensure connectivity from root via DFS
        ch = self.children()
        visited: Set[NodeId] = set()
        in_stack: Set[NodeId] = set()

        def dfs(u: NodeId) -> None:
            visited.add(u)
            in_stack.add(u)
            for v in ch.get(u, []):
                if v in in_stack:
                    raise ValueError(f"Cycle detected: edge {u} -> {v}")
                if v not in visited:
                    dfs(v)
            in_stack.remove(u)

        dfs(self.root_id)

        # Ensure all nodes reachable (no orphan subtrees)
        if len(visited) != len(self.nodes):
            unreachable = sorted(set(self.nodes.keys()) - visited)
            raise ValueError(f"Unreachable nodes from root {self.root_id}: {unreachable}")

@dataclass(frozen=True)
class OpenToQ:
    """
    A ToQ together with a tuple of external input node-ids.

    Convention: placeholders [A<input_id>] may appear in node texts, and are
    intended to be filled by answers supplied externally (i.e., from outside
    this OpenToQ).
    """
    toq: ToQ
    inputs: tuple[NodeId, ...]
    root_id: NodeId

# %%
