# core/transforms.py

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Mapping, Sequence, Optional, Set, Tuple, FrozenSet

from operadic_consistency.core.toq_types import ToQ, ToQNode, NodeId, OpenToQ

@dataclass(frozen=True)
class CollapsePlan:
    cut_edges: tuple[NodeId, ...]
    # Each NodeId c represents cutting the edge parent(c) -> c (so c becomes a component root).

def enumerate_collapse_plans(
    toq: ToQ,
    *,
    include_empty: bool = True,
) -> Sequence[CollapsePlan]:
    """
    Enumerate all partial collapses by choosing which edges to cut.
    If the tree has k edges, this returns 2^k plans.
    (Edges are identified by their child node id; i.e., all nodes except root.)
    """
    toq.validate()

    edges: List[NodeId] = [nid for nid in toq.nodes if nid != toq.root_id]  # |edges| = k
    plans: List[CollapsePlan] = []

    n = len(edges)
    for r in range(n + 1):
        if r == 0 and not include_empty:
            continue
        for subset in combinations(edges, r):
            plans.append(CollapsePlan(tuple(sorted(subset))))

    return plans

def component_roots(toq: ToQ, plan: CollapsePlan) -> Tuple[NodeId, ...]:
    """
    The cut edges define a partition into components.
    The roots (tops) of those components are exactly the union of {root_id} with the cut_edges.
    """
    roots = set(plan.cut_edges) | {toq.root_id}
    return tuple(sorted(roots))

def _component_root(toq: ToQ, nid: NodeId, cut: Set[NodeId]) -> NodeId:
    """
    Return the root of nid's component under the cut-set.
    Walk upward until hitting the global root, or until crossing a cut edge.
    """
    cur = nid
    while True:
        p = toq.nodes[cur].parent
        if p is None:
            return cur
        if cur in cut:
            return cur
        cur = p

def extract_open_toq(toq: ToQ, plan: CollapsePlan, *, root: NodeId) -> OpenToQ:
    """
    Extract the component rooted at `root` as an OpenToQ.

    - Internal nodes are those reachable from `root` without crossing a cut edge.
    - `inputs` are the frontier cut nodes immediately below this component:
        i.e. children v of some internal node u where v is a cut edge.
      These are external leaves: their answers are *not* computed inside the component.
    """
    toq.validate()
    cut: Set[NodeId] = set(plan.cut_edges)
    ch = toq.children()

    internal: Set[NodeId] = set()
    frontier: Set[NodeId] = set()

    stack: List[NodeId] = [root]
    while stack:
        u = stack.pop()
        if u in internal:
            continue
        internal.add(u)

        for v in ch.get(u, []):
            if v in cut:
                frontier.add(v)     # boundary: external input
            else:
                stack.append(v)     # stays internal

    # Build induced ToQ on internal nodes, keeping original ids/texts.
    # Parent pointers: if a node's parent is outside internal, set parent=None (it becomes root).
    new_nodes: Dict[NodeId, ToQNode] = {}
    for nid in internal:
        node = toq.nodes[nid]
        p = node.parent
        new_parent = p if (p in internal) else None
        new_nodes[nid] = ToQNode(id=nid, text=node.text, parent=new_parent)

    # Root_id for this OpenToQ is `root` (it should be the unique root in the induced subgraph).
    open_toq = ToQ(nodes=new_nodes, root_id=root)
    open_toq.validate()

    return OpenToQ(
        toq=open_toq,
        inputs=tuple(sorted(frontier)),
        root_id=root,
    )

@dataclass(frozen=True)
class CollapsedToQ:
    toq: ToQ
    plan: CollapsePlan
    removed_nodes: FrozenSet[NodeId]
    collapsed_question_by_root: Mapping[NodeId, str]
    component_roots: tuple[NodeId, ...]
    open_toq_by_root: Optional[Mapping[NodeId, OpenToQ]] = None
    # Quotient ToQ after contracting each component to its root node.

def apply_collapse_plan(
    toq: ToQ,
    plan: CollapsePlan,
    collapsed_question_by_root: Mapping[NodeId, str],
) -> CollapsedToQ:
    """
    Apply the edge cuts, forming a partition into sub-ToQs, then contract each
    component to a single node (its component root).

    Resulting ToQ:
      - nodes are exactly the component roots {root_id} unioned with the cut_edges
      - parent pointers are re-wired between component roots:
            parent(r) = component_root(parent_original(r))
      - node text is replaced by collapsed_question_by_root[r]
    """
    toq.validate()
    cut: Set[NodeId] = set(plan.cut_edges)

    # Validate cut edges: must be non-root nodes with a parent
    if toq.root_id in cut:
        raise ValueError("Invalid cut: root_id cannot be a cut edge (no incoming edge).")
    for c in cut:
        if c not in toq.nodes:
            raise ValueError(f"Invalid cut: node {c} not in ToQ.")
        if toq.nodes[c].parent is None:
            raise ValueError(f"Invalid cut: node {c} has parent=None (is a root).")

    roots = component_roots(toq, plan)

    # Require a collapsed question for each component root (including the global root)
    for r in roots:
        if r not in collapsed_question_by_root:
            raise ValueError(f"Missing collapsed question for component root {r}.")

    # Build quotient nodes (keep ids = component roots for stable provenance)
    new_nodes: Dict[NodeId, ToQNode] = {}
    for r in roots:
        if r == toq.root_id:
            new_parent = None
        else:
            p = toq.nodes[r].parent
            assert p is not None
            new_parent = _component_root(toq, p, cut)

        new_nodes[r] = ToQNode(
            id=r,
            text=collapsed_question_by_root[r],
            parent=new_parent,
        )

    removed_nodes = frozenset(set(toq.nodes.keys()) - set(roots))

    new_toq = ToQ(nodes=new_nodes, root_id=toq.root_id)
    new_toq.validate()

    return CollapsedToQ(
        toq=new_toq,
        plan=plan,
        removed_nodes=removed_nodes,
        collapsed_question_by_root=collapsed_question_by_root,
        component_roots=roots,
    )
