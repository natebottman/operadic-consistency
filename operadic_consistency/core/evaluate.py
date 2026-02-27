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
# core/evaluate.py

# %%
# Dev setup
# %load_ext autoreload
# %autoreload 2

# %%
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, List

from operadic_consistency.core.toq_types import ToQ, NodeId
from operadic_consistency.core.interfaces import Answer, Answerer


# %%
@dataclass(frozen=True)
class EvalTrace:
    rendered_question: Mapping[NodeId, str]
    # Actual question text asked at each node (after substitution)

    answer: Mapping[NodeId, Answer]
    # Model answers at each node


class Substituter(Protocol):
    def __call__(self, template: str, child_answers: Mapping[NodeId, str]) -> str:
        # Combine child answers into the parent question template
        ...


def default_substituter(template: str, child_answers: Mapping[NodeId, str]) -> str:
    """
    Default convention: child answers are referenced as [A<child_id>].
    Example: "Which is bigger, [A1] or [A2]?"
    """
    out = template
    for cid, ans in child_answers.items():
        out = out.replace(f"[A{cid}]", ans)
    return out


def _postorder(toq: ToQ) -> List[NodeId]:
    """Return node ids in postorder (children before parent), starting at root."""
    ch = toq.children()
    order: List[NodeId] = []
    visited = set()

    def dfs(n: NodeId) -> None:
        if n in visited:
            return
        visited.add(n)
        for c in ch.get(n, []):
            dfs(c)
        order.append(n)

    dfs(toq.root_id)
    return order


def evaluate_toq(
    toq: ToQ,
    *,
    answerer: Answerer,
    substituter: Optional[Substituter] = None,
    context: Optional[str] = None,
) -> EvalTrace:
    # Evaluate a ToQ leaves->root by repeatedly answering questions

    toq.validate()
    sub = substituter or default_substituter

    ch = toq.children()
    order = _postorder(toq)

    rendered: Dict[NodeId, str] = {}
    answers: Dict[NodeId, Answer] = {}

    for nid in order:
        template = toq.nodes[nid].text
        child_ids = ch.get(nid, [])

        # Gather child answers (text only) for substitution
        child_ans_text: Dict[NodeId, str] = {cid: answers[cid].text for cid in child_ids}

        # Render question at this node:
        # - leaves: ask template as-is
        # - internal nodes: apply substituter
        if len(child_ids) == 0:
            q = template
        else:
            q = sub(template, child_ans_text)

        rendered[nid] = q

        # Ask model
        answers[nid] = answerer(q, context=context)

    return EvalTrace(rendered_question=rendered, answer=answers)

# %%
