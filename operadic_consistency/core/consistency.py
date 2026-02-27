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
# core/consistency.py

# %%
# Dev setup
# %load_ext autoreload
# %autoreload 2

# %%
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Dict, List

from operadic_consistency.core.toq_types import ToQ, NodeId, OpenToQ
from operadic_consistency.core.interfaces import Answer, Answerer, Collapser, Normalizer, QuestionDecomposer
from operadic_consistency.core.transforms import (
    CollapsePlan,
    CollapsedToQ,
    enumerate_collapse_plans,
    component_roots,
    apply_collapse_plan,
    extract_open_toq
)
from operadic_consistency.core.evaluate import EvalTrace, Substituter, evaluate_toq


# %%
@dataclass(frozen=True)
class RunRecord:
    plan: CollapsePlan
    collapsed: CollapsedToQ
    trace: EvalTrace
    root_answer: Answer
    normalized_root: Optional[str]
    # Result of evaluating one partial-collapse variant


@dataclass(frozen=True)
class ConsistencyReport:
    base_trace: EvalTrace
    base_root_answer: Answer
    runs: Sequence[RunRecord]
    summary: Mapping[str, Any]
    # Full results of the consistency check


def run_consistency_check(
    toq: ToQ,
    *,
    answerer: Answerer,
    collapser: Collapser,
    normalizer: Optional[Normalizer] = None,
    substituter: Optional[Substituter] = None,
    context: Optional[str] = None,
    plan_opts: Optional[Mapping[str, Any]] = None,
    cache: Optional[MutableMapping[tuple, str]] = None,
) -> ConsistencyReport:
    """
    Run the operadic consistency check on a given ToQ:

      1) Evaluate original ToQ (baseline)
      2) For each edge-cut plan:
         - Extract each component as an OpenToQ
         - Collapse each OpenToQ to a single question (cached by interface)
         - Build quotient ToQ
         - Evaluate quotient
    """

    toq.validate()

    # -------------------------
    # Baseline
    # -------------------------
    base_trace = evaluate_toq(
        toq,
        answerer=answerer,
        substituter=substituter,
        context=context,
    )
    base_root_answer = base_trace.answer[toq.root_id]

    include_empty = True
    if plan_opts is not None and "include_empty" in plan_opts:
        include_empty = bool(plan_opts["include_empty"])

    plans = enumerate_collapse_plans(toq, include_empty=include_empty)

    if cache is None:
        cache = {}

    runs: List[RunRecord] = []

    # -------------------------
    # Iterate collapse plans
    # -------------------------
    for plan in plans:
        roots = component_roots(toq, plan)

        open_toq_by_root: Dict[NodeId, OpenToQ] = {}
        collapsed_question_by_root: Dict[NodeId, str] = {}

        for r in roots:
            open_toq = extract_open_toq(toq, plan, root=r)
            open_toq_by_root[r] = open_toq

            cache_key = ("collapsed_question_open_toq", r, open_toq.inputs)

            if cache_key in cache:
                cq = cache[cache_key]
            else:
                cq = collapser(open_toq, context=context)
                cache[cache_key] = cq

            collapsed_question_by_root[r] = cq

        collapsed = apply_collapse_plan(
            toq,
            plan,
            collapsed_question_by_root,
        )

        # Attach provenance if supported
        if hasattr(collapsed, "open_toq_by_root"):
            collapsed = CollapsedToQ(
                toq=collapsed.toq,
                plan=collapsed.plan,
                removed_nodes=collapsed.removed_nodes,
                collapsed_question_by_root=collapsed.collapsed_question_by_root,
                component_roots=collapsed.component_roots,
                open_toq_by_root=open_toq_by_root,
            )

        trace = evaluate_toq(
            collapsed.toq,
            answerer=answerer,
            substituter=substituter,
            context=context,
        )

        root_answer = trace.answer[collapsed.toq.root_id]
        normalized = (
            normalizer(root_answer.text) if normalizer is not None else None
        )

        runs.append(
            RunRecord(
                plan=plan,
                collapsed=collapsed,
                trace=trace,
                root_answer=root_answer,
                normalized_root=normalized,
            )
        )

    return ConsistencyReport(
        base_trace=base_trace,
        base_root_answer=base_root_answer,
        runs=runs,
        summary={},  # metrics layer fills this later
    )

def run_consistency_check_from_question(
    question: str,
    *,
    decomposer: QuestionDecomposer,
    answerer: Answerer,
    collapser: Collapser,
    normalizer: Optional[Normalizer] = None,
    substituter: Optional[Substituter] = None,
    context: Optional[str] = None,
    plan_opts: Optional[Mapping[str, Any]] = None,
    cache: Optional[MutableMapping[tuple, str]] = None,
) -> ConsistencyReport:
    """
    End-to-end entry point:
      1) Decompose raw question into a ToQ
      2) Run standard consistency check
    """
    toq = decomposer(question, context=context)
    toq.validate()

    return run_consistency_check(
        toq,
        answerer=answerer,
        collapser=collapser,
        normalizer=normalizer,
        substituter=substituter,
        context=context,
        plan_opts=plan_opts,
        cache=cache,
    )

# %%
