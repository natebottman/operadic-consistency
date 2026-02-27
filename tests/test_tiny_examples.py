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
# test_tiny_examples.ipynb
# Verbose end-to-end demonstrations for operadic_consistency

# %%
# Dev setup
# %load_ext autoreload
# %autoreload 2

# %%
from typing import Optional

from operadic_consistency.core.toq_types import ToQ, ToQNode, OpenToQ
from operadic_consistency.core.interfaces import Answer
from operadic_consistency.core.evaluate import evaluate_toq
from operadic_consistency.core.consistency import run_consistency_check

# %%
# -------------------------------------------------------------------
# Tiny toy implementations (OpenToQ-aware)
# -------------------------------------------------------------------

from typing import Optional

from operadic_consistency.core.toq_types import ToQ, ToQNode
from operadic_consistency.core.interfaces import Answer
from operadic_consistency.core.evaluate import evaluate_toq
from operadic_consistency.core.consistency import run_consistency_check
from operadic_consistency.core.transforms import OpenToQ


class TinyAnswerer:
    """
    Deterministic toy answerer for demonstrations.
    """
    def __call__(self, question: str, *, context: Optional[str] = None) -> Answer:
        q = question.lower()

        if "when did ww2 end" in q:
            return Answer("1945")

        if "president at time" in q:
            return Answer("Harry Truman")

        if "president when ww2 ended" in q:
            return Answer("Harry Truman")

        if "wife" in q:
            return Answer("Bess Truman")

        return Answer("UNKNOWN")


class TinyCollapser:
    """
    Brute-force-ish rule:
      - If this OpenToQ has external inputs, DON'T fuse: return the root template as-is.
      - If it has no inputs (closed component), do a tiny custom fuse for this example.
    """
    def __call__(self, open_toq: OpenToQ, *, context: Optional[str] = None) -> str:
        root_text = open_toq.toq.nodes[open_toq.root_id].text

        # Open component => must keep placeholders
        if open_toq.inputs:
            return root_text

        # Closed component => we can fuse (toy logic, specific to this demo)
        # If we see the 2-node WW2->President chain, fuse to a nicer question:
        texts = {n.text for n in open_toq.toq.nodes.values()}
        if (
            "When did WW2 end?" in texts
            and "Who was President at time [A1]?" in texts
            and open_toq.root_id in open_toq.toq.nodes
        ):
            return "Who was President when WW2 ended?"

        # Fallback: just return root as-is
        return root_text


# -------------------------------------------------------------------
# Pretty-print helpers
# -------------------------------------------------------------------

def print_toq(toq: ToQ, title: str):
    print(f"\n=== {title} ===")
    for nid, node in sorted(toq.nodes.items()):
        print(f"Node {nid}:")
        print(f"  text   = {node.text}")
        print(f"  parent = {node.parent}")
    print(f"root_id = {toq.root_id}")


def print_eval_trace(trace):
    print("\nEvaluation trace (leaves -> root):")
    for nid in trace.rendered_question:
        print(f"  Node {nid}")
        print(f"    rendered question: {trace.rendered_question[nid]}")
        print(f"    answer: {trace.answer[nid].text}")


def print_consistency_report(report):
    print("\n=== Consistency runs ===")
    for i, run in enumerate(report.runs):
        print(f"\n--- Run {i+1} ---")
        print(f"cut_edges = {run.plan.cut_edges}")

        print("collapsed ToQ nodes:")
        for nid, node in sorted(run.collapsed.toq.nodes.items()):
            print(f"  Node {nid}: parent={node.parent}, text={node.text}")

        print(f"root answer = {run.root_answer.text}")

    print("\n=== Summary ===")
    print(f"num runs = {len(report.runs)}")
    print(f"baseline answer = {report.base_root_answer.text}")


# -------------------------------------------------------------------
# Demo: 2-node chain
# -------------------------------------------------------------------

def demo_president_when_ww2_ended():
    print("\n###############################")
    print("Demo: Who was President when WW2 ended?")
    print("###############################")

    nodes = {
        1: ToQNode(1, "When did WW2 end?", parent=2),
        2: ToQNode(2, "Who was President at time [A1]?", parent=None),
    }
    toq = ToQ(nodes=nodes, root_id=2)

    print_toq(toq, "Original ToQ")

    answerer = TinyAnswerer()
    collapser = TinyCollapser()

    trace = evaluate_toq(toq, answerer=answerer)
    print_eval_trace(trace)

    report = run_consistency_check(
        toq,
        answerer=answerer,
        collapser=collapser,
        plan_opts={"include_empty": True},
    )
    print_consistency_report(report)


demo_president_when_ww2_ended()

# %%
