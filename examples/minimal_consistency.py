"""Minimal runnable operadic_consistency example."""

from __future__ import annotations

from typing import Optional

from operadic_consistency import ToQ, ToQNode, run_consistency_check
from operadic_consistency.core.interfaces import Answer
from operadic_consistency.core.transforms import OpenToQ


class TinyAnswerer:
    """
    Mocks an LLM that can provide an answer for the specific questions in this example
    """
    def __call__(self, question: str, *, context: Optional[str] = None) -> Answer:
        q = question.lower()
        if "when did ww2 end" in q:
            return Answer("1945")
        if "president" in q:
            return Answer("Harry Truman")
        return Answer("UNKNOWN")


class TinyCollapser:
    def __call__(self, open_toq: OpenToQ, *, context: Optional[str] = None) -> str:
        # Keep this example simple: use component root question text as-is.
        return open_toq.toq.nodes[open_toq.root_id].text


def main() -> None:
    toq = ToQ(
        nodes={
            1: ToQNode(1, "When did WW2 end?", parent=2),
            2: ToQNode(2, "Who was President at time [A1]?", parent=None),
        },
        root_id=2,
    )

    report = run_consistency_check(
        toq,
        answerer=TinyAnswerer(),
        collapser=TinyCollapser(),
        plan_opts={"include_empty": True},
    )

    print(f"baseline: {report.base_root_answer.text}")
    for idx, run in enumerate(report.runs, start=1):
        print(f"run {idx}: cut_edges={run.plan.cut_edges} answer={run.root_answer.text}")


if __name__ == "__main__":
    main()
