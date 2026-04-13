from typing import Optional

import pytest

from operadic_consistency.core.toq_types import ToQ, ToQNode
from operadic_consistency.core.interfaces import Answer, QuestionDecomposer
from operadic_consistency.core.consistency import (
    run_consistency_check,
    run_consistency_check_from_question,
)


class TinyAnswerer:
    def __call__(self, question: str, *, context: Optional[str] = None) -> Answer:
        q = question.lower()
        if "when did ww2 end" in q:
            return Answer("1945")
        if "president at time" in q or "president when ww2 ended" in q:
            return Answer("Harry Truman")
        return Answer("UNKNOWN")


class TinyCollapser:
    def __call__(self, open_toq, *, context: Optional[str] = None) -> str:
        return open_toq.toq.nodes[open_toq.root_id].text


class TinyDecomposer(QuestionDecomposer):
    def __call__(self, question: str, *, context: Optional[str] = None) -> ToQ:
        nodes = {
            1: ToQNode(1, "When did WW2 end?", parent=2),
            2: ToQNode(2, "Who was President at time [A1]?", parent=None),
        }
        return ToQ(nodes=nodes, root_id=2)


def _root_answers(report) -> list[str]:
    return [run.root_answer.text for run in report.runs]


def _cut_edges(report) -> list[tuple[int, ...]]:
    return [tuple(run.plan.cut_edges) for run in report.runs]


def test_run_from_question_matches_direct_toq():
    q = "Who was President when WW2 ended?"

    decomposer = TinyDecomposer()
    toq = decomposer(q)
    answerer = TinyAnswerer()
    collapser = TinyCollapser()

    rep_direct = run_consistency_check(
        toq,
        answerer=answerer,
        collapser=collapser,
        plan_opts={"include_empty": True},
        cache={},
    )
    rep_from_q = run_consistency_check_from_question(
        q,
        decomposer=decomposer,
        answerer=answerer,
        collapser=collapser,
        plan_opts={"include_empty": True},
        cache={},
    )

    assert rep_direct.base_root_answer.text == rep_from_q.base_root_answer.text
    assert _cut_edges(rep_direct) == _cut_edges(rep_from_q)
    assert _root_answers(rep_direct) == _root_answers(rep_from_q)


def test_decomposer_invalid_toq_raises():
    class BadDecomposer(QuestionDecomposer):
        def __call__(self, question: str, *, context: Optional[str] = None) -> ToQ:
            return ToQ(nodes={}, root_id=123)

    with pytest.raises(ValueError):
        run_consistency_check_from_question(
            "anything",
            decomposer=BadDecomposer(),
            answerer=TinyAnswerer(),
            collapser=TinyCollapser(),
            plan_opts={"include_empty": True},
            cache={},
        )
