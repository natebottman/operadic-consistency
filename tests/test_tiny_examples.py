"""
End-to-end integration tests using tiny toy implementations.
"""
from typing import Optional

from operadic_consistency.core.toq_types import ToQ, ToQNode, OpenToQ
from operadic_consistency.core.interfaces import Answer
from operadic_consistency.core.evaluate import evaluate_toq
from operadic_consistency.core.consistency import run_consistency_check
from operadic_consistency.core.metrics import agreement_rate


class TinyAnswerer:
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
    def __call__(self, open_toq: OpenToQ, *, context: Optional[str] = None) -> str:
        root_text = open_toq.toq.nodes[open_toq.root_id].text
        if open_toq.inputs:
            return root_text
        texts = {n.text for n in open_toq.toq.nodes.values()}
        if (
            "When did WW2 end?" in texts
            and "Who was President at time [A1]?" in texts
        ):
            return "Who was President when WW2 ended?"
        return root_text


def _ww2_toq():
    nodes = {
        1: ToQNode(1, "When did WW2 end?", parent=2),
        2: ToQNode(2, "Who was President at time [A1]?", parent=None),
    }
    return ToQ(nodes=nodes, root_id=2)


def test_ww2_baseline_answer():
    toq = _ww2_toq()
    answerer = TinyAnswerer()
    trace = evaluate_toq(toq, answerer=answerer)
    assert trace.answer[1].text == "1945"
    assert trace.answer[2].text == "Harry Truman"


def test_ww2_consistency_report():
    toq = _ww2_toq()
    answerer = TinyAnswerer()
    collapser = TinyCollapser()

    report = run_consistency_check(
        toq,
        answerer=answerer,
        collapser=collapser,
        plan_opts={"include_empty": True},
    )

    # 1 edge => 2 plans => 2 runs
    assert len(report.runs) == 2
    assert report.base_root_answer.text == "Harry Truman"

    # All runs should agree: perfect consistency
    rate = agreement_rate(report, use_normalized=False)
    assert rate == 1.0

    for run in report.runs:
        assert run.root_answer.text == "Harry Truman"
