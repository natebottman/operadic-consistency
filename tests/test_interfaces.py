from operadic_consistency.core.toq_types import ToQ, ToQNode
from operadic_consistency.core.interfaces import Answer


def _toy_answerer(q: str, *, context=None) -> Answer:
    return Answer(text=f"ans({q})", meta={"context": context})


def _toy_normalizer(s: str) -> str:
    return s.strip().lower()


def test_answer_is_answer():
    a = _toy_answerer("What is 2+2?", context="ctx")
    assert isinstance(a, Answer)
    assert isinstance(a.text, str)
    assert a.text == "ans(What is 2+2?)"


def test_normalizer():
    assert _toy_normalizer("  HeLLo  ") == "hello"


def test_answer_no_meta():
    a = Answer(text="yes")
    assert a.text == "yes"
    assert a.meta is None
