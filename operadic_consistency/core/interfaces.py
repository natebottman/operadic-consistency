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
# core/interfaces.py

# %%
# Dev setup
# %load_ext autoreload
# %autoreload 2

# %%
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

from operadic_consistency.core.toq_types import ToQ, OpenToQ


# %%
@dataclass(frozen=True)
class Answer:
    text: str
    meta: Optional[Mapping[str, Any]] = None
    # Raw model answer plus optional metadata (logprobs, tokens, etc.)

class Answerer(Protocol):
    def __call__(self, question: str, *, context: Optional[str] = None) -> Answer:
        # Given a fully-instantiated question, return a model answer
        ...

class Collapser(Protocol):
    def __call__(self, open_toq: OpenToQ, *, context: Optional[str] = None) -> str:
        # Given an open ToQ, produce a single question summarizing it.
        # The returned question may still contain placeholders [A<input_id>] for open_toq.inputs.
        ...

class Normalizer(Protocol):
    def __call__(self, answer_text: str) -> str:
        # Map semantically equivalent answers to a canonical form
        ...

class QuestionDecomposer(Protocol):
    def __call__(self, question: str, *, context: Optional[str] = None) -> ToQ:
        """
        Given a raw question, return a validated ToQ decomposition.
        """
        ...

# %%
