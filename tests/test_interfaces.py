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
# Dev setup
# %load_ext autoreload
# %autoreload 2

# %%
from operadic_consistency.core.toq_types import NodeId, ToQ, ToQNode
from operadic_consistency.core.interfaces import Answer

# %%
# ---- tests for core/interfaces.py (runtime sanity) ----

def toy_answerer(q: str, *, context=None) -> Answer:
    return Answer(text=f"ans({q})", meta={"context": context})

def toy_collapser(toq: ToQ, *, root_id: NodeId, context=None) -> str:
    return f"COLLAPSE(root={root_id})"

def toy_normalizer(s: str) -> str:
    return s.strip().lower()

# smoke: can we call them in the expected way?
a = toy_answerer("What is 2+2?", context="ctx")
assert isinstance(a, Answer)
assert isinstance(a.text, str)

cq = toy_collapser(ToQ(nodes={1: ToQNode(1, "Q?", None)}, root_id=1), root_id=1)
assert isinstance(cq, str)

ns = toy_normalizer("  HeLLo  ")
assert ns == "hello"

print("interfaces runtime smoke tests passed")

# %%
