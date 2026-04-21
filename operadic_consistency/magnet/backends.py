"""
LLM backend abstraction for the MAGNET predictor.

Kitware's TA2 harness will provide its own inference setup (vllm, HELM-managed
endpoints, etc.), so the predictor must not hardcode any one API. This module
defines a small ``LLMBackend`` Protocol that any completion backend can
implement, plus a ``TogetherBackend`` reference implementation used for local
development and calibration runs.

Adding a new backend
--------------------
Implement a class with a ``complete(prompt, *, max_tokens, temperature, stop)
-> str`` method. No inheritance is required; the Protocol is structural.

Example::

    class VLLMBackend:
        def __init__(self, endpoint: str, model: str):
            self.endpoint = endpoint
            self.model = model

        def complete(self, prompt, *, max_tokens=128, temperature=0.0, stop=None):
            # ... POST to vllm server, return generated text ...
            return text

A backend is bound to a single model. Pass one backend per model role
(e.g. an answerer backend and a decomposer backend) to the predictor.
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence, runtime_checkable

log = logging.getLogger(__name__)


@runtime_checkable
class LLMBackend(Protocol):
    """Structural protocol for LLM completion backends.

    A backend is bound to one model and exposes a single ``complete`` method.
    Backends are expected to return an empty string on failure rather than
    raise, so that the caller can treat a failed decomposition or sub-question
    answer as a non-fatal "inconsistent" result.
    """

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop: Optional[Sequence[str]] = None,
    ) -> str: ...


class TogetherBackend:
    """``LLMBackend`` implementation that calls the Together.ai completions API.

    Parameters
    ----------
    model:
        Together.ai model ID (e.g. ``"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"``).
    api_key:
        Together.ai API key.
    default_stop:
        Default stop sequences used when the caller does not pass ``stop``.
        Tuned for the short-answer prompts used by the predictor.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        default_stop: Sequence[str] = ("\n\n", "###"),
    ):
        self.model = model
        self.api_key = api_key
        self.default_stop = tuple(default_stop)
        self._client = None  # Lazily constructed so the `together` import is optional.

    def _get_client(self):
        if self._client is None:
            import together  # deferred so the import is only required at call time
            self._client = together.Together(api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        try:
            client = self._get_client()
            resp = client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=list(stop) if stop is not None else list(self.default_stop),
            )
            return resp.choices[0].text.strip()
        except Exception as e:  # noqa: BLE001 — intentional broad catch
            log.warning("Together.ai call failed (model=%s): %s", self.model, e)
            return ""
