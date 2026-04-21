"""
MAGNET integration for operadic consistency.

Provides an OperadicConsistencyPredictor that implements the MAGNET RunPredictor
interface using operadic consistency as a reliability signal. All LLM inference
goes through a swappable ``LLMBackend`` so the TA2 harness can supply its own
completion endpoint.
"""
from operadic_consistency.magnet.backends import LLMBackend, TogetherBackend
from operadic_consistency.magnet.predictor import OperadicConsistencyPredictor

__all__ = ["OperadicConsistencyPredictor", "LLMBackend", "TogetherBackend"]
