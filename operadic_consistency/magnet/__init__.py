"""
MAGNET integration for operadic consistency.

Provides an OperadicConsistencyPredictor that implements the MAGNET RunPredictor
interface using operadic consistency as a reliability signal.
"""
from operadic_consistency.magnet.predictor import OperadicConsistencyPredictor

__all__ = ["OperadicConsistencyPredictor"]
