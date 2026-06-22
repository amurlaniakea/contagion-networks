"""Contagion Networks core modules."""

from .models import (
    Agent,
    ContagionResult,
    Evaluation,
    EvaluatorBiasProfile,
    PropagationRegime,
)
from .contagion_matrix import ContagionMatrix
from .simulation import ContagionSimulator
from .analysis import ContagionAnalyzer
from .mitigation import MitigationEngine

__all__ = [
    "Agent",
    "ContagionResult",
    "Evaluation",
    "EvaluatorBiasProfile",
    "PropagationRegime",
    "ContagionMatrix",
    "ContagionSimulator",
    "ContagionAnalyzer",
    "MitigationEngine",
]
