"""Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems.

Based on: arXiv:2606.20493 (Jun 2026)
Copyright (C) 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
License: AGPL-3.0
"""

from .core.models import (
    Agent,
    ContagionResult,
    Evaluation,
    EvaluatorBiasProfile,
    PropagationRegime,
)
from .core.contagion_matrix import ContagionMatrix

__version__ = "0.1.0"
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


# Re-export simulator, analyzer, and mitigation for convenience
from .core.simulation import ContagionSimulator
from .core.analysis import ContagionAnalyzer
from .core.mitigation import MitigationEngine
