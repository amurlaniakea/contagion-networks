"""Contagion Networks data models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PropagationRegime(str, Enum):
    SUPPRESSION = "suppression"
    PERSISTENCE = "persistence"
    CASCADE = "cascade"


class EvaluatorBiasProfile(str, Enum):
    STRUCTURED = "structured"
    BALANCED = "balanced"
    EVIDENCE_BASED = "evidence_based"
    CREATIVE = "creative"
    CONCISE = "concise"


@dataclass
class Agent:
    agent_id: str
    model_name: str
    bias_profile: EvaluatorBiasProfile
    bias_strength: float = 0.5
    absorbed_bias: float = 0.0
    generation_count: int = 0
    evaluation_count: int = 0

    @property
    def effective_bias(self) -> float:
        return min(1.0, self.bias_strength + self.absorbed_bias)


@dataclass
class Evaluation:
    evaluator_id: str
    target_id: str
    score: float
    bias_injected: float
    hop_distance: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ContagionResult:
    num_agents: int
    num_hops: int
    contagion_matrix: list[list[float]]
    spectral_radius: float
    regime: PropagationRegime
    evaluations: list[Evaluation]
    agent_biases: dict[str, float]
    effective_contagion: float
    mitigation_applied: bool = False
    committee_size: int = 1
