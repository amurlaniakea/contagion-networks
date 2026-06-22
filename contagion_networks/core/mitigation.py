"""Contagion mitigation strategies."""

from __future__ import annotations

from .contagion_matrix import ContagionMatrix
from .models import PropagationRegime


class MitigationEngine:
    """Implements mitigation strategies for contagion networks."""

    @staticmethod
    def committee_mitigation(
        matrix: ContagionMatrix, committee_size: int = 3, diversity_factor: float = 0.5
    ) -> ContagionMatrix:
        n = matrix.num_agents
        mitigated = ContagionMatrix(n)
        reduction = 1.0 / (committee_size * (1 + diversity_factor))
        for i in range(n):
            for j in range(n):
                if i != j:
                    mitigated.matrix[i][j] = matrix.matrix[i][j] * reduction
        return mitigated

    @staticmethod
    def topology_mitigation(
        num_agents: int, coefficient: float, target: PropagationRegime = PropagationRegime.SUPPRESSION
    ) -> ContagionMatrix:
        if target == PropagationRegime.SUPPRESSION:
            return ContagionMatrix.chain(num_agents, coefficient * 0.5)
        elif target == PropagationRegime.PERSISTENCE:
            return ContagionMatrix.ring(num_agents, coefficient * 0.7)
        return ContagionMatrix.homogeneous(num_agents, coefficient)

    @staticmethod
    def critical_committee_size(
        matrix: ContagionMatrix, target: PropagationRegime = PropagationRegime.SUPPRESSION, max_k: int = 10
    ) -> int:
        for k in range(1, max_k + 1):
            m = MitigationEngine.committee_mitigation(matrix, k)
            if m.get_regime() == target:
                return k
        return -1
