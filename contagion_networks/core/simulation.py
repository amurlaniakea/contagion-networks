"""Contagion simulation engine."""

from __future__ import annotations

import time
from typing import Any
from .models import Agent, ContagionResult, Evaluation, PropagationRegime
from .contagion_matrix import ContagionMatrix


class ContagionSimulator:
    """Simulates evaluator bias propagation in multi-agent systems."""

    def __init__(self, agents: list[Agent] | None = None):
        self.agents = agents or []

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def simulate(
        self,
        num_hops: int = 10,
        topology: str = "fully_connected",
        base_coefficient: float = 0.25,
    ) -> ContagionResult:
        n = len(self.agents)
        if n == 0:
            raise ValueError("No agents in simulation")

        matrix = self._build_matrix(n, topology, base_coefficient)
        initial_biases = [agent.effective_bias for agent in self.agents]
        bias_history = matrix.propagate(initial_biases, num_hops)
        evaluations = self._generate_evaluations(matrix, bias_history)
        final_biases = bias_history[-1]
        effective_contagion = sum(final_biases) / n if n > 0 else 0.0

        return ContagionResult(
            num_agents=n, num_hops=num_hops,
            contagion_matrix=matrix.matrix,
            spectral_radius=matrix.spectral_radius(),
            regime=matrix.get_regime(),
            evaluations=evaluations,
            agent_biases={a.agent_id: a.effective_bias for a in self.agents},
            effective_contagion=effective_contagion,
        )

    def _build_matrix(self, n, topology, coefficient):
        if topology == "fully_connected":
            return ContagionMatrix.homogeneous(n, coefficient)
        elif topology == "chain":
            return ContagionMatrix.chain(n, coefficient)
        elif topology == "ring":
            return ContagionMatrix.ring(n, coefficient)
        elif topology == "star":
            cm = ContagionMatrix(n)
            for i in range(1, n):
                cm.matrix[0][i] = coefficient
                cm.matrix[i][0] = coefficient * 0.5
            return cm
        return ContagionMatrix.homogeneous(n, coefficient)

    def _generate_evaluations(self, matrix, bias_history):
        evaluations = []
        n = len(self.agents)
        for hop, biases in enumerate(bias_history):
            for i in range(n):
                for j in range(n):
                    if i != j and matrix.matrix[i][j] > 0:
                        coeff = matrix.matrix[i][j]
                        bias_injected = coeff * biases[i]
                        evaluations.append(Evaluation(
                            evaluator_id=self.agents[i].agent_id,
                            target_id=self.agents[j].agent_id,
                            score=max(0, min(1, 0.5 + bias_injected)),
                            bias_injected=bias_injected,
                            hop_distance=hop,
                        ))
        return evaluations
