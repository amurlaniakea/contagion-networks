"""Contagion analysis tools."""

from __future__ import annotations

import math
from typing import Any
from .models import ContagionResult


class ContagionAnalyzer:
    """Analyzes contagion simulation results."""

    @staticmethod
    def compute_diversity_index(agent_biases: dict[str, float]) -> float:
        biases = list(agent_biases.values())
        if not biases:
            return 0.0
        mean_bias = sum(biases) / len(biases)
        variance = sum((b - mean_bias) ** 2 for b in biases) / len(biases)
        return min(1.0, math.sqrt(variance) * 2)

    @staticmethod
    def compute_bias_attenuation(initial: list[float], final: list[float]) -> float:
        s = sum(initial)
        return sum(final) / s if s > 1e-10 else 0.0

    @staticmethod
    def analyze_hop_decay(evaluations: list) -> list[float]:
        if not evaluations:
            return []
        max_hop = max(e.hop_distance for e in evaluations)
        hop_biases: dict[int, list[float]] = {}
        for e in evaluations:
            hop_biases.setdefault(e.hop_distance, []).append(e.bias_injected)
        return [sum(hop_biases.get(h, [0])) / len(hop_biases.get(h, [1])) for h in range(max_hop + 1)]

    @staticmethod
    def generate_report(result: ContagionResult) -> str:
        lines = [
            "# Contagion Networks Analysis Report",
            f"**Agents:** {result.num_agents} | **Hops:** {result.num_hops}",
            f"**Spectral Radius:** {result.spectral_radius:.4f} | **Regime:** {result.regime.value.upper()}",
            f"**Effective Contagion:** {result.effective_contagion:.4f}", "",
            "## Agent Biases", "",
        ]
        for aid, bias in result.agent_biases.items():
            lines.append(f"- **{aid}**: {bias:.4f}")
        lines.extend(["", "## Contagion Matrix", ""])
        n = result.num_agents
        lines.append("| | " + " | ".join(f"A{i}" for i in range(n)) + " |")
        lines.append("|" + "---|" * (n + 1))
        for i in range(n):
            lines.append(f"| A{i} | " + " | ".join(f"{result.contagion_matrix[i][j]:.3f}" for j in range(n)) + " |")
        lines.extend(["", "## Interpretation", ""])
        if result.regime.value == "suppression":
            lines.append("- **Suppression**: Biases attenuate. Low risk.")
        elif result.regime.value == "persistence":
            lines.append("- **Persistence**: Biases maintain. Medium risk.")
        else:
            lines.append("- **CASCADE**: Biases amplify! HIGH RISK.")
        lines.append("")
        return "\n".join(lines)
