"""Contagion matrix operations."""
from ..core.models import *
import math

__all__ = ["ContagionMatrix"]


class ContagionMatrix:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.matrix: list[list[float]] = [[0.0] * num_agents for _ in range(num_agents)]

    def set_contagion(self, from_agent: int, to_agent: int, coefficient: float):
        self.matrix[from_agent][to_agent] = max(0.0, min(1.0, coefficient))

    def get_contagion(self, from_agent: int, to_agent: int) -> float:
        return self.matrix[from_agent][to_agent]

    def spectral_radius(self) -> float:
        n = self.num_agents
        if n == 0:
            return 0.0
        b = [1.0] * n
        b_new = [0.0] * n
        for _ in range(100):
            b_new = [0.0] * n
            for i in range(n):
                for j in range(n):
                    b_new[i] += self.matrix[i][j] * b[j]
            norm = math.sqrt(sum(x ** 2 for x in b_new))
            if norm < 1e-10:
                return 0.0
            b = [x / norm for x in b_new]
        # Rayleigh quotient: λ = bᵀ(Mb) = Σ b_new[i] * b[i] (b is already normalized)
        eigenvalue = sum(b_new[i] * b[i] for i in range(n))
        return abs(eigenvalue)

    def get_regime(self) -> PropagationRegime:
        rho = self.spectral_radius()
        if rho < 0.95:
            return PropagationRegime.SUPPRESSION
        elif rho < 1.05:
            return PropagationRegime.PERSISTENCE
        else:
            return PropagationRegime.CASCADE

    def propagate(self, initial_biases: list[float], hops: int) -> list[list[float]]:
        biases = [initial_biases[:]]
        current = initial_biases[:]
        for _ in range(hops):
            new_biases = [0.0] * self.num_agents
            for i in range(self.num_agents):
                new_biases[i] = current[i]
                for j in range(self.num_agents):
                    if i != j:
                        absorbed = self.matrix[j][i] * current[j]
                        new_biases[i] = min(1.0, new_biases[i] + absorbed)
            current = new_biases
            biases.append(current[:])
        return biases

    def to_dict(self):
        return {"num_agents": self.num_agents, "matrix": self.matrix,
                "spectral_radius": round(self.spectral_radius(), 4),
                "regime": self.get_regime().value}

    @classmethod
    def from_dict(cls, data):
        cm = cls(data["num_agents"])
        cm.matrix = data["matrix"]
        return cm

    @classmethod
    def homogeneous(cls, n, coeff):
        cm = cls(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    cm.matrix[i][j] = coeff
        return cm

    @classmethod
    def chain(cls, n, coeff):
        cm = cls(n)
        for i in range(n - 1):
            cm.matrix[i][i + 1] = coeff
        return cm

    @classmethod
    def ring(cls, n, coeff):
        cm = cls(n)
        for i in range(n):
            cm.matrix[i][(i + 1) % n] = coeff
        return cm
