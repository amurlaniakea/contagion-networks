"""Contagion Networks tests."""

import json
import pytest

from contagion_networks import (
    Agent,
    ContagionAnalyzer,
    ContagionMatrix,
    ContagionResult,
    ContagionSimulator,
    EvaluatorBiasProfile,
    MitigationEngine,
    PropagationRegime,
)
from contagion_networks.core.models import Evaluation


class TestContagionMatrix:
    def test_create_matrix(self):
        cm = ContagionMatrix(3)
        assert cm.num_agents == 3
        assert len(cm.matrix) == 3

    def test_set_get_contagion(self):
        cm = ContagionMatrix(3)
        cm.set_contagion(0, 1, 0.5)
        assert cm.get_contagion(0, 1) == 0.5

    def test_spectral_radius_zero(self):
        cm = ContagionMatrix(3)
        assert cm.spectral_radius() == 0.0

    def test_spectral_radius_homogeneous(self):
        cm = ContagionMatrix.homogeneous(3, 0.3)
        rho = cm.spectral_radius()
        assert rho > 0

    def test_regime_suppression(self):
        cm = ContagionMatrix.homogeneous(3, 0.1)
        assert cm.get_regime() == PropagationRegime.SUPPRESSION

    def test_regime_cascade(self):
        cm = ContagionMatrix.homogeneous(5, 0.9)
        assert cm.get_regime() == PropagationRegime.CASCADE

    def test_chain_topology(self):
        cm = ContagionMatrix.chain(5, 0.3)
        assert cm.matrix[0][1] == 0.3
        assert cm.matrix[1][0] == 0.0  # One-directional

    def test_ring_topology(self):
        cm = ContagionMatrix.ring(4, 0.3)
        assert cm.matrix[0][1] == 0.3
        assert cm.matrix[3][0] == 0.3  # Wraps around

    def test_propagate(self):
        cm = ContagionMatrix.homogeneous(3, 0.2)
        initial = [0.5, 0.3, 0.1]
        history = cm.propagate(initial, 5)
        assert len(history) == 6  # Initial + 5 hops
        assert history[0] == initial

    def test_to_from_dict(self):
        cm = ContagionMatrix.homogeneous(3, 0.3)
        data = cm.to_dict()
        cm2 = ContagionMatrix.from_dict(data)
        assert cm2.num_agents == cm.num_agents
        assert cm2.matrix == cm.matrix


class TestContagionSimulator:
    def test_simulate_basic(self):
        agents = [
            Agent("a1", "model", EvaluatorBiasProfile.STRUCTURED, 0.5),
            Agent("a2", "model", EvaluatorBiasProfile.BALANCED, 0.3),
            Agent("a3", "model", EvaluatorBiasProfile.EVIDENCE_BASED, 0.4),
        ]
        sim = ContagionSimulator(agents)
        result = sim.simulate(num_hops=5, topology="fully_connected", base_coefficient=0.25)
        assert result.num_agents == 3
        assert result.num_hops == 5
        assert result.regime in PropagationRegime

    def test_simulate_chain(self):
        agents = [
            Agent(f"a{i}", "model", EvaluatorBiasProfile.STRUCTURED, 0.5)
            for i in range(5)
        ]
        sim = ContagionSimulator(agents)
        result = sim.simulate(num_hops=10, topology="chain", base_coefficient=0.3)
        assert result.regime == PropagationRegime.SUPPRESSION

    def test_simulate_no_agents(self):
        sim = ContagionSimulator()
        with pytest.raises(ValueError):
            sim.simulate()


class TestMitigationEngine:
    def test_committee_mitigation(self):
        cm = ContagionMatrix.homogeneous(5, 0.5)
        mitigated = MitigationEngine.committee_mitigation(cm, committee_size=3)
        # Mitigated should have lower coefficients
        assert mitigated.spectral_radius() < cm.spectral_radius()

    def test_committee_reduces_contagion(self):
        cm = ContagionMatrix.homogeneous(3, 0.8)
        original_eff = cm.spectral_radius()
        mitigated = MitigationEngine.committee_mitigation(cm, committee_size=3)
        assert mitigated.spectral_radius() < original_eff

    def test_topology_mitigation_suppression(self):
        cm = MitigationEngine.topology_mitigation(
            5, 0.5, PropagationRegime.SUPPRESSION
        )
        assert cm.get_regime() == PropagationRegime.SUPPRESSION

    def test_critical_committee_size(self):
        cm = ContagionMatrix.homogeneous(5, 0.8)
        k = MitigationEngine.critical_committee_size(cm)
        assert k > 0

    def test_mitigation_achieves_suppression(self):
        cm = ContagionMatrix.homogeneous(3, 0.9)
        mitigated = MitigationEngine.committee_mitigation(cm, committee_size=3)
        assert mitigated.get_regime() == PropagationRegime.SUPPRESSION


class TestContagionAnalyzer:
    def test_diversity_index_uniform(self):
        biases = {"a1": 0.5, "a2": 0.5, "a3": 0.5}
        diversity = ContagionAnalyzer.compute_diversity_index(biases)
        assert diversity == 0.0

    def test_diversity_index_diverse(self):
        biases = {"a1": 0.1, "a2": 0.5, "a3": 0.9}
        diversity = ContagionAnalyzer.compute_diversity_index(biases)
        assert diversity > 0.5

    def test_bias_attenuation_suppression(self):
        initial = [0.5, 0.5, 0.5]
        final = [0.3, 0.3, 0.3]
        ratio = ContagionAnalyzer.compute_bias_attenuation(initial, final)
        assert ratio < 1.0

    def test_bias_attenuation_cascade(self):
        initial = [0.3, 0.3, 0.3]
        final = [0.8, 0.8, 0.8]
        ratio = ContagionAnalyzer.compute_bias_attenuation(initial, final)
        assert ratio > 1.0

    def test_hop_decay(self):
        from contagion_networks import Evaluation
        evaluations = [
            Evaluation("a1", "a2", 0.7, 0.3, 0),
            Evaluation("a1", "a2", 0.6, 0.2, 1),
            Evaluation("a1", "a2", 0.55, 0.1, 2),
        ]
        decay = ContagionAnalyzer.analyze_hop_decay(evaluations)
        assert len(decay) == 3
        # Bias should decrease with hop distance
        assert decay[0] > decay[1] > decay[2]

    def test_generate_report(self):
        result = ContagionResult(
            num_agents=3,
            num_hops=5,
            contagion_matrix=[[0, 0.3, 0.2], [0.3, 0, 0.1], [0.2, 0.1, 0]],
            spectral_radius=0.5,
            regime=PropagationRegime.SUPPRESSION,
            evaluations=[],
            agent_biases={"a1": 0.5, "a2": 0.3, "a3": 0.4},
            effective_contagion=0.3,
        )
        report = ContagionAnalyzer.generate_report(result)
        assert "# Contagion Networks Analysis Report" in report
        assert "SUPPRESSION" in report


class TestAgent:
    def test_effective_bias(self):
        agent = Agent("a1", "model", EvaluatorBiasProfile.STRUCTURED, 0.5)
        assert agent.effective_bias == 0.5

    def test_effective_bias_with_absorbed(self):
        agent = Agent("a1", "model", EvaluatorBiasProfile.STRUCTURED, 0.5, 0.3)
        assert agent.effective_bias == 0.8

    def test_effective_bias_capped(self):
        agent = Agent("a1", "model", EvaluatorBiasProfile.STRUCTURED, 0.7, 0.5)
        assert agent.effective_bias == 1.0  # Capped at 1.0
