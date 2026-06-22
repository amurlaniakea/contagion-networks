# Contagion Networks

**Evaluator Bias Propagation in Multi-Agent LLM Systems**

Formal framework for measuring how evaluator biases spread across interacting LLM agents.
Based on the paper ["Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems"](https://arxiv.org/abs/2606.20493) (arXiv:2606.20493, Jun 2026).

## The Problem

In multi-agent LLM systems, agents evaluate each other's outputs to guide collaboration.
When an LLM serves as evaluator, its systematic biases propagate through the agent network
like an infection. This can cause:

- **Strategy convergence**: All agents adopt the same strategy, eliminating cognitive diversity
- **Preference collapse**: The network converges to a single evaluation criterion
- **Cascading failures**: Biases amplify across hops (A → B → C → ...)

## The Solution

Contagion Networks provides:

1. **Cross-Agent Contagion Matrix (Γ)**: Quantifies bias propagation between N agents
2. **Propagation Regime Theorem**: Three regimes (suppression/persistence/cascade) governed by spectral radius ρ(Γ)
3. **Cascade Condition**: ρ(Γ) > 1 is the threshold for system-wide cascade
4. **Mitigation Strategies**: Committee-based evaluation, diversity thresholds

## Architecture

```
contagion_networks/
├── core/
│   ├── models.py          # Agent, Evaluator, Network models
│   ├── contagion_matrix.py # Cross-Agent Contagion Matrix Γ
│   ├── simulation.py      # Multi-agent simulation engine
│   ├── analysis.py        # Analysis tools (diversity, attenuation, report)
│   └── mitigation.py      # Mitigation strategies (committee, topology)
└── cli.py                 # CLI interface (simulate, analyze, mitigate)
```

## Key Concepts

### Cross-Agent Contagion Matrix (Γ)

An N×N matrix where Γ[i][j] represents the contagion coefficient from agent i to agent j.
- γ ∈ [0, 1]: 0 = no propagation, 1 = full propagation
- Homogeneous models: γ ≈ 0.15-0.35 (suppression regime)
- Cross-model: γ ≈ 0.85-1.3 (cascade regime)

### Propagation Regimes

| Regime | Spectral Radius | Behavior |
|--------|----------------|----------|
| Suppression | ρ(Γ) < 1 | Bias attenuates across hops |
| Persistence | ρ(Γ) = 1 | Bias maintains constant |
| Cascade | ρ(Γ) > 1 | Bias amplifies system-wide |

### Mitigation: Committee Size

Increasing evaluator committee size from k=1 to k=3 reduces effective contagion by 66.7%
(with default diversity_factor=0.5). The paper reports 72.4% for empirical measurements
on specific model configurations.

## Quick Start

```bash
# Clone
git clone https://github.com/amurlaniakea/contagion-networks.git
cd contagion-networks

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Run simulation
python3 -m contagion_networks simulate --agents 5 --hops 10

# Analyze a saved matrix
python3 -m contagion_networks simulate --agents 5 --hops 10 --output results.json
python3 -m contagion_networks analyze --matrix results.json

# Apply mitigation
python3 -m contagion_networks mitigate --matrix results.json --strategy committee --k 3

# JSON output (all commands)
python3 -m contagion_networks simulate --agents 5 --hops 10 --json
```

## License

AGPL-3.0 — Copyright (C) 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>

## References

- Liu Zewen. "Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems."
  arXiv:2606.20493, Jun 2026.
