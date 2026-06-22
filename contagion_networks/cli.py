"""CLI for Contagion Networks: simulate, analyze, mitigate."""

from __future__ import annotations

import argparse
import json
import sys

from contagion_networks import (
    Agent,
    ContagionAnalyzer,
    ContagionMatrix,
    ContagionSimulator,
    EvaluatorBiasProfile,
    MitigationEngine,
)


def cmd_simulate(args):
    """Run a contagion simulation."""
    agents = [
        Agent(
            agent_id=f"a{i}",
            model_name="model",
            bias_profile=EvaluatorBiasProfile.STRUCTURED,
            bias_strength=0.3 + (i * 0.1),
        )
        for i in range(args.agents)
    ]
    sim = ContagionSimulator(agents)
    result = sim.simulate(
        num_hops=args.hops,
        topology=args.topology,
        base_coefficient=args.coefficient,
    )

    if args.json:
        output = {
            "num_agents": result.num_agents,
            "num_hops": result.num_hops,
            "spectral_radius": round(result.spectral_radius, 4),
            "regime": result.regime.value,
            "effective_contagion": round(result.effective_contagion, 4),
            "agent_biases": result.agent_biases,
            "contagion_matrix": result.contagion_matrix,
        }
        print(json.dumps(output, indent=2))
    else:
        report = ContagionAnalyzer.generate_report(result)
        print(report)

    if args.output:
        data = {
            "num_agents": result.num_agents,
            "num_hops": result.num_hops,
            "spectral_radius": result.spectral_radius,
            "regime": result.regime.value,
            "effective_contagion": result.effective_contagion,
            "agent_biases": result.agent_biases,
            "contagion_matrix": result.contagion_matrix,
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        if not args.json:
            print(f"\nResults saved to: {args.output}")


def cmd_analyze(args):
    """Analyze a contagion matrix from JSON file."""
    with open(args.matrix) as f:
        data = json.load(f)

    # Support both ContagionMatrix.to_dict() format and simulate output
    if "matrix" in data:
        cm = ContagionMatrix.from_dict(data)
    elif "contagion_matrix" in data:
        n = data.get("num_agents", len(data["contagion_matrix"]))
        cm = ContagionMatrix(n)
        cm.matrix = data["contagion_matrix"]
    else:
        print("Error: JSON must contain 'matrix' or 'contagion_matrix' key", file=sys.stderr)
        sys.exit(1)
    rho = cm.spectral_radius()
    regime = cm.get_regime()

    if args.json:
        print(json.dumps({
            "spectral_radius": round(rho, 4),
            "regime": regime.value,
            "matrix": cm.matrix,
        }, indent=2))
    else:
        print(f"Spectral Radius: {rho:.4f}")
        print(f"Regime: {regime.value.upper()}")
        print(f"Agents: {cm.num_agents}")
        print()
        if regime.value == "suppression":
            print("Biases attenuate across hops. Low risk.")
        elif regime.value == "persistence":
            print("Biases maintain constant. Medium risk.")
        else:
            print("CASCADE: Biases amplify system-wide! HIGH RISK.")


def cmd_mitigate(args):
    """Apply mitigation strategies to a contagion matrix."""
    with open(args.matrix) as f:
        data = json.load(f)

    # Support both ContagionMatrix.to_dict() format and simulate output
    if "matrix" in data:
        cm = ContagionMatrix.from_dict(data)
    elif "contagion_matrix" in data:
        n = data.get("num_agents", len(data["contagion_matrix"]))
        cm = ContagionMatrix(n)
        cm.matrix = data["contagion_matrix"]
    else:
        print("Error: JSON must contain 'matrix' or 'contagion_matrix' key", file=sys.stderr)
        sys.exit(1)

    if args.strategy == "committee":
        mitigated = MitigationEngine.committee_mitigation(cm, committee_size=args.k)
    elif args.strategy == "topology":
        mitigated = MitigationEngine.topology_mitigation(
            cm.num_agents, args.coefficient, target=cm.get_regime()
        )
    else:
        print(f"Unknown strategy: {args.strategy}", file=sys.stderr)
        sys.exit(1)

    rho_orig = cm.spectral_radius()
    rho_mit = mitigated.spectral_radius()
    reduction = (1 - rho_mit / rho_orig) * 100 if rho_orig > 0 else 0

    if args.json:
        print(json.dumps({
            "original_spectral_radius": round(rho_orig, 4),
            "mitigated_spectral_radius": round(rho_mit, 4),
            "reduction_percent": round(reduction, 1),
            "original_regime": cm.get_regime().value,
            "mitigated_regime": mitigated.get_regime().value,
        }, indent=2))
    else:
        print(f"Original:  ρ = {rho_orig:.4f} ({cm.get_regime().value})")
        print(f"Mitigated: ρ = {rho_mit:.4f} ({mitigated.get_regime().value})")
        print(f"Reduction: {reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems",
        epilog="Based on arXiv:2606.20493 (Jun 2026)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # simulate
    sim_parser = subparsers.add_parser("simulate", help="Run contagion simulation")
    sim_parser.add_argument("--agents", "-n", type=int, default=5, help="Number of agents")
    sim_parser.add_argument("--hops", type=int, default=10, help="Number of hops")
    sim_parser.add_argument("--topology", default="fully_connected",
                            choices=["fully_connected", "chain", "ring", "star"],
                            help="Network topology")
    sim_parser.add_argument("--coefficient", "-c", type=float, default=0.25,
                            help="Base contagion coefficient")
    sim_parser.add_argument("--output", "-o", help="Save results to JSON file")
    sim_parser.add_argument("--json", action="store_true", help="Output JSON")
    sim_parser.set_defaults(func=cmd_simulate)

    # analyze
    ana_parser = subparsers.add_parser("analyze", help="Analyze contagion matrix")
    ana_parser.add_argument("--matrix", "-m", required=True, help="Matrix JSON file")
    ana_parser.add_argument("--json", action="store_true", help="Output JSON")
    ana_parser.set_defaults(func=cmd_analyze)

    # mitigate
    mit_parser = subparsers.add_parser("mitigate", help="Apply mitigation strategies")
    mit_parser.add_argument("--matrix", "-m", required=True, help="Matrix JSON file")
    mit_parser.add_argument("--strategy", "-s", default="committee",
                            choices=["committee", "topology"],
                            help="Mitigation strategy")
    mit_parser.add_argument("--k", type=int, default=3, help="Committee size")
    mit_parser.add_argument("--coefficient", "-c", type=float, default=0.25,
                            help="Base coefficient for topology strategy")
    mit_parser.add_argument("--json", action="store_true", help="Output JSON")
    mit_parser.set_defaults(func=cmd_mitigate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
