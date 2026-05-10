"""Terminal visualization for pairwise Iterated Prisoner's Dilemma simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.strategy_registry import available_strategy_names, create_strategy
from Simulation.trace import RoundTrace, simulate_match_trace


def format_state(state: tuple[str, ...]) -> str:
    return " ".join(state)


def format_move(intended: str, actual: str, flipped: bool) -> str:
    if flipped:
        return f"{intended}->{actual}!"
    return actual


def format_trace_table(strategy_a: str, strategy_b: str, trace: list[RoundTrace]) -> str:
    lines = [
        f"Dilemma Dynamics trace: {strategy_a} vs {strategy_b}",
        "Legend: C=cooperate, D=defect, !=trembling-hand error flip",
        "",
        (
            "Rnd | Memory      | A move | B move | Out | Payoff | "
            "Cumulative"
        ),
        "----+-------------+--------+--------+-----+--------+-----------",
    ]

    for row in trace:
        memory = format_state(row.state_before)
        move_a = format_move(row.intended_move_1, row.actual_move_1, row.error_flip_1)
        move_b = format_move(row.intended_move_2, row.actual_move_2, row.error_flip_2)
        lines.append(
            f"{row.round_index:>3} | "
            f"{memory:<11} | "
            f"{move_a:<6} | "
            f"{move_b:<6} | "
            f"{row.outcome:<3} | "
            f"{row.payoff_1:>2.0f}-{row.payoff_2:<2.0f}  | "
            f"{row.cumulative_score_1:>4.0f}-{row.cumulative_score_2:<4.0f}"
        )

    if trace:
        final = trace[-1]
        lines.extend(
            [
                "",
                f"Timeline: {' '.join(row.outcome for row in trace)}",
                (
                    "Final score: "
                    f"{strategy_a} {final.cumulative_score_1:.0f}, "
                    f"{strategy_b} {final.cumulative_score_2:.0f}"
                ),
            ]
        )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    strategies = available_strategy_names()
    parser = argparse.ArgumentParser(
        description="Render a round-by-round terminal trace for a pairwise IPD matchup."
    )
    parser.add_argument("--list-strategies", action="store_true", help="Print strategy names and exit.")
    parser.add_argument("--strategy-a", default="TitForTat", choices=strategies)
    parser.add_argument("--strategy-b", default="AlwaysDefect", choices=strategies)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--error", type=float, default=0.0)
    parser.add_argument("--initial-state", default="CC", choices=["CC", "CD", "DC", "DD"])
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_strategies:
        print("\n".join(available_strategy_names()))
        return 0

    strategy_a = create_strategy(args.strategy_a)
    strategy_b = create_strategy(args.strategy_b)
    trace = simulate_match_trace(
        strategy_a,
        strategy_b,
        rounds=args.rounds,
        error=args.error,
        initial_state=args.initial_state,
        seed=args.seed,
    )
    print(format_trace_table(args.strategy_a, args.strategy_b, trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
