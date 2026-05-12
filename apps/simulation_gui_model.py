"""Pure data helpers for the Tkinter simulation GUI."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from Experiments.strategy_registry import available_strategy_names, create_strategy
from Simulation.trace import RoundTrace, simulate_match_trace


@dataclass(frozen=True)
class ScorePoint:
    """A cumulative score point suitable for plotting."""

    round_index: int
    strategy_a_score: float
    strategy_b_score: float


@dataclass(frozen=True)
class TraceTableRow:
    """A table-ready projection of a simulation trace row."""

    round_index: int
    memory: str
    strategy_a_move: str
    strategy_b_move: str
    outcome: str
    payoff_a: float
    payoff_b: float
    cumulative_score_a: float
    cumulative_score_b: float


@dataclass(frozen=True)
class TraceSummary:
    """Headless simulation output for GUI controls to render."""

    strategy_a: str
    strategy_b: str
    rounds: int
    error: float
    initial_state: str
    seed: int | None
    timeline: tuple[str, ...]
    final_score_a: float
    final_score_b: float
    trace: tuple[RoundTrace, ...]
    score_points: tuple[ScorePoint, ...]
    table_rows: tuple[TraceTableRow, ...]


def list_strategy_names() -> tuple[str, ...]:
    """Return names that can be shown in strategy selection controls."""
    return tuple(available_strategy_names())


def build_trace_summary(
    strategy_a: str,
    strategy_b: str,
    *,
    rounds: int = 50,
    error: float = 0.0,
    initial_state: str = "CC",
    seed: int | None = None,
) -> TraceSummary:
    """Run a named pairwise simulation and return GUI-friendly pure data."""
    player_a = _create_strategy(strategy_a, "strategy_a")
    player_b = _create_strategy(strategy_b, "strategy_b")
    trace = tuple(
        simulate_match_trace(
            player_a,
            player_b,
            rounds=rounds,
            error=error,
            initial_state=initial_state,
            seed=seed,
        )
    )
    final_score_a = trace[-1].cumulative_score_1 if trace else 0.0
    final_score_b = trace[-1].cumulative_score_2 if trace else 0.0

    return TraceSummary(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        rounds=rounds,
        error=error,
        initial_state=initial_state,
        seed=seed,
        timeline=tuple(row.outcome for row in trace),
        final_score_a=final_score_a,
        final_score_b=final_score_b,
        trace=trace,
        score_points=score_series(trace),
        table_rows=table_rows(trace),
    )


def score_series(trace: Iterable[RoundTrace]) -> tuple[ScorePoint, ...]:
    """Return cumulative scores with an initial zero point for charting."""
    return (
        ScorePoint(0, 0.0, 0.0),
        *(
            ScorePoint(
                row.round_index,
                row.cumulative_score_1,
                row.cumulative_score_2,
            )
            for row in trace
        ),
    )


def table_rows(trace: Iterable[RoundTrace]) -> tuple[TraceTableRow, ...]:
    """Project trace records into stable table rows without GUI imports."""
    return tuple(
        TraceTableRow(
            round_index=row.round_index,
            memory=" ".join(row.state_before),
            strategy_a_move=_display_move(
                row.intended_move_1, row.actual_move_1, row.error_flip_1
            ),
            strategy_b_move=_display_move(
                row.intended_move_2, row.actual_move_2, row.error_flip_2
            ),
            outcome=row.outcome,
            payoff_a=row.payoff_1,
            payoff_b=row.payoff_2,
            cumulative_score_a=row.cumulative_score_1,
            cumulative_score_b=row.cumulative_score_2,
        )
        for row in trace
    )


def _create_strategy(name: str, field_name: str):
    try:
        return create_strategy(name)
    except ValueError as exc:
        valid = ", ".join(list_strategy_names())
        raise ValueError(
            f"Invalid {field_name} strategy name {name!r}. "
            f"Available strategies: {valid}"
        ) from exc


def _display_move(intended: str, actual: str, flipped: bool) -> str:
    if flipped:
        return f"{intended}->{actual}!"
    return actual
