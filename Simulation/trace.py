"""Round-by-round simulation traces for pairwise strategy matchups."""

from dataclasses import asdict, dataclass
import random

import pandas as pd

from Utils.gamestates import state_to_last_moves, state_to_last_moves_reversed
from Utils.payoff_matrix import payoff_matrix
from Utils.random_seed import set_seed


@dataclass(frozen=True)
class RoundTrace:
    round_index: int
    state_before: tuple[str, ...]
    intended_move_1: str
    intended_move_2: str
    actual_move_1: str
    actual_move_2: str
    error_flip_1: bool
    error_flip_2: bool
    outcome: str
    payoff_1: float
    payoff_2: float
    cumulative_score_1: float
    cumulative_score_2: float
    state_after: tuple[str, ...]


def flip_move(move: str) -> str:
    """Return the opposite Prisoner's Dilemma move."""
    if move == "C":
        return "D"
    if move == "D":
        return "C"
    raise ValueError(f"Unsupported move: {move!r}")


def validate_trace_config(rounds: int, error: float, initial_state: str) -> None:
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if not 0.0 <= error <= 1.0:
        raise ValueError("error must be between 0.0 and 1.0")
    if initial_state not in payoff_matrix:
        raise ValueError(
            f"initial_state must be one of {tuple(payoff_matrix)}, got {initial_state!r}"
        )


def simulate_match_trace(
    strat1,
    strat2,
    rounds: int = 50,
    error: float = 0.0,
    initial_state: str = "CC",
    seed: int | None = None,
) -> list[RoundTrace]:
    """
    Simulate a single pairwise matchup and return per-round trace records.

    This is intentionally separate from `MonteCarloGame`: batch experiments can
    keep using aggregate scores, while visualization tools can inspect each
    decision, error flip, payoff, and memory-window update.
    """
    validate_trace_config(rounds, error, initial_state)

    if seed is not None:
        set_seed(seed)

    strat1.reset()
    strat2.reset()

    max_memory = max(1, strat1.memory_size, strat2.memory_size)
    state = tuple([initial_state] * max_memory)
    cumulative_score_1 = 0.0
    cumulative_score_2 = 0.0
    trace = []

    for round_index in range(1, rounds + 1):
        state_before = state
        intended_move_1 = strat1.next_move(state_before, state_to_last_moves)
        intended_move_2 = strat2.next_move(state_before, state_to_last_moves_reversed)

        error_flip_1 = random.random() < error
        error_flip_2 = random.random() < error
        actual_move_1 = flip_move(intended_move_1) if error_flip_1 else intended_move_1
        actual_move_2 = flip_move(intended_move_2) if error_flip_2 else intended_move_2

        outcome = actual_move_1 + actual_move_2
        payoff_1, payoff_2 = payoff_matrix[outcome]
        cumulative_score_1 += payoff_1
        cumulative_score_2 += payoff_2

        if max_memory > 1:
            state_after = (*state_before[1:], outcome)
        else:
            state_after = (outcome,)

        trace.append(
            RoundTrace(
                round_index=round_index,
                state_before=state_before,
                intended_move_1=intended_move_1,
                intended_move_2=intended_move_2,
                actual_move_1=actual_move_1,
                actual_move_2=actual_move_2,
                error_flip_1=error_flip_1,
                error_flip_2=error_flip_2,
                outcome=outcome,
                payoff_1=payoff_1,
                payoff_2=payoff_2,
                cumulative_score_1=cumulative_score_1,
                cumulative_score_2=cumulative_score_2,
                state_after=state_after,
            )
        )
        state = state_after

    return trace


def trace_to_dataframe(trace: list[RoundTrace]) -> pd.DataFrame:
    """Convert trace records to a Pandas DataFrame for analysis or plotting."""
    return pd.DataFrame(asdict(row) for row in trace)
