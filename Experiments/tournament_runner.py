"""Shared round-robin tournament execution utilities."""

import itertools

import numpy as np
import pandas as pd

from Game.game import MarkovGame, MonteCarloGame


def run_tournament(
    competitors,
    engine_type="markov",
    rounds=50,
    trials=10000,
    error=0.0,
):
    """
    Run a pairwise round-robin tournament and return a payoff matrix.

    The diagonal is left as NaN because self-play is not evaluated. Each
    off-diagonal cell contains the cumulative score earned by the row strategy
    against the column strategy.
    """
    names = [s.name for s in competitors]
    n_competitors = len(competitors)
    payoff_matrix = np.full((n_competitors, n_competitors), np.nan, dtype=float)

    for i, j in itertools.combinations(range(n_competitors), 2):
        strat_i = competitors[i]
        strat_j = competitors[j]

        strat_i.reset()
        strat_j.reset()

        if engine_type.lower() == "markov":
            game = MarkovGame(strat_i, strat_j, rounds=rounds, error=error)
            score_i, score_j, _ = game.run()
        elif engine_type.lower() == "montecarlo":
            game = MonteCarloGame(
                strat_i, strat_j, rounds=rounds, trials=trials, error=error
            )
            score_i, score_j = game.run()
        else:
            raise ValueError(f"Unknown engine_type: {engine_type!r}")

        payoff_matrix[i, j] = score_i
        payoff_matrix[j, i] = score_j

    return pd.DataFrame(payoff_matrix, index=names, columns=names)


def total_payoffs_per_round(payoff_matrix: pd.DataFrame, rounds: int) -> pd.Series:
    """Return each strategy's aggregate payoff normalized by rounds per match."""
    return payoff_matrix.sum(axis=1) / rounds
