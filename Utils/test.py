# ──────────────────────────────────────────────────────────
# Author: Joshua Chua Han Wei – 32781555
# File: test.py
# Purpose: Quick sanity script – three sample match-ups using MarkovGame.
# ──────────────────────────────────────────────────────────

import numpy as np
from Game.game import MarkovGame, MonteCarloGame
from Strategies.m0strategies import AlwaysDefect, RandomStrategy
from Strategies.m1strategies import TitForTat
from Strategies.m3strategies import TitForThreeTats

# Instantiate strategies
s1 = TitForTat()
s2 = AlwaysDefect()

s3 = RandomStrategy()

# Setup game configs
rounds = 50
trials = 10000
error = 0.0

# TitForTat vs Random
markov_game = MarkovGame(s1, s3, rounds=rounds, error=error)
score1, score3, _ = markov_game.run()
print(f"TitForTat vs RandomStrategy → TFT: {score1:.2f}, Random: {score3:.2f}")

# AD vs Random
markov_game = MarkovGame(s2, s3, rounds=rounds, error=error)
score2, score3, _ = markov_game.run()
print(f"AlwaysDefect vs RandomStrategy → TFT: {score2:.2f}, Random: {score3:.2f}")

# TitForTat vs AlwaysDefect
markov_game = MarkovGame(s1, s2, rounds=rounds, error=error)
score1, score2, _ = markov_game.run()
print(f"TitForTat vs AlwaysDefect → TFT: {score1:.2f}, AD: {score2:.2f}")

markov_game = MarkovGame(s2,s1, rounds=rounds, error=error)
score2, score1, _ = markov_game.run()
print(f"AlwaysDefect vs TitForTat → TFT: {score2:.2f}, AD: {score1:.2f}")

# TitForTat vs Random again
markov_game = MarkovGame(s1, s3, rounds=rounds, error=error)
score1, score3, _ = markov_game.run()
print(f"TitForTat vs RandomStrategy → TFT: {score1:.2f}, Random: {score3:.2f}")




# tournament_test.py

import itertools
import numpy as np
import pandas as pd

from Game.game import MarkovGame, MonteCarloGame
from Strategies.m0strategies import AlwaysDefect, RandomStrategy
from Strategies.m1strategies import TitForTat

def run_tournament(competitors, engine_type="markov", rounds=50, trials=10000, error=0.0):
    names = [s.name for s in competitors]
    N = len(competitors)
    payoff_matrix = np.full((N, N), np.nan, dtype=float)

    for i, j in itertools.combinations(range(N), 2):
        strat_i = competitors[i]
        strat_j = competitors[j]

        strat_i.reset()
        strat_j.reset()

        if engine_type.lower() == "markov":
            game = MarkovGame(strat_i, strat_j, rounds=rounds, error=error)
            score_i, score_j, _ = game.run()
        else:
            game = MonteCarloGame(strat_i, strat_j, rounds=rounds, trials=trials, error=error)
            score_i, score_j = game.run()
        
        print(strat_i.name)
        print(strat_j.name)
        print(score_i)
        print(score_j)
        print('\n')

        payoff_matrix[i, j] = score_i
        payoff_matrix[j, i] = score_j

    return pd.DataFrame(payoff_matrix, index=names, columns=names)


if __name__ == "__main__":
    competitors = [
        AlwaysDefect(),
        RandomStrategy(coop_prob=0.5),
        TitForTat(),
    ]

    df_markov = run_tournament(competitors, engine_type="markov", rounds=50, error=0.0)
    # print("\n--- Markov Tournament (error=0.00) ---")
    # print(df_markov.round(2))

    df_mc = run_tournament(competitors, engine_type="montecarlo", rounds=50, trials=5000, error=0.0)
    # print("\n--- Monte Carlo Tournament (error=0.00) ---")
    # print(df_mc.round(2))

