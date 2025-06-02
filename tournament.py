# tournament_driver.py

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your game engines
from Game.game import MarkovGame, MonteCarloGame

# Import all hand-coded strategies
from Strategies.m0strategies import AlwaysCooperate, AlwaysDefect, RandomStrategy
from Strategies.m1strategies import TitForTat, WinStayLoseShift, ReverseTitForTat, GrimTrigger
from Strategies.m2strategies import (
    TitForTwoTats,
    ClearGrudger,
    Pavlov2,
    GenerousTwoTitForTwo,
    SuspiciousTf2T,
    Prober,
    Grim2,
    Vindictive2
)
from Strategies.m3strategies import (
    TitForThreeTats,
    TwoForgiveOnePunish,
    ThreeGrudger,
    PatternFollower3,
    Pavlov3,
    Generous3,
    UnforgivingPatternHunter
)

# Import Chromosome-based strategy (example)
from Strategies.chromosomes import ChromosomeStrategy

# ----------------------------------------------------------------------
# 1) DEFINE YOUR COMPETITOR LIST HERE
#
#    Simply instantiate one object of each strategy class you want to include.
#    You can add as many ChromosomeStrategy(...) instances as you like—just
#    make sure the bitstring length is 4**m for m = {0,1,2,3}.
# ----------------------------------------------------------------------
competitors = [
    # Memory-0 strategies
    #AlwaysCooperate(),
    AlwaysDefect(),
    RandomStrategy(coop_prob=0.5),

    # Memory-1 strategies
    TitForTat(),
    WinStayLoseShift(),
    ReverseTitForTat(),
    #GrimTrigger(),

    # Memory-2 strategies
    TitForTwoTats(),
    ClearGrudger(),
    Pavlov2(),
    GenerousTwoTitForTwo(),
    SuspiciousTf2T(),
    #Prober(),
    #Grim2(),
    #Vindictive2(),

    # Memory-3 strategies
    TitForThreeTats(),
    TwoForgiveOnePunish(),
    ThreeGrudger(),
    PatternFollower3(),
    Pavlov3(),
    Generous3(),
    UnforgivingPatternHunter(),

    # Example Chromosome-based strategies:
    #   - "0101" is exactly a memory-1 TitForTat
    #   - We could build any 4^m bitstring for m=0..3. Below are two examples:
    #ChromosomeStrategy("01"),       # m=1, equivalent to AlwaysDefect vs. AlwaysCooperate mapping
    #ChromosomeStrategy("0101"),     # m=1, exactly TitForTat
    # (if we had a 16-bit string we would get a memory-2 chromosome, etc.)
]

# ----------------------------------------------------------------------
# 2) UTILITY: extract "name" from each strategy for labeling
# ----------------------------------------------------------------------
strategy_names = [s.name for s in competitors]
N = len(competitors)

# ----------------------------------------------------------------------
# 3) MAIN TOURNAMENT FUNCTION
#
#    - engine_type: either "markov" or "montecarlo"
#    - rounds: number of rounds per pairing (e.g. 50)
#    - trials: only used if engine_type == "montecarlo" (e.g. 10_000)
#    - error: trembling-hand error probability (0.0 means no error; you can try 0.01, etc.)
#
#    Returns a DataFrame `df_payoffs` such that df_payoffs.loc[s_i, s_j]
#    = expected payoff of strategy i when playing strategy j.
# ----------------------------------------------------------------------
def run_tournament(
    competitors,
    engine_type="markov",
    rounds=50,
    trials=10000,
    error=0.0,
):
    """
    Run a round-robin tournament over the given list of strategy instances.
    Skip self-matches and avoid redundant matches by only iterating i < j.
    Fill in a symmetric payoff matrix: payoff[i,j] = payoff_i_vs_j, payoff[j,i] = payoff_j_vs_i.
    """

    names = [s.name for s in competitors]
    N = len(competitors)

    # Initialize two N×N numpy arrays of floats; fill diagonals with np.nan (no self-play)
    payoff_matrix = np.full((N, N), np.nan, dtype=float)

    # Loop over all unordered pairs (i < j)
    for i, j in itertools.combinations(range(N), 2):
        strat_i = competitors[i]
        strat_j = competitors[j]

        # Ensure clean state
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

        # Place scores into the payoff_matrix; i vs j → score_i and j vs i → score_j
        payoff_matrix[i, j] = score_i
        payoff_matrix[j, i] = score_j

    # Build a pandas DataFrame for convenience
    df = pd.DataFrame(payoff_matrix, index=names, columns=names)
    return df


# ----------------------------------------------------------------------
# 4) EXAMPLE USAGE
#
#    Run four different tournaments:
#      A) Markov, error=0.0
#      B) Markov, error=0.05
#      C) Monte Carlo, error=0.0
#      D) Monte Carlo, error=0.05
#
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ROUNDS = 50
    TRIALS = 10000
    ERR_0 = 0.0
    ERR_1 = 0.05

    # A) Markov, no error
    df_markov_noerr = run_tournament(
        competitors, engine_type="markov", rounds=ROUNDS, error=ERR_0
    )
    print("\n--- Markov Tournament (error=0.00) ---")
    print(df_markov_noerr.round(2))

    # B) Markov, with error
    df_markov_err = run_tournament(
        competitors, engine_type="markov", rounds=ROUNDS, error=ERR_1
    )
    print("\n--- Markov Tournament (error=0.05) ---")
    print(df_markov_err.round(2))

    # C) Monte Carlo, no error
    df_mc_noerr = run_tournament(
        competitors,
        engine_type="montecarlo",
        rounds=ROUNDS,
        trials=TRIALS,
        error=ERR_0,
    )
    print("\n--- Monte Carlo Tournament (error=0.00) ---")
    print(df_mc_noerr.round(2))

    # D) Monte Carlo, with error
    df_mc_err = run_tournament(
        competitors,
        engine_type="montecarlo",
        rounds=ROUNDS,
        trials=TRIALS,
        error=ERR_1,
    )
    print("\n--- Monte Carlo Tournament (error=0.05) ---")
    print(df_mc_err.round(2))

    # ------------------------------------------------------------------
    # 5) OPTIONAL: Compare Markov vs Monte Carlo for memory-3 strategies
    #
    #    Extract only those rows/columns corresponding to memory_size == 3.
    #    Then compute difference: (Markov_noerr - MC_noerr).
    # ------------------------------------------------------------------
    mem3_names = [
        s.name for s in competitors if getattr(s, "memory_size", 0) == 3
    ]
    df_mk3 = df_markov_noerr.loc[mem3_names, mem3_names]
    df_mc3 = df_mc_noerr.loc[mem3_names, mem3_names]
    diff_mem3 = df_mk3 - df_mc3

    print("\n--- Difference (Markov_noerr − MonteCarlo_noerr) for Memory-3 Strategies ---")
    print(diff_mem3.round(2))

    # ------------------------------------------------------------------
    # 6) OPTIONAL: PLOTTING HEATMAPS
    #
    #    We’ll show a basic heatmap of the Markov payoff matrix (error=0.00).
    #    You can repeat for any DataFrame printed above.
    # ------------------------------------------------------------------
    def plot_heatmap(df, title):
        plt.figure(figsize=(8, 6))
        plt.imshow(df.values, interpolation="nearest", cmap="viridis")
        plt.colorbar(label="Cumulative payoff")
        plt.xticks(range(len(df)), df.columns, rotation=90)
        plt.yticks(range(len(df)), df.index)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    plot_heatmap(df_markov_noerr, "Markov Payoffs (error=0.00)")
    plot_heatmap(df_mc_noerr, "Monte Carlo Payoffs (error=0.00)")

    # ------------------------------------------------------------------
    # 7) OPTIONAL: RANKING AND NORMALIZING PAYOFFS
    #
    #    You might want to sum each row to get a “total tournament payoff” and
    #    rank strategies by how well they did on average.
    # ------------------------------------------------------------------
    ranking_markov = df_markov_noerr.sum(axis=1).sort_values(ascending=False)
    print("\n--- Total Markov Scores (error=0.00), Ranked ---")
    print(ranking_markov.round(1))

    ranking_mc = df_mc_noerr.sum(axis=1).sort_values(ascending=False)
    print("\n--- Total Monte Carlo Scores (error=0.00), Ranked ---")
    print(ranking_mc.round(1))

    # If you want “average payoff per round” instead of cumulative, simply divide by ROUNDS:
    avg_per_round_markov = ranking_markov / ROUNDS
    avg_per_round_mc = ranking_mc / ROUNDS

    print("\n--- Average per Round (Markov) ---")
    print(avg_per_round_markov.round(3))
    print("\n--- Average per Round (Monte Carlo) ---")
    print(avg_per_round_mc.round(3))
