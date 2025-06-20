# ──────────────────────────────────────────────────────────
# Author: Joshua Chua Han Wei – 32781555
# File: tournament.py
# Purpose: Full round-robin tournament driver with optional heat-maps / rankings.
# ──────────────────────────────────────────────────────────
from Utils.save_figure import save_fig
from Utils.random_seed import set_seed
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your game engines
from Game.game import MarkovGame, MonteCarloGame

# Import all hand‐coded strategies
from Strategies.m0strategies import AlwaysCooperate, AlwaysDefect, RandomStrategy
from Strategies.m1strategies import TitForTat, WinStayLoseShift, ReverseTitForTat, GrimTrigger
from Strategies.m2strategies import (
    TitForTwoTats,
    #ClearGrudger,
    Pavlov2,
    GenerousTwoTitForTwo,
    SuspiciousTf2T,
    # Prober,
    # Grim2,
    # Vindictive2
)
from Strategies.m3strategies import (
    #TitForThreeTats,
    TwoForgiveOnePunish,
    ThreeGrudger,
    PatternFollower3,
    Pavlov3,
    Generous3,
    UnforgivingPatternHunter
)

# Import Chromosome‐based strategy
from Strategies.chromosomes import ChromosomeStrategy

# ----------------------------------------------------------------------
# 1) DEFINE COMPETITOR LIST 
# ----------------------------------------------------------------------
competitors = [
    # Memory‐0 strategies
    # AlwaysCooperate(),
    AlwaysDefect(),
    RandomStrategy(coop_prob=0.5),

    # Memory‐1 strategies
    TitForTat(),
    WinStayLoseShift(),
    ReverseTitForTat(),
    # GrimTrigger(),

    # Memory‐2 strategies
    TitForTwoTats(),
    #ClearGrudger(),
    Pavlov2(),
    GenerousTwoTitForTwo(),
    SuspiciousTf2T(),
    # Prober(),
    # Grim2(),
    # Vindictive2(),

    # Memory‐3 strategies
    #TitForThreeTats(),
    TwoForgiveOnePunish(),
    ThreeGrudger(),
    PatternFollower3(),
    Pavlov3(),
    Generous3(),
    UnforgivingPatternHunter(),

    # Example Chromosome‐based strategies:
    # ChromosomeStrategy("01"),       # m=1, equivalent to AlwaysDefect vs. AlwaysCooperate mapping
    # ChromosomeStrategy("0101"),     # m=1, exactly TitForTat
]

# ----------------------------------------------------------------------
# 2) UTILITY: extract "name" from each strategy for labeling
# ----------------------------------------------------------------------
set_seed()
strategy_names = [s.name for s in competitors]
N = len(competitors)
# ----------------------------------------------------------------------
# 3) MAIN TOURNAMENT FUNCTION
# ----------------------------------------------------------------------
def run_tournament(
    competitors,
    engine_type="markov",
    rounds=50,
    trials=10000,
    error=0.0,
):
    """
    Run a round‐robin tournament over the given list of strategy instances.
    Skip self‐matches and avoid redundant matches by only iterating i < j.
    Fill in a symmetric payoff matrix: payoff[i,j] = payoff_i_vs_j, payoff[j,i] = payoff_j_vs_i.
    """
    names = [s.name for s in competitors]
    N = len(competitors)

    # Initialize two N×N numpy arrays of floats; fill diagonals with np.nan (no self‐play)
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
# 4)  USAGE 
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ROUNDS = 50
    TRIALS = 10_000
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
    # 5) Compare Markov vs Monte Carlo for memory‐3 strategies
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
    # 6) PLOTTING HEATMAPS
    # ------------------------------------------------------------------
    def plot_heatmap(df, title):
        plt.figure(figsize=(8, 6))
        plt.imshow(df.values, interpolation="nearest", cmap="viridis")
        plt.colorbar(label="Cumulative payoff")
        plt.xticks(range(len(df)), df.columns, rotation=90)
        plt.yticks(range(len(df)), df.index)
        plt.title(title)
        plt.tight_layout()
        save_fig(f"{title.replace(' ', '_')}.png", dpi=300, show=True)

    plot_heatmap(df_markov_noerr, "Markov Payoffs (error=0.00)")
    plot_heatmap(df_mc_noerr, "Monte Carlo Payoffs (error=0.00)")

    # ------------------------------------------------------------------
    # 7) RANKING AND NORMALIZING PAYOFFS
    # ------------------------------------------------------------------
    ranking_markov = df_markov_noerr.sum(axis=1).sort_values(ascending=False)
    print("\n--- Total Markov Scores (error=0.00), Ranked ---")
    print(ranking_markov.round(1))

    ranking_mc = df_mc_noerr.sum(axis=1).sort_values(ascending=False)
    print("\n--- Total Monte Carlo Scores (error=0.00), Ranked ---")
    print(ranking_mc.round(1))

    # If you want “average payoff per round” instead of cumulative, divide by ROUNDS:
    avg_per_round_markov = ranking_markov / ROUNDS
    avg_per_round_mc = ranking_mc / ROUNDS

    print("\n--- Average per Round (Markov) ---")
    print(avg_per_round_markov.round(3))
    print("\n--- Average per Round (Monte Carlo) ---")
    print(avg_per_round_mc.round(3))


    # ==============================================================================
    # 8) HORIZONTAL  BAR CHART OF TOTAL SCORES (COLOR‐CODED BY niceness)
    # ==============================================================================

    # 8a) Calculate total‐score Series (sum over rows of df_markov_noerr)
    total_scores = df_markov_noerr.sum(axis=1)

    # 8b) Build quick lookups for each strategy’s is_nice and memory_size
    name_to_nice = {s.name: s.is_nice for s in competitors}
    name_to_memory = {s.name: s.memory_size for s in competitors}

    # 8c) Sort strategies by total score (ascending=True so barh plots bottom up)
    sorted_scores = total_scores.sort_values(ascending=True)

    # 8d) Assign red/green color based on niceness
    colors = ["green" if name_to_nice[name] else "red" for name in sorted_scores.index]

    # 8e) Plot horizontal bar chart
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_scores.index, sorted_scores.values, color=colors)
    plt.xlabel("Total Cumulative Score")
    plt.title("Total Tournament Scores (Markov, no error)\n(Green = Nice – Red = Not Nice)")
    plt.tight_layout()
    save_fig("Total_tournament_scores.png", dpi=300, show=True)


    # ==============================================================================
    # 9) BOXPLOT OF “AVERAGE SCORE PER ROUND” GROUPED BY memory_size
    # ==============================================================================

    # 9a) Create a small DataFrame with one row per strategy
    stats_df = pd.DataFrame({
        "strategy": total_scores.index,
        "total_score": total_scores.values,
        "avg_per_round": total_scores.values / ROUNDS,
        "memory_size": [name_to_memory[name] for name in total_scores.index],
        "is_nice": [name_to_nice[name] for name in total_scores.index],
    })

    # 9b) Boxplot: average‐per‐round distributions by memory size
    plt.figure(figsize=(8, 6))
    stats_df.boxplot(column="avg_per_round", by="memory_size")
    plt.xlabel("Memory Size (m)")
    plt.ylabel("Average Score per Round")
    plt.title("Distribution of Average Score per Round by Memory Size")
    # Remove the automatic "Boxplot grouped by memory_size" supertitle
    plt.suptitle("")
    plt.tight_layout()
    save_fig("memory_size_distribution", dpi=300, show=True)


    # ------------------------------------------------------------------
    # 10) FURTHER ANALYSIS
    # ------------------------------------------------------------------
    # For instance, you could also compare “nice” vs “not nice” average distributions:
    #
    plt.figure(figsize=(6, 4))
    stats_df.boxplot(column="avg_per_round", by="is_nice")
    plt.xlabel("Is Nice?")
    plt.ylabel("Average Score per Round")
    plt.title("Nice vs. Non‐nice Strategy Performance")
    plt.suptitle("")
    plt.tight_layout()
    save_fig("nice_nasty_distribution", dpi=300, show=True)


