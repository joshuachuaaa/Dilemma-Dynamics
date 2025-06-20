# ──────────────────────────────────────────────────────────
# Author: Joshua Chua Han Wei – 32781555
# File: tournament_test.py
# Purpose: Minimal test harness to compare Markov vs Monte-Carlo pay-offs.
# ──────────────────────────────────────────────────────────

# ------------------------------------------------------------------- imports
from Utils.save_figure import save_fig
from Utils.random_seed import set_seed
import itertools, time
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from Game.game import MonteCarloGame


# --------------------------- strategy imports (unchanged) -------------------
from Strategies.m0strategies import AlwaysDefect, RandomStrategy
from Strategies.m1strategies import TitForTat, WinStayLoseShift, ReverseTitForTat
from Strategies.m2strategies import (
    TitForTwoTats, Pavlov2, GenerousTwoTitForTwo, SuspiciousTf2T)
from Strategies.m3strategies import (
    TwoForgiveOnePunish, ThreeGrudger, PatternFollower3,
    Pavlov3, Generous3, UnforgivingPatternHunter)

# ---------------------------------------------------------------- constants
set_seed()
ROUNDS       = 50
TRIALS       = 10_000
ERROR_LEVELS = [0.00, 0.05, 0.10]
MAKE_BARCHART = False          # set True if want per-ε bar charts

# ----------------------------------------------------------- competitor list
competitors = [
    AlwaysDefect(),
    RandomStrategy(0.5),
    TitForTat(), WinStayLoseShift(), ReverseTitForTat(),
    TitForTwoTats(), Pavlov2(), GenerousTwoTitForTwo(), SuspiciousTf2T(),
    TwoForgiveOnePunish(), ThreeGrudger(), PatternFollower3(),
    Pavlov3(), Generous3(), UnforgivingPatternHunter()
]

name_to_nice   = {s.name: s.is_nice     for s in competitors}
name_to_memory = {s.name: s.memory_size for s in competitors}
strategy_names = [s.name for s in competitors]
N = len(competitors)

# ------------------------------------------------------------- helpers
def timestamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def run_tournament(err: float) -> pd.Series:
    """Round-robin Monte-Carlo; returns per-strategy avg pay-off / round."""
    pay = np.zeros(N)
    for i, j in itertools.combinations(range(N), 2):
        s_i, s_j = competitors[i], competitors[j]
        s_i.reset(); s_j.reset()
        game = MonteCarloGame(s_i, s_j, ROUNDS, err, trials=TRIALS)
        sc_i, sc_j = game.run()
        pay[i] += sc_i / ROUNDS
        pay[j] += sc_j / ROUNDS
    return pd.Series(pay, index=strategy_names)

def class_gap(series: pd.Series) -> tuple[float,float,float]:
    nice  = series[[k for k in series.index if name_to_nice[k]]].mean()
    nasty = series[[k for k in series.index if not name_to_nice[k]]].mean()
    return nice, nasty, nasty - nice

# ------------------------------------------------------------- main sweep
if __name__ == "__main__":
    timestamp("Starting Monte-Carlo sweep")
    payoff_sweep = pd.DataFrame(index=strategy_names)
    gap_records  = []

    for ε in ERROR_LEVELS:
        label = f"{int(ε*100)}%"
        timestamp(f"Simulating ε = {ε:.0%}")
        series = run_tournament(ε)
        payoff_sweep[label] = series

        nice_m, nasty_m, gap = class_gap(series)
        gap_records.append({"ε": ε, "nice": nice_m,
                            "nasty": nasty_m, "gap": gap})
        timestamp(f"  → nice = {nice_m:.2f}, nasty = {nasty_m:.2f}, gap = {gap:+.2f}")

    gap_df = pd.DataFrame(gap_records).set_index("ε")
    print("\n=== Class gap summary ===")
    print(gap_df.round(3))
    # ----------------------------------------------------------------- Fig A
    plt.figure(figsize=(9, 6))
    for strat in payoff_sweep.index:
        plt.plot(
            payoff_sweep.columns,
            payoff_sweep.loc[strat],
            marker='o', lw=1, alpha=0.8,
            label=strat,                       # ← NEW: give each line a label
        )

    plt.xlabel("trembling-hand error ε")
    plt.ylabel("avg pay-off / round")
    plt.title("A) Strategy performance vs noise")
    plt.grid(ls='--', lw=0.4)

    # --- NEW legend ----------------------------------------------------------
    plt.legend(
        bbox_to_anchor=(1.03, 1),             # push legend outside the plot
        loc="upper left",
        fontsize="small",
        frameon=False
    )
    # -------------------------------------------------------------------------
    plt.tight_layout()
    save_fig("A_performance_vs_noise.png", dpi=300, show=True)


    # ----------------------------------------------------------------- Fig B
    fig, axes = plt.subplots(1, len(ERROR_LEVELS), figsize=(12, 4), sharey=True)
    for ax, ε in zip(axes, ERROR_LEVELS):
        col = f"{int(ε*100)}%"
        tmp = payoff_sweep[col].to_frame('score')
        tmp['memory'] = tmp.index.map(name_to_memory)
        tmp.boxplot(column='score', by='memory', ax=ax)
        ax.set_title(f"ε = {ε:.0%}")
        ax.set_xlabel("memory (m)")
        if ax is axes[0]:
            ax.set_ylabel("avg pay-off / round")
    fig.suptitle("B) Distribution by memory size"); fig.tight_layout()
    save_fig("B_boxplot_memory.png", dpi=300, show=True)

    # ----------------------------------------------------------------- Fig C  (nice vs nasty box-plots)
    fig, axes = plt.subplots(1, len(ERROR_LEVELS), figsize=(10, 4), sharey=True)
    for ax, ε in zip(axes, ERROR_LEVELS):
        col  = f"{int(ε*100)}%"
        tm   = payoff_sweep[col].to_frame('score')
        tm['niceness'] = tm.index.map(name_to_nice)
        tm.boxplot(column='score', by='niceness', ax=ax)
        ax.set_title(f"ε = {ε:.0%}")
        ax.set_xlabel("is nice?")
        if ax is axes[0]:
            ax.set_ylabel("avg pay-off / round")
    fig.suptitle("C) Nice vs nasty distribution"); fig.tight_layout()
    save_fig("C_boxplot_niceness.png", dpi=300, show=True)

    # ----------------------------------------------------------------- Fig D
    plt.figure(figsize=(5,4))
    plt.plot(gap_df.index, gap_df['gap'], marker='o')
    plt.axhline(0, ls='--', lw=0.8)
    plt.xlabel("ε"); plt.ylabel("gap (nasty – nice)")
    plt.title("D) Class gap shrinkage"); plt.tight_layout()
    save_fig("D_class_gap.png", dpi=300, show=True)

    # ----------------------------------------------------------------- Fig E
    mu, sigma = payoff_sweep.mean(), payoff_sweep.std()
    plt.figure(figsize=(5.5,4))
    plt.errorbar(mu.index, mu, yerr=sigma, fmt='s-', capsize=5)
    plt.xlabel("ε"); plt.ylabel("mean pay-off / round")
    plt.title("E) Global μ ± σ"); plt.grid(ls='--', axis='y', lw=0.4)
    plt.tight_layout()
    save_fig("E_global_mu_sigma.png", dpi=300, show=True)

    # ----------------------------------------------------------------- Fig F
    if MAKE_BARCHART:
        for ε in ERROR_LEVELS:
            lbl = f"{int(ε*100)}%"
            sorted_vals = payoff_sweep[lbl].sort_values()
            colors = ["green" if name_to_nice[n] else "red"
                      for n in sorted_vals.index]
            plt.figure(figsize=(7,6))
            plt.barh(sorted_vals.index, sorted_vals.values, color=colors)
            plt.xlabel("avg pay-off / round")
            plt.title(f"F) Pay-offs at ε = {ε:.0%}")
            plt.tight_layout()
            save_fig(f"F_bar_{lbl}.png", dpi=300, show=True)

    timestamp("All plots rendered – done.")


import numpy as np, pandas as pd, scipy.stats as st

# slopes from payoff_sweep: (μ_10% - μ_0%) / 10
slopes = (payoff_sweep["10%"] - payoff_sweep["0%"]) / 10

nice_slopes  = slopes[[n for n in slopes.index if name_to_nice[n]]]
nasty_slopes = slopes[[n for n in slopes.index if not name_to_nice[n]]]

def mean_ci(data, alpha=0.05):
    m  = data.mean()
    se = data.std(ddof=1) / np.sqrt(len(data))
    t  = st.t.ppf(1 - alpha/2, len(data)-1)
    return m, m - t*se, m + t*se

print("nice   slope, 95% CI:", mean_ci(nice_slopes))
print("nasty  slope, 95% CI:", mean_ci(nasty_slopes))
u,p = st.mannwhitneyu(nice_slopes, nasty_slopes, alternative="two-sided")
print("Mann–Whitney U, p:", u, p)
