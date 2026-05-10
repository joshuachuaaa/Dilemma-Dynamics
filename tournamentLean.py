# ----------------------------------------------------------
# Author: Joshua Chua Han Wei
# File: tournamentLean.py
# Purpose: Noise-sensitivity study for Monte-Carlo strategy pay-offs.
# ----------------------------------------------------------

# ------------------------------------------------------------------- imports
from Utils.save_figure import save_fig
from Utils.random_seed import set_seed
import time
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

from Experiments.competitors import default_competitors
from Experiments.tournament_runner import run_tournament as run_pairwise_tournament
from Experiments.tournament_runner import total_payoffs_per_round


# ---------------------------------------------------------------- constants
ROUNDS       = 50
TRIALS       = 10_000
ERROR_LEVELS = [0.00, 0.05, 0.10]
MAKE_BARCHART = False          # set True if want per-epsilon bar charts

# ----------------------------------------------------------- competitor list
competitors = default_competitors()

name_to_nice   = {s.name: s.is_nice     for s in competitors}
name_to_memory = {s.name: s.memory_size for s in competitors}
strategy_names = [s.name for s in competitors]

# ------------------------------------------------------------- helpers
def timestamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def run_tournament(err: float) -> pd.Series:
    """Round-robin Monte-Carlo; returns per-strategy avg pay-off / round."""
    payoff_matrix = run_pairwise_tournament(
        competitors,
        engine_type="montecarlo",
        rounds=ROUNDS,
        trials=TRIALS,
        error=err,
    )
    return total_payoffs_per_round(payoff_matrix, ROUNDS)

def class_gap(series: pd.Series) -> tuple[float,float,float]:
    cooperative  = series[[k for k in series.index if name_to_nice[k]]].mean()
    exploitative = series[[k for k in series.index if not name_to_nice[k]]].mean()
    return cooperative, exploitative, exploitative - cooperative


def run_noise_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the Monte-Carlo noise sweep and return payoffs plus class gaps."""
    timestamp("Starting Monte-Carlo sweep")
    payoff_sweep = pd.DataFrame(index=strategy_names)
    gap_records  = []

    for epsilon in ERROR_LEVELS:
        label = f"{int(epsilon*100)}%"
        timestamp(f"Simulating epsilon = {epsilon:.0%}")
        series = run_tournament(epsilon)
        payoff_sweep[label] = series

        cooperative_m, exploitative_m, gap = class_gap(series)
        gap_records.append({"epsilon": epsilon, "cooperative": cooperative_m,
                            "exploitative": exploitative_m, "gap": gap})
        timestamp(
            f"  cooperative = {cooperative_m:.2f}, "
            f"exploitative = {exploitative_m:.2f}, gap = {gap:+.2f}"
        )

    gap_df = pd.DataFrame(gap_records).set_index("epsilon")
    print("\n=== Class gap summary ===")
    print(gap_df.round(3))
    return payoff_sweep, gap_df


def plot_noise_sweep(payoff_sweep: pd.DataFrame, gap_df: pd.DataFrame) -> None:
    """Render and persist the standard noise-sensitivity figures."""
    plt.figure(figsize=(9, 6))
    for strat in payoff_sweep.index:
        plt.plot(
            payoff_sweep.columns,
            payoff_sweep.loc[strat],
            marker='o', lw=1, alpha=0.8,
            label=strat,
        )

    plt.xlabel("trembling-hand error epsilon")
    plt.ylabel("avg pay-off / round")
    plt.title("A) Strategy performance vs noise")
    plt.grid(ls='--', lw=0.4)

    plt.legend(
        bbox_to_anchor=(1.03, 1),
        loc="upper left",
        fontsize="small",
        frameon=False
    )
    plt.tight_layout()
    save_fig("A_performance_vs_noise.png", dpi=300, show=True)

    fig, axes = plt.subplots(1, len(ERROR_LEVELS), figsize=(12, 4), sharey=True)
    for ax, epsilon in zip(axes, ERROR_LEVELS):
        col = f"{int(epsilon*100)}%"
        tmp = payoff_sweep[col].to_frame('score')
        tmp['memory'] = tmp.index.map(name_to_memory)
        tmp.boxplot(column='score', by='memory', ax=ax)
        ax.set_title(f"epsilon = {epsilon:.0%}")
        ax.set_xlabel("memory (m)")
        if ax is axes[0]:
            ax.set_ylabel("avg pay-off / round")
    fig.suptitle("B) Distribution by memory size"); fig.tight_layout()
    save_fig("B_boxplot_memory.png", dpi=300, show=True)

    fig, axes = plt.subplots(1, len(ERROR_LEVELS), figsize=(10, 4), sharey=True)
    for ax, epsilon in zip(axes, ERROR_LEVELS):
        col  = f"{int(epsilon*100)}%"
        tm   = payoff_sweep[col].to_frame('score')
        tm['cooperative'] = tm.index.map(name_to_nice)
        tm.boxplot(column='score', by='cooperative', ax=ax)
        ax.set_title(f"epsilon = {epsilon:.0%}")
        ax.set_xlabel("is cooperative?")
        if ax is axes[0]:
            ax.set_ylabel("avg pay-off / round")
    fig.suptitle("C) Cooperative vs exploitative distribution"); fig.tight_layout()
    save_fig("C_boxplot_cooperative_class.png", dpi=300, show=True)

    plt.figure(figsize=(5,4))
    plt.plot(gap_df.index, gap_df['gap'], marker='o')
    plt.axhline(0, ls='--', lw=0.8)
    plt.xlabel("epsilon"); plt.ylabel("gap (exploitative - cooperative)")
    plt.title("D) Class gap shrinkage"); plt.tight_layout()
    save_fig("D_class_gap.png", dpi=300, show=True)

    mu, sigma = payoff_sweep.mean(), payoff_sweep.std()
    plt.figure(figsize=(5.5,4))
    plt.errorbar(mu.index, mu, yerr=sigma, fmt='s-', capsize=5)
    plt.xlabel("epsilon"); plt.ylabel("mean pay-off / round")
    plt.title("E) Global mean and standard deviation"); plt.grid(ls='--', axis='y', lw=0.4)
    plt.tight_layout()
    save_fig("E_global_mu_sigma.png", dpi=300, show=True)

    if MAKE_BARCHART:
        for epsilon in ERROR_LEVELS:
            lbl = f"{int(epsilon*100)}%"
            sorted_vals = payoff_sweep[lbl].sort_values()
            colors = ["green" if name_to_nice[n] else "red"
                      for n in sorted_vals.index]
            plt.figure(figsize=(7,6))
            plt.barh(sorted_vals.index, sorted_vals.values, color=colors)
            plt.xlabel("avg pay-off / round")
            plt.title(f"F) Pay-offs at epsilon = {epsilon:.0%}")
            plt.tight_layout()
            save_fig(f"F_bar_{lbl}.png", dpi=300, show=True)


def mean_ci(data, alpha=0.05):
    m  = data.mean()
    se = data.std(ddof=1) / np.sqrt(len(data))
    t  = st.t.ppf(1 - alpha/2, len(data)-1)
    return m, m - t*se, m + t*se


def print_slope_statistics(payoff_sweep: pd.DataFrame) -> None:
    """Compare noise sensitivity by strategy class."""
    slopes = (payoff_sweep["10%"] - payoff_sweep["0%"]) / 10
    cooperative_slopes = slopes[[n for n in slopes.index if name_to_nice[n]]]
    exploitative_slopes = slopes[[n for n in slopes.index if not name_to_nice[n]]]

    print("cooperative  slope, 95% CI:", mean_ci(cooperative_slopes))
    print("exploitative slope, 95% CI:", mean_ci(exploitative_slopes))
    u, p = st.mannwhitneyu(
        cooperative_slopes, exploitative_slopes, alternative="two-sided"
    )
    print("Mann-Whitney U, p:", u, p)


def main() -> None:
    set_seed()
    payoff_sweep, gap_df = run_noise_sweep()
    plot_noise_sweep(payoff_sweep, gap_df)
    print_slope_statistics(payoff_sweep)
    timestamp("All plots rendered - done.")


if __name__ == "__main__":
    main()
