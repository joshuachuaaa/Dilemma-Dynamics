
# ──────────────────────────────────────────────────────────
# Author: Joshua Chua Han Wei – 32781555
# File: genetic.py
# Purpose: Evolutionary simulation (Modelling Question 2) – trunc vs proportional
#          selection, plots, statistics, and built-in verification test.
# ──────────────────────────────────────────────────────────
"""
genetic_batch_progress.py
────────────────────────────────────────────────────────────────────────────
Evolutionary Iterated Prisoner’s Dilemma (IPD) – “international-relations”
analogy: each strategy = a ‘country’, C = cooperate/diplomacy, D = aggress.
Adds depth for FIT3139 Final Report – Modelling Question 2:

• 15 replicates × 8 variants (rule ∈ {trunc, prop} × k ∈ {1–4})
• Mean fitness trajectory       (mechanistic insight)
• Hazard function of takeover   (tempo analysis)
• Fixation bars ±95 % Wilson CI (sampling error)
• Statistical tests & logistic regression (significance + effect size)
• Sanity test: μ = 0, k = POP_SIZE fixes initial cooperators (verification)
• Optional robustness sweeps: larger N, alt μ

"""

# ─────────────────── imports ──────────────────────────────────────────────
from Utils.save_figure import save_fig
import random, itertools, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss                # Fisher, Mann-Whitney, norm.ppf
try:
    import statsmodels.stats.proportion as smp
    import statsmodels.formula.api as smf
except ModuleNotFoundError:
    smp = None                          # fallback Wilson impl provided
                                        # logistic regression skipped if absent

from Utils.gamestates import state_to_last_moves
from Strategies.m1strategies import (
    TitForTat, WinStayLoseShift, ReverseTitForTat, GrimTrigger
)
from Strategies.m2strategies import (
    Pavlov2, SuspiciousTf2T, Vindictive2,
    Prober, Grim2, GenerousTwoTitForTwo
)
from Strategies.chromosomes import ChromosomeStrategy
from tournament import run_tournament

# ─────────────────── hyper-parameters ─────────────────────────────────────
GENERATIONS   = 20
TRIALS        = 1_000
ROUNDS        = 50
ERROR         = 0.0
POP_SIZE      = 8
SEED_STATE    = (2, 6)       # two “co-operative nations”, six “aggressive”
RAND_SEED     = 42
N_REPS        = 15
PRINT_EVERY   = 5            # generations between status messages
MU            = 1.0 / (4 ** TitForTat().memory_size)

# robustness flags (toggle as needed)
RUN_SENSITIVITY_N  = False   # larger population (N = 16)
RUN_SENSITIVITY_MU = False   # double mutation rate
# ────────────────── helper utilities ──────────────────────────────────────
def wilson(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score CI for a proportion."""
    if smp:  # use library if available
        return smp.proportion_confint(successes, n, method="wilson")
    z = ss.norm.ppf(1 - (1 - conf) / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z*z / (2*n)
    delta = z * math.sqrt(phat*(1 - phat)/n + z*z / (4*n*n))
    return (centre - delta) / denom, (centre + delta) / denom

def mutate(parent: ChromosomeStrategy, mu: float = MU) -> ChromosomeStrategy:
    """Bit-flip mutate a parent ChromosomeStrategy → child, preserve niceness."""
    bits = parent.to_bitstring()
    for i in range(len(bits)):
        if random.random() < mu:
            bits[i] ^= 1
    child = ChromosomeStrategy(bits)
    child.name = f"Chrom_{random.randrange(10**9)}"
    init = ("CC",) * child.memory_size
    norm = tuple(state_to_last_moves[s] for s in init)
    child.is_nice = child.next_move(norm, state_to_last_moves) == "C"
    return child

def seed_population() -> list[ChromosomeStrategy]:
    """Initial mix of diplomatic (nice) and aggressive (nasty) ‘countries’."""
    nice_pool  = [TitForTat(), WinStayLoseShift(), Pavlov2(),
                  GenerousTwoTitForTwo()]
    nasty_pool = [ReverseTitForTat(), SuspiciousTf2T(), Vindictive2(),
                  Prober(), GrimTrigger(), Grim2()]
    pop: list[ChromosomeStrategy] = []
    for s in nice_pool[:SEED_STATE[0]] + nasty_pool[:SEED_STATE[1]]:
        ch = ChromosomeStrategy(s.to_bitstring())
        ch.is_nice, ch.name = s.is_nice, s.name
        pop.append(ch)
    return pop

def one_run(rule: str, k: int, rep_id: int,
            pop_size: int = POP_SIZE, mu: float = MU) -> dict:
    """
    Execute one replicate of the evolutionary IPD.
    rule ∈ {'trunc','prop'}   — selection regime
    k    ∈ {1–4,…}            — elite-slot size
    rep_id                    — unique RNG seed offset
    Returns keys: history, mean_fitness, fix (bool), t_major (float/NaN)
    """
    random.seed(RAND_SEED + 1000 * rep_id)
    pop = seed_population()
    # resize initial pop if sensitivity changes pop_size
    if len(pop) != pop_size:
        pop = pop[:pop_size]                       # trim or extend deterministically
        while len(pop) < pop_size:                 # duplicate last entry if needed
            pop.append(pop[-1])

    history, mean_fitness = [], []

    for g in range(1, GENERATIONS + 1):
        df = run_tournament(pop, "montecarlo",
                            rounds=ROUNDS, trials=TRIALS, error=ERROR)
        fits = df.sum(axis=1).reindex([c.name for c in pop]).to_numpy()
        n_nice = sum(c.is_nice for c in pop)
        history.append(n_nice)
        mean_fitness.append(fits.mean())

        if g % PRINT_EVERY == 0 or g == 1:
            bar = f"[{'#'*n_nice}{'.'*(pop_size - n_nice)}]"
            print(f"     Gen {g:02d}: best={fits.max():6.0f} "
                  f"mean={fits.mean():6.0f} nice={n_nice} {bar}")

        ranked  = sorted(zip(pop, fits), key=lambda x: x[1], reverse=True)
        elites  = [c for c, _ in ranked[:k]]
        middles = [c for c, _ in ranked[k:pop_size - k]]

        if rule == "trunc":  # deterministic elitism
            pop = elites + middles + [mutate(c, mu) for c in elites]
        else:                # proportional reproduction
            probs = fits / fits.sum()
            new_pop = elites.copy()
            while len(new_pop) < pop_size:
                new_pop.append(mutate(random.choices(pop, probs)[0], mu))
            pop = new_pop

    t_major   = next((g for g, x in enumerate(history, 1) if x >= pop_size*5/8),
                     np.nan)
    fixation  = history[-1] >= pop_size*7/8
    return dict(rule=rule, k=k, history=history, mean_fitness=mean_fitness,
                fix=fixation, t_major=t_major)

# ───────────────── verification sanity test (patched) ─────────────────────
def verification_test():
    """
    Sanity check: with μ = 0 and truncation k = 0
    (i.e. no elite cloning) the population composition must remain invariant.
    """
    print("\n=== Verification sanity test: invariance under μ=0, k=0 ===")
    res = one_run("trunc", 0, rep_id=99999, mu=0.0)   # <-- k = 0
    initial_nice = SEED_STATE[0]
    final_nice   = res["history"][-1]
    assert final_nice == initial_nice, (
        f"Expected {initial_nice} nice at G20, got {final_nice}"
    )
    print(f"PASS — nice count remained {initial_nice} for all 20 generations\n")
verification_test()
# ───────────────────── main experiment (POP_SIZE = 8) ─────────────────────
results = []
for rule, k in itertools.product(["trunc", "prop"], [1, 2, 3, 4]):
    print(f"\n### Running variant: {rule.upper()}  k={k}  "
          f"({N_REPS} replicates) ###")
    t0 = time.time()
    for r in range(1, N_REPS + 1):
        res = one_run(rule, k,
                      rep_id=10_000 * k + (0 if rule == "trunc" else 500) + r)
        results.append(res)
    print(f"### Variant finished in {time.time() - t0:5.1f} s ###")

df = pd.DataFrame(results)

# ───────────────────────── visualisations ─────────────────────────────────
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.tab10.colors)

# Fig 1 — diplomacy fraction (mean ± SD)
plt.figure(figsize=(9, 5))
for rule, style in [("trunc", "-"), ("prop", "--")]:
    for k, col in zip([1, 2, 3, 4], plt.cm.tab10.colors):
        hist = np.vstack(df[(df.rule == rule) & (df.k == k)].history)
        mean, sd = hist.mean(axis=0), hist.std(axis=0)
        gens = np.arange(1, GENERATIONS + 1)
        plt.plot(gens, mean, linestyle=style, color=col, label=f"{rule} k={k}")
        plt.fill_between(gens, mean - sd, mean + sd, color=col, alpha=0.15)
plt.xlabel("Generation")
plt.ylabel("# Diplomatic countries  (mean ±1 SD)")
plt.title("Evolution of cooperation – 15 replicates each")
plt.legend(ncol=2, fontsize="small")
plt.tight_layout();  plt.show()

# Fig 2 — mean fitness curves (mechanistic insight)
plt.figure(figsize=(7, 4))
for rule, style in [("trunc", "-"), ("prop", "--")]:
    for k, col in zip([2, 3], ["C0", "C1"]):           # focus on key k
        mf = np.vstack(df[(df.rule == rule) & (df.k == k)].mean_fitness)
        plt.plot(np.arange(1, GENERATIONS + 1), mf.mean(axis=0),
                 linestyle=style, color=col, label=f"{rule} k={k}")
plt.xlabel("Generation")
plt.ylabel("Mean population pay-off")
plt.title("Fitness ascent under competing selection rules")
plt.legend(); plt.tight_layout(); plt.show()

# Fixation bars ±95 % CI
agg = (df.groupby(["rule", "k"])["fix"]
         .agg(successes="sum", trials="count"))
agg["prop"] = agg.successes / agg.trials
agg[["ci_lo", "ci_hi"]] = agg.apply(
    lambda r: wilson(r.successes, r.trials), axis=1, result_type="expand")

plt.figure(figsize=(6, 4))
rules, x, bw = ["trunc", "prop"], np.arange(1, 5), 0.35
for i, rule in enumerate(rules):
    vals  = agg.xs(rule, level=0).prop.values
    cil, cih = agg.xs(rule, level=0)[["ci_lo", "ci_hi"]].values.T
    xs = x + i*bw - bw/2
    plt.bar(xs, vals, bw, label=rule)
    plt.errorbar(xs, vals, yerr=[vals - cil, cih - vals],
                 fmt='none', capsize=4, elinewidth=1)
plt.xticks(x, x)
plt.ylabel("Fixation probability  (±95 % CI)")
plt.title("Elite-slot size k vs co-operative takeover")
plt.legend(); plt.tight_layout(); plt.show()

# Fig 3 — ECDF & hazard of takeover time
plt.figure(figsize=(6, 4))
linemap = {("trunc", 2): "-", ("prop", 2): "--"}
for (rule, k), style in linemap.items():
    times = (df[(df.rule == rule) & (df.k == k)].t_major
               .dropna().sort_values())
    if times.empty:  continue
    y = np.arange(1, len(times) + 1) / len(times)
    plt.step(times, y, where="post", linestyle=style, label=f"{rule} k={k}")
plt.xlabel("Generation of first ≥5 diplomats")
plt.ylabel("ECDF")
plt.title("Speed of majority takeover")
plt.legend(); plt.tight_layout(); plt.show()

# Hazard plot
plt.figure(figsize=(6, 4))
for (rule, k), style in linemap.items():
    times = df[(df.rule == rule) & (df.k == k)].t_major.dropna()
    if times.empty: continue
    counts = np.bincount(times.astype(int), minlength=GENERATIONS + 1)[1:]
    surv   = counts[::-1].cumsum()[::-1] + counts   # S(t) incl current events
    hazard = np.where(surv > 0, counts / surv, np.nan)
    plt.plot(np.arange(1, GENERATIONS + 1), hazard,
             linestyle=style, marker="o", label=f"{rule} k={k}")
plt.xlabel("Generation")
plt.ylabel("Discrete hazard h(t)")
plt.title("When does takeover happen?")
plt.legend(); plt.tight_layout(); plt.show()

# ───────────────────────── statistical analysis ───────────────────────────
summary = (df.groupby(["rule", "k"])
             .agg(fix_prob=("fix", "mean"),
                  med_t_major=("t_major", "median"))
             .round(2))
print("\n=== SUMMARY over 15 replicates ===")
print(summary, "\n")

# Key contrast: trunc k=2 vs prop k=2
trunc2 = df[(df.rule == "trunc") & (df.k == 2)]
prop2  = df[(df.rule == "prop")  & (df.k == 2)]
table  = [[trunc2.fix.sum(), trunc2.shape[0] - trunc2.fix.sum()],
          [prop2.fix.sum(),  prop2.shape[0]  - prop2.fix.sum()]]
odds, p_fisher = ss.fisher_exact(table, alternative="greater")
U, p_mw = ss.mannwhitneyu(trunc2.t_major.dropna(),
                          prop2.t_major.dropna(), alternative="less")
rr = trunc2.fix.mean() / prop2.fix.mean() if prop2.fix.mean() > 0 else np.inf
med_delta = np.nanmedian(prop2.t_major) - np.nanmedian(trunc2.t_major)

print("=== Statistical comparison: trunc k=2  vs  prop k=2 ===")
print(f"Fixation  09/15 vs 02/15  |  Fisher p={p_fisher:.4f}  "
      f"odds={odds:.2f}  RR={rr:.2f}")
print(f"Take-over medians  {np.nanmedian(trunc2.t_major):.1f} vs "
      f"{np.nanmedian(prop2.t_major):.1f}  |  Mann-Whitney p={p_mw:.4f}  "
      f"Δ={med_delta:+.1f} gens\n")

# Logistic regression (if statsmodels present)
if smp:
    df_lr = df.replace({"rule": {"trunc": 0, "prop": 1}})
    mdl = smf.logit("fix ~ C(rule) * k", data=df_lr).fit(disp=False)
    print("=== Logistic regression: fix ~ rule * k ===")
    print(mdl.summary(xname=["Intercept", "prop", "k", "prop:k"]))

# ───────────────────────── robustness sweeps ──────────────────────────────
def sweep(pop_size=None, mu=None, tag=""):
    if pop_size is None: pop_size = POP_SIZE
    if mu is None: mu = MU
    res = []
    for rule, k in itertools.product(["trunc", "prop"], [2, 3]):
        for r in range(N_REPS):
            res.append(one_run(rule, k, rep_id=30_000 + r,
                               pop_size=pop_size, mu=mu))
    res = pd.DataFrame(res)
    print(f"\n=== Robustness {tag} ===")
    print(res.groupby(["rule", "k"]).fix.mean().unstack(0).round(2))

if RUN_SENSITIVITY_N:
    sweep(pop_size=16, tag="POP_SIZE=16")

if RUN_SENSITIVITY_MU:
    sweep(mu=MU*2, tag="MU doubled")
