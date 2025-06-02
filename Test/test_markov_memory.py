import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Game.game import MarkovGame, MonteCarloGame

# Memory‐0 strategies
from Strategies.m0strategies import AlwaysCooperate, AlwaysDefect, RandomStrategy

# Memory‐1 strategies
from Strategies.m1strategies import TitForTat, WinStayLoseShift, ReverseTitForTat

# Memory‐2 strategies
from Strategies.m2strategies import (
    TitForTwoTats,
    ClearGrudger,
    Pavlov2,
    GenerousTwoTitForTwo,
    SuspiciousTf2T,
)

# Test parameters
rounds = 50
error = 0.05
trials = 50000
tolerance = 2.0  # acceptable point difference between Markov and Monte Carlo

# Group strategies by memory level
memory_0_strats = [
    AlwaysCooperate(),
    AlwaysDefect(),
    RandomStrategy(coop_prob=0.5)
]

memory_1_strats = [
    TitForTat(),
    WinStayLoseShift(),
    ReverseTitForTat()
]

memory_2_strats = [
    TitForTwoTats(),
    ClearGrudger(),
    Pavlov2(),
    GenerousTwoTitForTwo(),
    SuspiciousTf2T(),
]

# Helper to run a single matchup
def run_matchup(strat1, strat2):
    print(f"\n--- Testing {strat1.name} vs {strat2.name} ---")

    # Run Markov Game
    try:
        markov_game = MarkovGame(strat1, strat2, rounds=rounds, error=error)
        m_p1, m_p2, _ = markov_game.run()
        markov_game.printResults()
    except Exception as e:
        print(f"❌ Markov calculation failed: {e}")
        return

    # Run Monte Carlo Game
    try:
        monte_game = MonteCarloGame(strat1, strat2, rounds=rounds, error=error, trials=trials)
        mc_p1, mc_p2 = monte_game.run()
        print(f"{strat1.name} Monte Carlo avg over {trials} trials: {mc_p1:.2f}")
        print(f"{strat2.name} Monte Carlo avg over {trials} trials: {mc_p2:.2f}")
    except Exception as e:
        print(f"❌ Monte Carlo simulation failed: {e}")
        return

    # Compare results
    diff_p1 = abs(mc_p1 - m_p1)
    diff_p2 = abs(mc_p2 - m_p2)
    print(f"Difference → {strat1.name}: {diff_p1:.4f}, {strat2.name}: {diff_p2:.4f}")

    if diff_p1 > tolerance or diff_p2 > tolerance:
        print("❌ WARNING: Monte Carlo and Markov results differ significantly!")
    else:
        print("✅ Check passed: Monte Carlo matches Markov within tolerance.")


# 1. Memory-2 vs Memory-0
for m2 in memory_2_strats:
    for m0 in memory_0_strats:
        run_matchup(m2, m0)

# 2. Memory-2 vs Memory-1
for m2 in memory_2_strats:
    for m1 in memory_1_strats:
        run_matchup(m2, m1)

# 3. (Optional) Memory-2 vs Memory-2 (sanity check)
for i in range(len(memory_2_strats)):
    for j in range(i + 1, len(memory_2_strats)):
        run_matchup(memory_2_strats[i], memory_2_strats[j])