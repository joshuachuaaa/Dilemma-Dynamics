import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game import MarkovGame, MonteCarloGame
from Strategies.m1strategies import (
    TitForTat, 
    WinStayLoseShift, ReverseTitForTat
)
from Strategies.m0strategies import ( AlwaysDefect, AlwaysCooperate,RandomStrategy
)

# Define strategy pairs to test
strategy_pairs = [
    (TitForTat(), AlwaysDefect()),
    (TitForTat(), AlwaysCooperate()),
    (TitForTat(), RandomStrategy(coop_prob=0.5)),
    (AlwaysDefect(), AlwaysCooperate()),
    (WinStayLoseShift(), ReverseTitForTat()),
    (RandomStrategy(coop_prob=0.3), RandomStrategy(coop_prob=0.7)),
]

rounds = 30
error = 0.005
trials = 100000
initial_state = 'CC'
tolerance = 2  # allowed difference between methods

for strat1, strat2 in strategy_pairs:
    print(f"\n--- Testing {strat1.name} vs {strat2.name} ---")

    # Run Markov Game
    markov_game = MarkovGame(strat1, strat2, rounds=rounds, error=error, initial_state=initial_state)
    markov_p1, markov_p2, _ = markov_game.run()
    markov_game.printResults()

    # Run Monte Carlo Game
    monte_game = MonteCarloGame(strat1, strat2, rounds=rounds, error=error, initial_state=initial_state, trials=trials)
    mc_p1, mc_p2 = monte_game.run()
    print(f"{strat1.name} Monte Carlo avg over {trials} trials: {mc_p1:.2f}")
    print(f"{strat2.name} Monte Carlo avg over {trials} trials: {mc_p2:.2f}")

    # Check differences
    diff_p1 = abs(mc_p1 - markov_p1)
    diff_p2 = abs(mc_p2 - markov_p2)

    print(f"Difference → {strat1.name}: {diff_p1:.4f}, {strat2.name}: {diff_p2:.4f}")

    if diff_p1 > tolerance or diff_p2 > tolerance:
        print("❌ WARNING: Monte Carlo and Markov results differ significantly!\n")
    else:
        print("✅ Check passed: Monte Carlo matches Markov expected values within tolerance.\n")
