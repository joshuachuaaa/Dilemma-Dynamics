# test_symmetry.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# test_symmetry_full.py

import unittest
import random

from Game.game import MarkovGame, MonteCarloGame

from Strategies.m0strategies import AlwaysDefect, RandomStrategy
from Strategies.m1strategies import TitForTat, WinStayLoseShift, ReverseTitForTat, GrimTrigger
from Strategies.m2strategies import (
    TitForTwoTats,
    ClearGrudger,
    Pavlov2,
    GenerousTwoTitForTwo,
    SuspiciousTf2T,
    Prober,
    Grim2,
    Vindictive2,
)
from Strategies.m3strategies import (
    TitForThreeTats,
    TwoForgiveOnePunish,
    ThreeGrudger,
    PatternFollower3,
    Pavlov3,
    Generous3,
    UnforgivingPatternHunter,
)

class TestGameSymmetryFull(unittest.TestCase):
    def setUp(self):
        # Instantiate one of each strategy:
        self.strategies = [
            # Memory-0
            AlwaysDefect(),
            RandomStrategy(coop_prob=0.5),

            # Memory-1
            TitForTat(),
            WinStayLoseShift(),
            ReverseTitForTat(),
            GrimTrigger(),

            # Memory-2
            TitForTwoTats(),
            ClearGrudger(),
            Pavlov2(),
            GenerousTwoTitForTwo(),
            SuspiciousTf2T(),
            Prober(),
            Grim2(),
            Vindictive2(),

            # Memory-3
            TitForThreeTats(),
            TwoForgiveOnePunish(),
            ThreeGrudger(),
            PatternFollower3(),
            Pavlov3(),
            Generous3(),
            UnforgivingPatternHunter(),
        ]

        # Tolerance for floating-point comparisons:
        self.markov_tol = 1e-8
        self.mc_tol = 1e-2  # Monte Carlo sampling noise

    # def test_markov_symmetry_full(self):
    #     rounds = 50
    #     error = 0.0

    #     n = len(self.strategies)
    #     for i in range(n):
    #         for j in range(i+1, n):
    #             strat_i = self.strategies[i]
    #             strat_j = self.strategies[j]

    #             # Reset any internal state before each pairing
    #             if hasattr(strat_i, "reset"):
    #                 strat_i.reset()
    #             if hasattr(strat_j, "reset"):
    #                 strat_j.reset()

    #             # Run MarkovGame(strat_i, strat_j)
    #             game_ij = MarkovGame(strat_i, strat_j, rounds=rounds, error=error)
    #             score_i_ij, score_j_ij, _ = game_ij.run()

    #             # Reset stateful strategies again before swapping
    #             if hasattr(strat_i, "reset"):
    #                 strat_i.reset()
    #             if hasattr(strat_j, "reset"):
    #                 strat_j.reset()

    #             # Run MarkovGame(strat_j, strat_i)
    #             game_ji = MarkovGame(strat_j, strat_i, rounds=rounds, error=error)
    #             score_j_ji, score_i_ji, _ = game_ji.run()

    #             # Assert exact symmetry (within tiny floating tolerance)
    #             self.assertAlmostEqual(
    #                 score_i_ij, score_i_ji, delta=self.markov_tol,
    #                 msg=f"Markov symmetry failed for {strat_i.name} vs {strat_j.name}: "
    #                     f"{score_i_ij:.8f} vs {score_i_ji:.8f}"
    #             )
    #             self.assertAlmostEqual(
    #                 score_j_ij, score_j_ji, delta=self.markov_tol,
    #                 msg=f"Markov symmetry failed for {strat_i.name} vs {strat_j.name}: "
    #                     f"{score_j_ij:.8f} vs {score_j_ji:.8f}"
    #             )

    def test_monte_carlo_symmetry_full(self):
        rounds = 50
        trials = 5000
        error = 0.0

        n = len(self.strategies)
        for i in range(n):
            for j in range(i+1, n):
                strat_i = self.strategies[i]
                strat_j = self.strategies[j]

                # Reset internal state and seed before first run
                if hasattr(strat_i, "reset"):
                    strat_i.reset()
                if hasattr(strat_j, "reset"):
                    strat_j.reset()
                random.seed(0)

                game_ij = MonteCarloGame(strat_i, strat_j, rounds=rounds, trials=trials, error=error)
                score_i_ij, score_j_ij = game_ij.run()

                # Reset internal state and seed before swapped run
                if hasattr(strat_i, "reset"):
                    strat_i.reset()
                if hasattr(strat_j, "reset"):
                    strat_j.reset()
                random.seed(0)

                game_ji = MonteCarloGame(strat_j, strat_i, rounds=rounds, trials=trials, error=error)
                score_j_ji, score_i_ji = game_ji.run()

                # Assert approximate symmetry (within sampling‐noise tolerance)
                self.assertAlmostEqual(
                    score_i_ij, score_i_ji, delta=self.mc_tol,
                    msg=f"Monte Carlo symmetry failed for {strat_i.name} vs {strat_j.name}: "
                        f"{score_i_ij:.4f} vs {score_i_ji:.4f}"
                )
                self.assertAlmostEqual(
                    score_j_ij, score_j_ji, delta=self.mc_tol,
                    msg=f"Monte Carlo symmetry failed for {strat_i.name} vs {strat_j.name}: "
                        f"{score_j_ij:.4f} vs {score_j_ji:.4f}"
                )

if __name__ == "__main__":
    unittest.main()
