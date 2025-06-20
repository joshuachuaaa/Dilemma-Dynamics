import unittest
import random
from itertools import product
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gamestates import states, state_to_last_moves
from Strategies.strategy    import Strategy
from Strategies.m1strategies import (
    TitForTat, WinStayLoseShift, ReverseTitForTat, GrimTrigger
)
from Strategies.m2strategies import (
    TitForTwoTats, ClearGrudger, Pavlov2, GenerousTwoTitForTwo, SuspiciousTf2T
)
from Strategies.m3strategies import (
    TitForThreeTats, TwoForgiveOnePunish, ThreeGrudger, PatternFollower3, Pavlov3
)
from Strategies.chromosomes  import ChromosomeStrategy


class TestToBitstringMemory1(unittest.TestCase):
    def test_tit_for_tat(self):
        bits = TitForTat().to_bitstring()
        # ["CC","CD","DC","DD"] → [0,1,0,1]
        self.assertEqual(bits, [0, 1, 0, 1])

    def test_win_stay_lose_shift(self):
        bits = WinStayLoseShift().to_bitstring()
        self.assertEqual(bits, [0, 1, 0, 1])

    def test_reverse_tit_for_tat(self):
        bits = ReverseTitForTat().to_bitstring()
        self.assertEqual(bits, [1, 0, 1, 0])

    def test_grim_trigger(self):
        bits = GrimTrigger().to_bitstring()
        # Each history is fresh → [0,1,0,1]
        self.assertEqual(bits, [0, 1, 0, 1])

    def test_invalid_memory_0(self):
        class M0(Strategy):
            def __init__(self):
                super().__init__()
                self.name = "M0"
                self.memory_size = 0
            def next_move(self, *_): return "C"
            def move_probabilities(self, *_): return {"C":1,"D":0}
        with self.assertRaises(ValueError):
            M0().to_bitstring()


class TestToBitstringMemory2(unittest.TestCase):
    def setUp(self):
        # histories of state‐strings for m=2
        self.histories = list(product(states, repeat=2))

    def test_length_16(self):
        for Strat in (TitForTwoTats, ClearGrudger, Pavlov2, GenerousTwoTitForTwo, SuspiciousTf2T):
            self.assertEqual(len(Strat().to_bitstring()), 16)

    def test_behavior_consistency(self):
        for Strat in (TitForTwoTats, Pavlov2):
            base = Strat()
            bits = base.to_bitstring()
            chrom = ChromosomeStrategy(bits)

            for hist in self.histories:
                # base expects history of strings:
                base.reset()
                expected = base.next_move(hist, state_to_last_moves)

                # chrom expects tuple-of-(move,move):
                key = tuple(state_to_last_moves[s] for s in hist)
                actual = chrom.next_move(key, state_to_last_moves)

                self.assertEqual(actual, expected,
                    f"{Strat.__name__} mismatch on history {hist}")


class TestToBitstringMemory3(unittest.TestCase):
    def setUp(self):
        self.histories = list(product(states, repeat=3))

    def test_length_64(self):
        for Strat in (TitForThreeTats, TwoForgiveOnePunish, ThreeGrudger, PatternFollower3, Pavlov3):
            self.assertEqual(len(Strat().to_bitstring()), 64)

    def test_behavior_consistency(self):
        for Strat in (TitForThreeTats, Pavlov3):
            base = Strat()
            bits = base.to_bitstring()
            chrom = ChromosomeStrategy(bits)

            for hist in random.sample(self.histories, 20):
                base.reset()
                expected = base.next_move(hist, state_to_last_moves)

                key = tuple(state_to_last_moves[s] for s in hist)
                actual = chrom.next_move(key, state_to_last_moves)

                self.assertEqual(actual, expected,
                    f"{Strat.__name__} mismatch on history {hist}")


if __name__ == "__main__":
    unittest.main()
