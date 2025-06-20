# Test/test_tit_for_tat_equivalence.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from Strategies.m1strategies import TitForTat
from Strategies.chromosomes import ChromosomeStrategy
from Utils.gamestates import state_to_last_moves

@pytest.mark.parametrize("state_code", ["CC", "CD", "DC", "DD"])
def test_tit_for_tat_chromosome_equivalence(state_code):
    tft = TitForTat()
    tft_chrom = ChromosomeStrategy(chromosome="0101")
    last_state_for_class = (state_code,)
    expected_move = tft.next_move(last_state_for_class, state_to_last_moves)
    hist = state_to_last_moves[state_code]
    actual_move = tft_chrom.next_move(hist, None)
    assert actual_move == expected_move

    prob1 = tft.move_probabilities(last_state_for_class, state_to_last_moves)
    prob2 = tft_chrom.move_probabilities(hist, None)
    assert prob1 == prob2

if __name__ == "__main__":
    import pytest
    pytest.main(["-q", __file__])
