import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Strategies.chromosomes import ChromosomeStrategy

# test_chromosome_strategy.py

import pytest
import math
from Strategies.chromosomes import ChromosomeStrategy

# 1. MEMORY‐1 (m = 1).  Here 4**1 = 4 bits.
#    Let’s pick chromosome = "0101" → bit‐list = [0,1,0,1].
#    That means:
#       history=('C','C') → index=0 → lookup_table[0] = 'C'
#       history=('C','D') → index=1 → lookup_table[1] = 'D'
#       history=('D','C') → index=2 → lookup_table[2] = 'C'
#       history=('D','D') → index=3 → lookup_table[3] = 'D'
def test_memory1_basic_next_move_and_probabilities():
    chrom_str = "0101"
    cs = ChromosomeStrategy(chromosome=chrom_str)

    # Verify memory_size and lookup_table length
    assert cs.memory_size == 1, "Expected memory_size = 1 for a 4‐bit chromosome"
    assert len(cs.lookup_table) == 4, "Lookup table should have length 4 when m=1"

    # The four possible single‐round joint‐outcomes in lexicographic order:
    #   index 0: ('C','C')
    #   index 1: ('C','D')
    #   index 2: ('D','C')
    #   index 3: ('D','D')

    # 1a. Check next_move for each possible last_state
    assert cs.next_move(('C','C'), None) == 'C'
    assert cs.next_move(('C','D'), None) == 'D'
    assert cs.next_move(('D','C'), None) == 'C'
    assert cs.next_move(('D','D'), None) == 'D'

    # 1b. Check move_probabilities returns a deterministic distribution
    #     We expect either {"C":1.0,"D":0.0} or {"C":0.0,"D":1.0}
    prob_cc = cs.move_probabilities(('C','C'), None)
    assert prob_cc == {"C": 1.0, "D": 0.0}

    prob_cd = cs.move_probabilities(('C','D'), None)
    assert prob_cd == {"C": 0.0, "D": 1.0}

    prob_dc = cs.move_probabilities(('D','C'), None)
    assert prob_dc == {"C": 1.0, "D": 0.0}

    prob_dd = cs.move_probabilities(('D','D'), None)
    assert prob_dd == {"C": 0.0, "D": 1.0}


# 2. MEMORY‐2 (m = 2).  Chromosome length must be 4**2 = 16 bits.
#    We construct a simple 16‐bit string (0000 1111 0011 1100)
#    so that we know exactly which index → which move.
def test_memory2_indexing_and_next_move():
    # Let’s build a 16‐bit string in this order:
    #   First 4 bits = "0000"  → indices 0..3  → all 'C'
    #   Next 4 bits = "1111"  → indices 4..7  → all 'D'
    #   Next 4 bits = "0011"  → indices 8..11 → two 'C', then two 'D'
    #   Last 4 bits = "1100"  → indices 12..15 → two 'D', then two 'C'
    chrom_str = "0000111100111100"
    cs = ChromosomeStrategy(chromosome=chrom_str)
    assert cs.memory_size == 2
    assert len(cs.lookup_table) == 16

    # Build the complete list of all 16 possible two‐round histories,
    # in exactly the order that ChromosomeStrategy’s internal mapping uses:
    #
    #   single_round_outcomes = [('C','C'), ('C','D'), ('D','C'), ('D','D')]
    #   all_histories = product(single_round_outcomes, repeat=2)
    #   → [
    #         (('C','C'),('C','C')),  # idx 0
    #         (('C','C'),('C','D')),  # idx 1
    #         (('C','C'),('D','C')),  # idx 2
    #         (('C','C'),('D','D')),  # idx 3
    #
    #         (('C','D'),('C','C')),  # idx 4
    #         (('C','D'),('C','D')),  # idx 5
    #         (('C','D'),('D','C')),  # idx 6
    #         (('C','D'),('D','D')),  # idx 7
    #
    #         (('D','C'),('C','C')),  # idx 8
    #         (('D','C'),('C','D')),  # idx 9
    #         (('D','C'),('D','C')),  # idx 10
    #         (('D','C'),('D','D')),  # idx 11
    #
    #         (('D','D'),('C','C')),  # idx 12
    #         (('D','D'),('C','D')),  # idx 13
    #         (('D','D'),('D','C')),  # idx 14
    #         (('D','D'),('D','D')),  # idx 15
    #      ]
    from itertools import product
    single_out = list(product(['C','D'], repeat=2))
    all_histories = list(product(single_out, repeat=2))

    # Sanity‐check: we should have exactly 16 distinct two‐round histories
    assert len(all_histories) == 16

    # 2a. Indices 0..3 in our chromosome are “0000” → should all be 'C'
    for idx in range(0, 4):
        hist = all_histories[idx]
        assert cs.history_to_index[hist] == idx
        assert cs.next_move(hist, None) == 'C'

    # 2b. Indices 4..7 in our chromosome are “1111” → should all be 'D'
    for idx in range(4, 8):
        hist = all_histories[idx]
        assert cs.history_to_index[hist] == idx
        assert cs.next_move(hist, None) == 'D'

    # 2c. Indices 8..9 are “00” → 'C',  indices 10..11 are “11” → 'D'
    #     because we used “0011” for bits 8..11
    for idx in [8, 9]:
        hist = all_histories[idx]
        assert cs.next_move(hist, None) == 'C'
    for idx in [10, 11]:
        hist = all_histories[idx]
        assert cs.next_move(hist, None) == 'D'

    # 2d. Indices 12..13 are “11” → 'D',  indices 14..15 are “00” → 'C'
    for idx in [12, 13]:
        hist = all_histories[idx]
        assert cs.next_move(hist, None) == 'D'
    for idx in [14, 15]:
        hist = all_histories[idx]
        assert cs.next_move(hist, None) == 'C'


# 3. INVALID LENGTH ⇒ ValueError
#    - length=3  is not a power of 4
#    - length=5  is not a power of 4
#    - length=8  is not a power of 4
#    - length=15 is not a power of 4
#    - length=5 when passed as list or string should raise
@pytest.mark.parametrize("bad_chrom", [
    "000",       # length=3
    "0101010",   # length=7
    [0,1,0,1,1], # length=5
    "0"*15       # length=15
])
def test_invalid_length_raises_value_error(bad_chrom):
    with pytest.raises(ValueError) as excinfo:
        ChromosomeStrategy(chromosome=bad_chrom)

    msg = str(excinfo.value).lower()
    assert "not a power of 4" in msg or "does not match expected size" in msg


# 4. EDGE CASES: All‐“C” or All‐“D” for m=1
def test_all_cooperate_and_all_defect_memory1():
    # (a) All “0” → always 'C'
    cs_all_c = ChromosomeStrategy(chromosome="0000")
    for hist in [('C','C'), ('C','D'), ('D','C'), ('D','D')]:
        assert cs_all_c.next_move(hist, None) == 'C'
        assert cs_all_c.move_probabilities(hist, None) == {"C": 1.0, "D": 0.0}

    # (b) All “1” → always 'D'
    cs_all_d = ChromosomeStrategy(chromosome="1111")
    for hist in [('C','C'), ('C','D'), ('D','C'), ('D','D')]:
        assert cs_all_d.next_move(hist, None) == 'D'
        assert cs_all_d.move_probabilities(hist, None) == {"C": 0.0, "D": 1.0}


# 5. RANDOMIZED SPOT‐CHECK for m=2:
#    Generate a random 16‐bit chromosome, then re‐compute what each index’s bit is,
#    and confirm that next_move returns the correct bit→move.
def test_randomized_spot_check_memory2():
    import random

    # Build a random 16‐bit list:
    bits = [random.choice([0,1]) for _ in range(16)]
    # Convert to string so ChromosomeStrategy handles it:
    chrom_str = "".join(str(b) for b in bits)
    cs_rand = ChromosomeStrategy(chromosome=chrom_str)
    assert cs_rand.memory_size == 2
    assert len(cs_rand.lookup_table) == 16

    # Re‐generate the “all_histories” ordering to check indices:
    from itertools import product
    single_out = list(product(['C','D'], repeat=2))
    all_histories = list(product(single_out, repeat=2))

    # Check 5 random indices
    for _ in range(5):
        idx = random.randrange(16)
        hist = all_histories[idx]
        move_expected = 'C' if bits[idx] == 0 else 'D'
        assert cs_rand.history_to_index[hist] == idx
        assert cs_rand.next_move(hist, None) == move_expected


# ======= RUN THESE TESTS WITH:  pytest -q test_chromosome_strategy.py  =======

if __name__ == "__main__":
    # Allow running this file directly with plain Python (it will invoke pytest main)
    pytest.main(["-q", __file__])
