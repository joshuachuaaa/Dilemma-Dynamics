# ──────────────────────────────────────────────────────────
# File   : Strategies/chromosomes.py
# Author : Joshua Chua Han Wei – 32781555
# Purpose: Bit-string / genetic “chromosome” strategy wrapper (variable memory)
# ──────────────────────────────────────────────────────────

import math
from itertools import product
from Utils.gamestates import states            # ["CC","CD","DC","DD"]
from Strategies.strategy import Strategy


class ChromosomeStrategy(Strategy):
    """
    Strategy encoded as a bit-string of length 4**m.
    Bit 0 → 'C', 1 → 'D'.  The index is determined by the last-m outcomes
    (each outcome ∈ {"CC","CD","DC","DD"}).
    """

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(self, chromosome):
        super().__init__()
        self.name = "ChromosomeStrategy"
        self.is_nice = None          # set externally later

        # 1) normalise chromosome to list[int]
        if isinstance(chromosome, str):
            chromosome = [int(bit) for bit in chromosome]
        length = len(chromosome)

        # 2) infer memory size m  (length must be 4**m)
        m_float = math.log(length, 4)
        if not m_float.is_integer():
            raise ValueError(
                f"Chromosome length {length} is not a power of 4 "
                "(i.e. not 4**m for any integer m)."
            )
        self.memory_size = int(m_float)

        # 3) build lookup table
        self.lookup_table = ["C" if bit == 0 else "D" for bit in chromosome]

        # 4) pre-compute mapping: history tuple → index
        self.history_to_index = {}
        for idx, hist in enumerate(product(states, repeat=self.memory_size)):
            # hist ('CC','DC',…) → (('C','C'),('D','C'),…)
            key = tuple((s[0], s[1]) for s in hist)
            self.history_to_index[key] = idx

    # ------------------------------------------------------------------ #
    # Strategy API
    # ------------------------------------------------------------------ #
    def next_move(self, last_state, state_matrix):
        """
        Accepts `last_state` in any of these forms:

        • "CC"                               — outcome string
        • ("CC","DC",…)                      — tuple/list of outcome strings
        • (('C','C'),('D','C'),…)            — tuple of char-pairs
        • ('C','D') when m == 1              — two single-char strings
        """

        # 1) convert to flat tuple  `seq`
        if isinstance(last_state, str):
            seq = (last_state,)

        elif (isinstance(last_state, (list, tuple))
              and len(last_state) == 2
              and all(isinstance(x, str) and len(x) == 1 for x in last_state)):
            # memory-1 special case: ('C','D') → ('CD',)
            seq = ("".join(last_state),)

        elif (isinstance(last_state, (list, tuple))
              and all(isinstance(s, str) for s in last_state)):
            seq = tuple(last_state)

        else:
            # assume tuple of char-pairs already
            seq = tuple(last_state)

        # 2) keep only last m entries
        if len(seq) != self.memory_size:
            seq = seq[-self.memory_size:]

        # 3) normalise each element to a char-pair
        normalized = []
        for s in seq:
            if isinstance(s, str) and len(s) == 2:           # "CD"
                normalized.append((s[0], s[1]))

            elif (isinstance(s, (list, tuple)) and len(s) == 2
                  and all(isinstance(c, str) and len(c) == 1 for c in s)):
                normalized.append(tuple(s))                  # ('C','D')

            else:
                raise ValueError(
                    f"Bad history element {s!r}. "
                    "Expected outcome string 'CD' or 2-char tuple ('C','D')."
                )

        key = tuple(normalized)
        idx = self.history_to_index[key]
        return self.lookup_table[idx]

    def move_probabilities(self, last_state, state_matrix):
        move = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if move == "C" else 0.0,
                "D": 1.0 if move == "D" else 0.0}
