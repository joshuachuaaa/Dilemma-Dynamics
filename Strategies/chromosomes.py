import math
from itertools import product
from gamestates import states  # ["CC","CD","DC","DD"]
from Strategies.strategy import Strategy

class ChromosomeStrategy(Strategy):
    def __init__(self, chromosome):
        super().__init__()
        self.name = "ChromosomeStrategy"
        self.is_nice = None  # to be set externally

        # 1) Normalize chromosome to list of ints
        if isinstance(chromosome, str):
            chromosome = [int(bit) for bit in chromosome]
        length = len(chromosome)

        # 2) Infer memory_size m from length = 4**m
        m = math.log(length, 4)
        if not m.is_integer():
            raise ValueError(f"Chromosome length {length} is not 4**m for integer m")
        self.memory_size = int(m)

        # 3) Build lookup_table: list of 'C'/'D'
        self.lookup_table = ['C' if bit == 0 else 'D' for bit in chromosome]

        # 4) Precompute mapping from each possible m‐tuple of states → index
        self.history_to_index = {}
        for idx, hist in enumerate(product(states, repeat=self.memory_size)):
            # hist is e.g. ('CC','DC',...)
            key = tuple((s[0], s[1]) for s in hist)
            self.history_to_index[key] = idx

    def next_move(self, last_state, state_matrix):
        """
        Accepts last_state in one of:
          • a string "CC"
          • a tuple/list of state‐strings ["CC","DC",...]
          • a tuple of char‐pairs [(C,C),(D,C),...]
        Truncates to the last self.memory_size elements, normalizes to
        a tuple of char-pairs, and looks up the action.
        """
        # 1) Turn anything into a flat tuple `seq`
        if isinstance(last_state, str):
            seq = (last_state,)
        elif isinstance(last_state, (list, tuple)) and all(isinstance(s, str) for s in last_state):
            seq = tuple(last_state)
        else:
            # assume it's already a tuple of char‐pairs
            seq = tuple(last_state)

        # 2) Truncate to the last m entries
        if len(seq) != self.memory_size:
            seq = seq[-self.memory_size :]

        # 3) Normalize each element into a char‐pair
        normalized = []
        for s in seq:
            if isinstance(s, str):
                # map "CD" → ('C','D')
                normalized.append((s[0], s[1]))
            else:
                # assume it's already ('C','D') form
                normalized.append(s)
        key = tuple(normalized)

        # 4) Lookup
        idx = self.history_to_index[key]
        return self.lookup_table[idx]

    def move_probabilities(self, last_state, state_matrix):
        move = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if move == "C" else 0.0,
                "D": 1.0 if move == "D" else 0.0}
