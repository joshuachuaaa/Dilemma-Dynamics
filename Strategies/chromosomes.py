import math
from itertools import product
from Strategies.strategy import Strategy

class ChromosomeStrategy(Strategy):
    def __init__(self, chromosome):
        """
        chromosome: either
          - a string of 0/1 of length 4**m  (e.g. "010110…"), or
          - a list of ints 0/1 of length 4**m.
        We infer m by requiring len(chromosome) == 4**m.  Then for each of the 4**m
        possible “last-m-round histories” (where each round outcome is one of
        ('C','C'), ('C','D'), ('D','C'), ('D','D')), we assign 0→'C' or 1→'D' from the chromosome.
        """

        super().__init__()
        self.name = "ChromosomeStrategy"
        self.is_nice = None  # could be inferred later if you want

        # If user passed in a string, convert it to a list of ints
        if isinstance(chromosome, str):
            chromosome = [int(bit) for bit in chromosome]

        # Figure out memory_size from the length of the chromosome:
        self.memory_size = self._infer_memory_size(len(chromosome))
        expected_length = 4 ** self.memory_size
        if len(chromosome) != expected_length:
            # Use self.memory_size, not an undefined variable
            raise ValueError(
                f"Chromosome length {len(chromosome)} does not match expected size {expected_length} "
                f"for memory {self.memory_size}"
            )

        # Convert each bit to 'C' or 'D'
        self.lookup_table = ['C' if bit == 0 else 'D' for bit in chromosome]

        # Build mapping from each possible “m‐round history” → index in [0..4**m - 1]
        self.history_to_index = self._generate_history_index_map(self.memory_size)

    def _infer_memory_size(self, chromosome_length):
        m = math.log(chromosome_length, 4)
        if not m.is_integer():
            raise ValueError(
                f"Chromosome length {chromosome_length} is not a power of 4; cannot infer memory size."
            )
        return int(m)

    def _generate_history_index_map(self, m):
        """
        Build a dict that maps each “last‐m‐round history” to a unique index in [0..4**m - 1].

        A single‐round outcome is one of these 4 tuples: ('C','C'), ('C','D'), ('D','C'), ('D','D').
        An “m‐round history” is therefore a LENGTH‐m tuple of those joint‐action tuples.
        """

        # All four possible single‐round joint outcomes:
        single_round_outcomes = list(product(['C', 'D'], repeat=2))
        # e.g. [('C','C'), ('C','D'), ('D','C'), ('D','D')]

        # Now take the Cartesian product of that list with itself m times:
        all_histories = list(product(single_round_outcomes, repeat=m))
        # If m=1, all_histories = [ (('C','C'),), (('C','D'),), (('D','C'),), (('D','D'),) ]
        # If m=2, all_histories has 16 entries like:
        #    ((('C','C'),('C','C')), (('C','C'),('C','D')), …, (('D','D'),('D','D'))) etc.

        mapping = {}
        for idx, history in enumerate(all_histories):
            mapping[history] = idx

        return mapping

    def next_move(self, last_state, state_matrix):
        """
        last_state: should represent “the last m rounds’ joint outcomes.”  Concretely,
                    each round’s joint outcome is a 2‐tuple like ('C','D').
                    So for memory_size = m:
                      • if m=1, last_state should be something that looks like ('C','D'),
                        but we will automatically wrap it into (('C','D'),) under the hood.
                      • if m=2, last_state should be a 2‐tuple of 2‐tuples, e.g.
                        (('D','C'), ('C','D')).  We do not re-wrap in that case.

        state_matrix: unused here (present only for interface compatibility).

        Returns 'C' or 'D' according to the lookup table.
        """

        # If memory_size == 1 but the user passed last_state as a flat 2‐tuple (e.g. ('C','C')),
        # wrap it into a single-element tuple so it matches our mapping’s key format:
        if self.memory_size == 1 and not isinstance(last_state[0], tuple):
            key = (last_state,)
        else:
            key = last_state

        idx = self.history_to_index[key]
        return self.lookup_table[idx]

    def move_probabilities(self, last_state, state_matrix):
        """
        Deterministic:  either 100% 'C' or 100% 'D' based on next_move.
        """
        move = self.next_move(last_state, state_matrix)
        return {
            "C": 1.0 if move == "C" else 0.0,
            "D": 1.0 if move == "D" else 0.0
        }
