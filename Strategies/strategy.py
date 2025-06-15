from itertools import product
from gamestates import states, state_to_last_moves

class Strategy:
    def __init__(self):
        self.name = "BaseStrategy"
        self.is_nice = True
        self.memory_size = 1
        
    def next_move(self, last_state, state_matrix):
        raise NotImplementedError("You must implement next_move() in the subclass.")

    def move_probabilities(self, last_state, state_matrix):
        raise NotImplementedError("You must implement move_probabilities() in the subclass.")

    def reset(self):
        pass

    def to_bitstring(self):
        """
        Enumerate all 4**m possible histories of length m (using gamestates.states)
        and call next_move() on each to build a 0/1 bitstring.  Each history is
        reset independently, so strategies with internal flags (like GrimTrigger)
        don’t carry state from one history to the next.
        """
        m = self.memory_size
        if m < 1:
            raise ValueError(f"{self.name}: memory_size must be ≥ 1 to extract a bitstring.")

        # All length-m tuples of the state‐strings, e.g. ("CC","DD") for m=2
        all_histories = list(product(states, repeat=m))

        bits = []
        for history in all_histories:
            # Reset transient state (e.g. GrimTrigger) before each call
            try:
                self.reset()
            except TypeError:
                pass

            move = self.next_move(history, state_to_last_moves)
            bits.append(0 if move == 'C' else 1)

        return bits
