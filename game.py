import numpy as np
import random
from gamestates import state_to_last_moves
from payoff_matrix import payoff_matrix

class MarkovGame:
    def __init__(self, strat1, strat2, rounds=50, error=0.0, initial_state=None):
        self.strat1 = strat1
        self.strat2 = strat2
        self.rounds = rounds
        self.error = error
        self.strat1Score = 0.0
        self.strat2Score = 0.0

        # Determine max memory size between the two strategies (minimum 1)
        self.max_memory = max(1, strat1.memory_size, strat2.memory_size)

        # Dynamically select the correct matrix builder
        if self.max_memory == 1:
            from Markov.markov1 import build_transition_matrix
        elif self.max_memory == 2:
            from Markov.markov2 import build_transition_matrix
        elif self.max_memory == 3:
            from Markov.markov3 import build_transition_matrix
        else:
            raise ValueError(f"Unsupported memory size: {self.max_memory}")

        # Build transition matrix and state list
        self.transition_matrix, self.states = build_transition_matrix(strat1, strat2, error)

        #
        # Patched initial_state handling:
        #
        if initial_state is None:
            if self.max_memory == 1:
                initial_state = "CC"
            else:
                initial_state = tuple(["CC"] * self.max_memory)

        elif isinstance(initial_state, (list, np.ndarray)):
            # Allow list or NumPy array input; convert to tuple
            initial_state = tuple(initial_state)

        elif isinstance(initial_state, str):
            if self.max_memory == 1:
                initial_state = initial_state
            else:
                # Replicate a single string into a max_memory‐tuple, e.g. "CC" → ("CC","CC")
                initial_state = tuple([initial_state] * self.max_memory)

        # Now validate and build the initial_distribution
        if self.max_memory == 1:
            # For memory‐1, states is ["CC","CD","DC","DD"]
            if not isinstance(initial_state, str):
                raise ValueError(
                    f"Memory-1 game requires initial_state to be a string, e.g. 'CC'. "
                    f"You passed {initial_state!r}."
                )
            try:
                idx = self.states.index(initial_state)
            except ValueError:
                raise ValueError(f"Initial state {initial_state!r} not found in 1-memory state list.")
            self.initial_distribution = np.zeros(len(self.states))
            self.initial_distribution[idx] = 1.0

        else:
            # For memory-2 (or higher), states is a list of tuples of length max_memory
            if not (isinstance(initial_state, tuple) and len(initial_state) == self.max_memory):
                raise ValueError(
                    f"Memory-{self.max_memory} game requires initial_state to be a tuple of length {self.max_memory}. "
                    f"You passed {initial_state!r}."
                )
            try:
                idx = self.states.index(initial_state)
            except ValueError:
                raise ValueError(f"Initial state {initial_state!r} not found in {self.max_memory}-memory state list.")
            self.initial_distribution = np.zeros(len(self.states))
            self.initial_distribution[idx] = 1.0

    def run(self):
        p_t = self.initial_distribution.copy()
        total1 = 0.0
        total2 = 0.0

        # Iterate over the specified number of rounds
        for _ in range(self.rounds):
            # Advance the distribution by one step
            p_t = p_t @ self.transition_matrix

            # Accumulate expected payoff based on the new distribution
            for i, state in enumerate(self.states):
                # In memory-1, state is a string like "CC"; in memory-2, state is a tuple like ("CC","DC")
                last_round = state if isinstance(state, str) else state[-1]
                payoff1, payoff2 = payoff_matrix[last_round]
                total1 += p_t[i] * payoff1
                total2 += p_t[i] * payoff2

        self.strat1Score = total1
        self.strat2Score = total2
        return total1, total2, p_t

    def printResults(self):
        print(f"{self.strat1.name} expected score: {self.strat1Score:.2f}")
        print(f"{self.strat2.name} expected score: {self.strat2Score:.2f}")


class MonteCarloGame:
    def __init__(self, strat1, strat2, rounds=50, error=0.0, initial_state='CC', trials=10000):
        self.strat1 = strat1
        self.strat2 = strat2
        self.rounds = rounds
        self.error = error
        self.initial_state = initial_state
        self.trials = trials

        # Determine max memory size (minimum 1)
        self.max_memory = max(1, strat1.memory_size, strat2.memory_size)

    def run(self):
        total_p1 = 0.0
        total_p2 = 0.0

        for _ in range(self.trials):
            # Initialize the padded history to the initial_state, repeated max_memory times
            state = tuple([self.initial_state] * self.max_memory)
            score_p1 = 0.0
            score_p2 = 0.0

            for _ in range(self.rounds):
                move1 = self.strat1.next_move(state, state_to_last_moves)
                move2 = self.strat2.next_move(state, state_to_last_moves)

                # Apply error flips
                if random.random() < self.error:
                    move1 = 'D' if move1 == 'C' else 'C'
                if random.random() < self.error:
                    move2 = 'D' if move2 == 'C' else 'C'

                outcome = move1 + move2
                payoff1, payoff2 = payoff_matrix[outcome]
                score_p1 += payoff1
                score_p2 += payoff2

                # Update history: keep only last max_memory rounds
                if self.max_memory > 1:
                    state = (*state[1:], outcome)
                else:
                    state = (outcome,)

            total_p1 += score_p1
            total_p2 += score_p2

        avg_p1 = total_p1 / self.trials
        avg_p2 = total_p2 / self.trials

        return avg_p1, avg_p2
