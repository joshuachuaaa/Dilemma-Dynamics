from gameplayutils import *
import random

class Markov_Game:
    def __init__(self, strat1, strat2, rounds=100, error=0.0, initial_state='CC'):
        self.strat1 = strat1
        self.strat2 = strat2
        self.rounds = rounds
        self.error = error
        self.strat1Score = 0
        self.strat2Score = 0

        # Build the transition matrix once
        self.transition_matrix = build_transition_matrix(strat1, strat2, error)

        # Set initial state distribution (1 for initial_state, 0 elsewhere)
        self.initial_distribution = np.zeros(len(states))
        initial_index = states.index(initial_state)
        self.initial_distribution[initial_index] = 1.0

    def run(self):
        p_t = np.copy(self.initial_distribution)
        total_p1 = 0
        total_p2 = 0

        for _ in range(self.rounds):
            # FIRST: update state distribution
            p_t = p_t @ self.transition_matrix

            # THEN: calculate expected payoff from the *new* distribution
            for i, state in enumerate(states):
                payoff1, payoff2 = payoff_matrix[state]
                total_p1 += p_t[i] * payoff1
                total_p2 += p_t[i] * payoff2

        # Save scores to the object
        self.strat1Score = total_p1
        self.strat2Score = total_p2

        return total_p1, total_p2, p_t

    def printResults(self):
        print(f"{self.strat1.name} expected score: {self.strat1Score:.2f}")
        print(f"{self.strat2.name} expected score: {self.strat2Score:.2f}")


class MonteCarloGame:
    def __init__(self, strat1, strat2, rounds=100, error=0.0, initial_state='CC', trials=10000):
        self.strat1 = strat1
        self.strat2 = strat2
        self.rounds = rounds
        self.error = error
        self.initial_state = initial_state
        self.trials = trials

    def run(self):
        total_p1 = total_p2 = 0

        for _ in range(self.trials):
            state = self.initial_state
            score_p1 = score_p2 = 0

            for _ in range(self.rounds):
                move1 = self.strat1.next_move(state, state_to_last_moves)
                move2 = self.strat2.next_move(state, state_to_last_moves)

                if random.random() < self.error:
                    move1 = 'D' if move1 == 'C' else 'C'
                if random.random() < self.error:
                    move2 = 'D' if move2 == 'C' else 'C'

                outcome = move1 + move2
                payoff1, payoff2 = payoff_matrix[outcome]
                score_p1 += payoff1
                score_p2 += payoff2

                state = outcome

            total_p1 += score_p1
            total_p2 += score_p2

        avg_p1 = total_p1 / self.trials
        avg_p2 = total_p2 / self.trials

        return avg_p1, avg_p2
