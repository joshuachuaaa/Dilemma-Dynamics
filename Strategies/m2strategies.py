# ============================
# Strategy Implementations (Memory-2)
# ============================
from Strategies.strategy import Strategy



class TitForTwoTats(Strategy):
    def __init__(self):
        self.name = "TitForTwoTats"
        self.is_nice = True
        self.memory_size = 2

    def next_move(self, last_two_states, state_matrix):
        last_round, prev_round = last_two_states
        _, opp_last = state_matrix[last_round]
        _, opp_prev = state_matrix[prev_round]

        if opp_last == "D" and opp_prev == "D":
            return "D"
        else:
            return "C"

    def move_probabilities(self, last_two_states, state_matrix):
        intended = self.next_move(last_two_states, state_matrix)
        return {"C": 1.0 if intended == "C" else 0.0, "D": 1.0 if intended == "D" else 0.0}

