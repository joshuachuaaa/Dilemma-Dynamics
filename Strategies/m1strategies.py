from Strategies.strategy import Strategy
# ============================
# Strategy Implementations (Memory-1)
# ============================

class TitForTat(Strategy):
    def __init__(self):
        self.name = "TitForTat"
        self.is_nice = True          # never defects first
        self.memory_size = 1         # looks at opponent's last move

    def next_move(self, last_state, state_matrix):
        _, opponent_last = state_matrix[last_state]
        return opponent_last

    def move_probabilities(self, last_state, state_matrix):
        intended = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if intended == "C" else 0.0, "D": 1.0 if intended == "D" else 0.0}

class WinStayLoseShift(Strategy):
    def __init__(self):
        self.name = "WinStayLoseShift"
        self.is_nice = True          # cooperates after mutual cooperation, only switches if mismatched
        self.memory_size = 1         # looks at last round's outcome

    def next_move(self, last_state, state_matrix):
        my_last, opp_last = state_matrix[last_state]
        if my_last == opp_last:
            return my_last  # win → stay
        else:
            return "C" if my_last == "D" else "D"  # lose → shift

    def move_probabilities(self, last_state, state_matrix):
        intended = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if intended == "C" else 0.0, "D": 1.0 if intended == "D" else 0.0}


class ReverseTitForTat(Strategy):
    def __init__(self):
        self.name = "ReverseTitForTat"
        self.is_nice = False         # defects against cooperation, rewards defection
        self.memory_size = 1         # looks at opponent's last move

    def next_move(self, last_state, state_matrix):
        _, opponent_last = state_matrix[last_state]
        return "D" if opponent_last == "C" else "C"

    def move_probabilities(self, last_state, state_matrix):
        intended = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if intended == "C" else 0.0, "D": 1.0 if intended == "D" else 0.0}
