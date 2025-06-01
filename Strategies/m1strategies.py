from Strategies.strategy import Strategy
import random
__all__ = [
    'Strategy', 'TitForTat', 'AlwaysDefect', 'AlwaysCooperate', 'WinStayLoseShift',
    'ReverseTitForTat', 'RandomStrategy'
]

# ============================
# Strategy Implementations (Memory-1)
# ============================
class TitForTat(Strategy):
    def __init__(self):
        self.name = "TitForTat"
    def next_move(self, last_state, state_matrix):
        _, opponent_last = state_matrix[last_state]
        return opponent_last

    def move_probabilities(self, last_state, state_matrix):
        intended = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if intended == "C" else 0.0, "D": 1.0 if intended == "D" else 0.0}

class AlwaysDefect(Strategy):
    
    def __init__(self):
        self.name = "AlwaysDefect"
    def next_move(self, _, __):
        return "D"

    def move_probabilities(self, _, __):
        return {"C": 0.0, "D": 1.0}

class AlwaysCooperate(Strategy):
    
    def __init__(self):
        self.name = "AlwaysCooperate"

    def next_move(self, _, __):
        return "C"

    def move_probabilities(self, _, __):
        return {"C": 1.0, "D": 0.0}

class WinStayLoseShift(Strategy):
    
    def __init__(self):
        self.name = "WinStayLoseShift"

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

    def next_move(self, last_state, state_matrix):
        _, opponent_last = state_matrix[last_state]
        return "D" if opponent_last == "C" else "C"

    def move_probabilities(self, last_state, state_matrix):
        intended = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if intended == "C" else 0.0, "D": 1.0 if intended == "D" else 0.0}

class RandomStrategy(Strategy):

    def __init__(self, coop_prob=0.5):
        self.name = "RandomStrategy"
        self.coop_prob = coop_prob  # probability to cooperate

    def next_move(self, _, __):
        return "C" if random.random() < self.coop_prob else "D"

    def move_probabilities(self, _, __):
        return {"C": self.coop_prob, "D": 1 - self.coop_prob}
