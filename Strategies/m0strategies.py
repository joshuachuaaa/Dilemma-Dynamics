from Strategies.strategy import Strategy
import random

# ============================
# Strategy Implementations (Memory-0)
# ============================
# Total Number of m-0 Strategies = 3
class AlwaysDefect(Strategy):
    def __init__(self):
        self.name = "AlwaysDefect"
        self.is_nice = False         # always defects
        self.memory_size = 0         # no memory, fixed move

    def next_move(self, _, __):
        return "D"

    def move_probabilities(self, _, __):
        return {"C": 0.0, "D": 1.0}
    

class AlwaysCooperate(Strategy):
    def __init__(self):
        self.name = "AlwaysCooperate"
        self.is_nice = True          # always cooperates
        self.memory_size = 0         # no memory, fixed move

    def next_move(self, _, __):
        return "C"

    def move_probabilities(self, _, __):
        return {"C": 1.0, "D": 0.0}


class RandomStrategy(Strategy):
    def __init__(self, coop_prob=0.5):
        self.name = "RandomStrategy"
        self.is_nice = True if coop_prob >= 0.5 else False   # majority pure-cooperators are nice
        self.memory_size = 0                                 # ignores history, fixed probability
        self.coop_prob = coop_prob                           # probability to cooperate

    def next_move(self, _, __):
        return "C" if random.random() < self.coop_prob else "D"

    def move_probabilities(self, _, __):
        return {"C": self.coop_prob, "D": 1 - self.coop_prob}
