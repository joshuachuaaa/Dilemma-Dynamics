# ============================
# Strategy Base Class
# ============================
class Strategy:
    def __init__(self):
        self.name = 'Base'
        
    def next_move(self, last_state, state_matrix):
        raise NotImplementedError("You must implement next_move() in the subclass.")

    def move_probabilities(self, last_state, state_matrix):
        """
        Returns a dictionary: {"C": prob_of_cooperate, "D": prob_of_defect}
        Deterministic strategies should return 1.0/0.0.
        Probabilistic strategies should reflect mixed probabilities.
        """
        raise NotImplementedError("You must implement move_probabilities() in the subclass.")