class Strategy:
    def __init__(self):
        self.name = "BaseStrategy"
        self.is_nice = True   # tracks niceness
        self.memory_size = 1  # default, override in subclass
        
    def next_move(self, last_state, state_matrix):
        raise NotImplementedError("You must implement next_move() in the subclass.")

    def move_probabilities(self, last_state, state_matrix):

        raise NotImplementedError("You must implement move_probabilities() in the subclass.")

    def reset(self):
        pass
