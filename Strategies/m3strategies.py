from Strategies.strategy import Strategy
# Total Number of m-3 Strategies = 6


class TitForThreeTats(Strategy):
    def __init__(self):
        self.name = "TitForThreeTats"
        self.memory_size = 3
        self.is_nice = True

    def next_move(self, last_state, state_matrix):
        a, b, c = last_state[-3:]
        # get opponent’s moves:
        _, opp_a = state_matrix[a]
        _, opp_b = state_matrix[b]
        _, opp_c = state_matrix[c]

        if opp_a == "D" and opp_b == "D" and opp_c == "D":
            return "D"
        else:
            return "C"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0,
                "D": 1.0 if m=="D" else 0.0}


class TwoForgiveOnePunish(Strategy):
    def __init__(self):
        self.name = "TwoForgiveOnePunish"
        self.memory_size = 3
        self.is_nice = True

    def next_move(self, last_state, state_matrix):
        a, b, c = last_state[-3:]
        _, opp_a = state_matrix[a]
        _, opp_b = state_matrix[b]
        _, opp_c = state_matrix[c]

        defect_count = (opp_a == "D") + (opp_b == "D") + (opp_c == "D")
        if defect_count >= 2:
            return "D"
        else:
            return "C"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0,
                "D": 1.0 if m=="D" else 0.0}


class ThreeGrudger(Strategy):
    def __init__(self):
        self.name = "ThreeGrudger"
        self.memory_size = 3
        self.is_nice = False

    def next_move(self, last_state, state_matrix):
        a, b, c = last_state[-3:]
        _, opp_a = state_matrix[a]
        _, opp_b = state_matrix[b]
        _, opp_c = state_matrix[c]

        if (opp_a == "C") and (opp_b == "C") and (opp_c == "C"):
            return "C"
        else:
            return "D"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0,
                "D": 1.0 if m=="D" else 0.0}

class PatternFollower3(Strategy):
    def __init__(self):
        self.name = "PatternFollower3"
        self.memory_size = 3
        self.is_nice = False

    def next_move(self, last_state, state_matrix):
        a, b, c = last_state[-3:]
        _, opp_a = state_matrix[a]
        _, opp_b = state_matrix[b]
        _, opp_c = state_matrix[c]

        if opp_a == "D" and opp_b == "C" and opp_c == "D":
            return "D"
        else:
            return "C"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0,
                "D": 1.0 if m=="D" else 0.0}


class Pavlov3(Strategy):
    def __init__(self):
        self.name = "Pavlov3"
        self.memory_size = 3
        self.is_nice = True

    def next_move(self, last_state, state_matrix):
        a, b, c = last_state[-3:]
        my_a, opp_a = state_matrix[a]
        my_b, opp_b = state_matrix[b]
        my_c, opp_c = state_matrix[c]

        # All three rounds were (C,C)?  Stay “C.”
        if my_a == "C" and opp_a == "C" and my_b == "C" and opp_b == "C" and my_c == "C" and opp_c == "C":
            return "C"

        # All three were (D,D)?  Stay “D.”
        if my_a == "D" and opp_a == "D" and my_b == "D" and opp_b == "D" and my_c == "D" and opp_c == "D":
            return "D"

        # Otherwise, shift away from your last move
        return "C" if my_c == "D" else "D"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0,
                "D": 1.0 if m=="D" else 0.0}


class Generous3(Strategy):
    def __init__(self):
        self.name = "Generous3"
        self.memory_size = 3
        self.is_nice = True

    def next_move(self, last_state, state_matrix):
        a, b, c = last_state[-3:]
        _, opp_a = state_matrix[a]
        _, opp_b = state_matrix[b]
        _, opp_c = state_matrix[c]

        coop_count = (opp_a == "C") + (opp_b == "C") + (opp_c == "C")
        if coop_count >= 2:
            return "C"
        else:
            return "D"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0,
                "D": 1.0 if m=="D" else 0.0}
