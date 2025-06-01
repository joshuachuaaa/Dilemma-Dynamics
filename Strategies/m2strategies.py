# ============================
# Strategy Implementations (Memory-2)
# ============================
from Strategies.strategy import Strategy

class TitForTwoTats(Strategy):
    def __init__(self):
        self.name = "TitForTwoTats"
        self.is_nice = True
        self.memory_size = 2

    def next_move(self, last_state, state_matrix):
        # Always slice the last two rounds
        second_last, last_round = last_state[-2:]
        _, opp_last = state_matrix[last_round]
        _, opp_prev = state_matrix[second_last]

        if opp_last == "D" and opp_prev == "D":
            return "D"
        else:
            return "C"

    def move_probabilities(self, last_state, state_matrix):
        intended = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if intended == "C" else 0.0, "D": 1.0 if intended == "D" else 0.0}

class ClearGrudger(Strategy):
    def __init__(self):
        self.name = "ClearGrudger"
        self.memory_size = 2
        self.is_nice = False  # since it punishes after any single D

    def next_move(self, last_state, state_matrix):
        prev, last = last_state[-2], last_state[-1]
        _, opp_prev = state_matrix[prev]
        _, opp_last = state_matrix[last]

        # If opponent cooperated in both rounds, cooperate; else punish:
        if opp_prev == "C" and opp_last == "C":
            return "C"
        else:
            return "D"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0, "D": 1.0 if m=="D" else 0.0}

class Pavlov2(Strategy):
    def __init__(self):
        self.name = "Pavlov2"
        self.memory_size = 2
        self.is_nice = True  # it still rewards mutual cooperation

    def next_move(self, last_state, state_matrix):
        prev, last = last_state[-2], last_state[-1]
        my_prev, opp_prev = state_matrix[prev]
        my_last, opp_last = state_matrix[last]

        # Case A: last two rounds were both mutual cooperation
        if (my_prev == "C" and opp_prev == "C") and (my_last == "C" and opp_last == "C"):
            return "C"

        # Case B: last two rounds were both mutual defection
        if (my_prev == "D" and opp_prev == "D") and (my_last == "D" and opp_last == "D"):
            return "D"

        # Otherwise, shift away from your own last move
        return "C" if (my_last == "D") else "D"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0, "D": 1.0 if m=="D" else 0.0}

class GenerousTwoTitForTwo(Strategy):
    def __init__(self):
        self.name = "G2T2T"
        self.memory_size = 2
        self.is_nice = True

    def next_move(self, last_state, state_matrix):
        prev, last = last_state[-2], last_state[-1]
        _, opp_prev = state_matrix[prev]
        _, opp_last = state_matrix[last]

        # If opponent cooperated at least once in those two rounds
        if opp_prev == "C" or opp_last == "C":
            return "C"
        else:
            return "D"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0, "D": 1.0 if m=="D" else 0.0}

class SuspiciousTf2T(Strategy):
    def __init__(self):
        self.name = "SuspiciousTf2T"
        self.memory_size = 2
        self.is_nice = False

    def next_move(self, last_state, state_matrix):
        prev, last = last_state[-2], last_state[-1]
        if prev == "CC" and last == "CC":
            # both were CC in rounds t-2 and t-1 → launch a D probe
            return "D"
        # otherwise, revert to normal Tf2T:
        _, opp_prev = state_matrix[prev]
        _, opp_last = state_matrix[last]
        if opp_prev == "D" and opp_last == "D":
            return "D"
        else:
            return "C"

    def move_probabilities(self, last_state, state_matrix):
        m = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if m=="C" else 0.0, "D": 1.0 if m=="D" else 0.0}




class Prober(Strategy):
    def __init__(self):
        self.name = "Prober"
        self.is_nice = False
        self.memory_size = 2
        self.initial_moves = ["D", "C", "C"]
        self.round_counter = 0
        self.detected_punishment = False

    def next_move(self, last_state, state_matrix):
        if self.round_counter < 3:
            move = self.initial_moves[self.round_counter]
            self.round_counter += 1
            return move
        else:
            # Always slice the last two rounds
            second_last, last_round = last_state[-2:]
            _, opp_last = state_matrix[last_round]

            if opp_last == "D":
                self.detected_punishment = True

            if self.detected_punishment:
                return opp_last  # Tit-for-Tat behavior
            else:
                return "D"  # exploit

    def move_probabilities(self, last_state, state_matrix):
        intended = self.next_move(last_state, state_matrix)
        return {"C": 1.0 if intended == "C" else 0.0, "D": 1.0 if intended == "D" else 0.0}


