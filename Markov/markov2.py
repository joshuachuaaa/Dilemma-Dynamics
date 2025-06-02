import numpy as np
from gamestates import state_to_last_moves, states, state_to_last_moves_reversed
# ============================
# Transition Matrix Logic for memory-2
# ============================
import numpy as np
from gamestates import state_to_last_moves, states  # same gamestates; states = ["CC","CD","DC","DD"]

# Build all ordered pairs of memory-1 states. E.g. [ ("CC","CC"), ("CC","CD"), …, ("DD","DD") ]
memory2_states = [(prev, last) for prev in states for last in states]
state_index = {memory2_states[i]: i for i in range(len(memory2_states))}

def build_transition_matrix(strat1, strat2, error=0.0):
    size = len(memory2_states)  # 16
    matrix = np.zeros((size, size))

    for i, (prev, last) in enumerate(memory2_states):
        # Ask each strategy for their probabilistic move given the last two‐rounds of history
        p1_probs = strat1.move_probabilities([prev, last], state_to_last_moves)
        p2_probs = strat2.move_probabilities([prev, last], state_to_last_moves_reversed)

        # Flip moves with probability=error
        final_p1 = {
            "C": (1-error)*p1_probs["C"] + error*p1_probs["D"],
            "D": (1-error)*p1_probs["D"] + error*p1_probs["C"],
        }
        final_p2 = {
            "C": (1-error)*p2_probs["C"] + error*p2_probs["D"],
            "D": (1-error)*p2_probs["D"] + error*p2_probs["C"],
        }

        # For each actual outcome (p1_actual + p2_actual), figure out the “next” memory‐2 state
        for p1_actual in ["C","D"]:
            for p2_actual in ["C","D"]:
                outcome = p1_actual + p2_actual
                next_state = (last, outcome)  # shift out “prev” and append “outcome” as the new “last”
                j = state_index[next_state]   # index in the 16 states
                matrix[i, j] += final_p1[p1_actual] * final_p2[p2_actual]

    return matrix, memory2_states
