import numpy as np
from gamestates import state_to_last_moves, states, state_to_last_moves_reversed
# ============================
# Transition Matrix Logic for memory-1
# ============================

import numpy as np
from gamestates import state_to_last_moves, states  # states = ["CC","CD","DC","DD"]

def build_transition_probs(current_state, strat1, strat2, error=0.0):
    # Get move probabilities for player 1 and player 2
    p1_probs = strat1.move_probabilities([current_state], state_to_last_moves)
    p2_probs = strat2.move_probabilities([current_state], state_to_last_moves_reversed)

    # Incorporate error (flip C↔D with probability=error)
    final_p1 = {
        "C": (1-error)*p1_probs["C"] + error*p1_probs["D"],
        "D": (1-error)*p1_probs["D"] + error*p1_probs["C"],
    }
    final_p2 = {
        "C": (1-error)*p2_probs["C"] + error*p2_probs["D"],
        "D": (1-error)*p2_probs["D"] + error*p2_probs["C"],
    }

    # Compute joint probabilities for the next‐round outcome
    probs = {s: 0.0 for s in states}
    for p1_actual in ["C","D"]:
        for p2_actual in ["C","D"]:
            outcome = p1_actual + p2_actual
            probs[outcome] += final_p1[p1_actual] * final_p2[p2_actual]

    return [probs[s] for s in states]

def build_transition_matrix(strat1, strat2, error=0.0):
    """
    Returns: (4×4 numpy array, states_list)
    where states_list == ["CC","CD","DC","DD"] in that order.
    """
    matrix = []
    for st in states:  # states = ["CC","CD","DC","DD"]
        row = build_transition_probs(st, strat1, strat2, error)
        matrix.append(row)
    return np.array(matrix), states
