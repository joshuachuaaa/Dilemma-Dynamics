import numpy as np
# Define Points (payoff matrix)
payoff_matrix = {
    'CC': (3, 3),
    'CD': (0, 5),
    'DC': (5, 0),
    'DD': (1, 1),
}

# Define states and state-to-last-moves map
states = ["CC", "CD", "DC", "DD"]
state_to_last_moves = {
    "CC": ("C", "C"),
    "CD": ("C", "D"),
    "DC": ("D", "C"),
    "DD": ("D", "D"),
}
# ============================
# Transition Matrix Logic
# ============================
def build_transition_probs(current_state, strat1, strat2, error=0.0):
    # Get move probabilities for player 1 and player 2
    p1_probs = strat1.move_probabilities(current_state, state_to_last_moves)
    p2_probs = strat2.move_probabilities(current_state, state_to_last_moves)

    # Incorporate error: flip intended moves with error probability
    final_p1_probs = {
        "C": (1 - error) * p1_probs["C"] + error * p1_probs["D"],
        "D": (1 - error) * p1_probs["D"] + error * p1_probs["C"],
    }
    final_p2_probs = {
        "C": (1 - error) * p2_probs["C"] + error * p2_probs["D"],
        "D": (1 - error) * p2_probs["D"] + error * p2_probs["C"],
    }

    # Compute joint probabilities over all combinations
    probs = {state: 0.0 for state in states}
    for p1_actual in ["C", "D"]:
        for p2_actual in ["C", "D"]:
            outcome = p1_actual + p2_actual
            probs[outcome] += final_p1_probs[p1_actual] * final_p2_probs[p2_actual]

    return [probs[s] for s in states]


def build_transition_matrix(strat1, strat2, error=0.0):
    matrix = []
    for state in states:
        row = build_transition_probs(state, strat1, strat2, error)
        matrix.append(row)
    return np.array(matrix)

