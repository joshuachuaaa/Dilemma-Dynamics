# Markov/markov3.py

import numpy as np
from gamestates import state_to_last_moves, states, state_to_last_moves_reversed
#   - `states` here is ["CC","CD","DC","DD"] (the memory-1 outcome set)
#   - `state_to_last_moves` maps e.g. "CD" -> ("C","D")

# Build the list of all 64 possible (t-3, t-2, t-1) tuples:
memory3_states = [
    (a, b, c)
    for a in states
    for b in states
    for c in states
]

# Create a quick lookup from each triple to its index in the 64-list:
state_index_3 = { memory3_states[i]: i for i in range(len(memory3_states)) }

def build_transition_matrix(strat1, strat2, error=0.0):
    """
    Returns:
      - A 64×64 numpy array `M` where M[i,j] = P(next_state = memory3_states[j] | current_state = memory3_states[i])
      - The list `memory3_states` (length 64) so that
            M[i] corresponds to transitions out of memory3_states[i].

    Each strategy must have memory_size == 3 (or else this function will still be called,
    but strat1/strat2.move_probabilities(...) should only honor the last three).
    """
    size = len(memory3_states)  # 64
    matrix = np.zeros((size, size))

    # For each “current” triple (t-3, t-2, t-1):
    for i, (s_tm3, s_tm2, s_tm1) in enumerate(memory3_states):
        # 1) ask each player what they would do after seeing those three outcomes
        #    (the strategy’s move_probabilities only looks at those three because memory_size=3)
        p1_probs = strat1.move_probabilities([s_tm3, s_tm2, s_tm1], state_to_last_moves)
        p2_probs = strat2.move_probabilities([s_tm3, s_tm2, s_tm1], state_to_last_moves_reversed)

        # 2) incorporate “trembling‐hand” error: flip C↔D with probability=error
        final_p1 = {
            "C": (1 - error) * p1_probs["C"] + error * p1_probs["D"],
            "D": (1 - error) * p1_probs["D"] + error * p1_probs["C"],
        }
        final_p2 = {
            "C": (1 - error) * p2_probs["C"] + error * p2_probs["D"],
            "D": (1 - error) * p2_probs["D"] + error * p2_probs["C"],
        }

        # 3) compute joint probabilities for each possible outcome ("CC","CD","DC","DD")
        #    and shift the triple window forward by one
        for move1_actual in ["C", "D"]:
            for move2_actual in ["C", "D"]:
                outcome = move1_actual + move2_actual
                prob = final_p1[move1_actual] * final_p2[move2_actual]

                # The new triple (t-2, t-1, t) becomes (s_tm2, s_tm1, outcome)
                next_trip = (s_tm2, s_tm1, outcome)
                j = state_index_3[next_trip]
                matrix[i, j] += prob

    return matrix, memory3_states