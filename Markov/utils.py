# ──────────────────────────────────────────────────────────
# Author: Joshua Chua Han Wei – 32781555
# File: Markov/utils.py
# Purpose: Small helper(s) for state-to-move mapping (e.g. flipped view).
# ──────────────────────────────────────────────────────────
def flipped_state_to_last_moves(st, state_to_last_moves):
            a, b = state_to_last_moves(st)
            return (b, a)