# ──────────────────────────────────────────────────────────
# Author: Joshua Chua Han Wei – 32781555
# File: gamestates.py
# Purpose: Global definitions of outcome strings and helper lookup tables.
# ──────────────────────────────────────────────────────────

states = ["CC", "CD", "DC", "DD"]
state_to_last_moves = {
    "CC": ("C", "C"),
    "CD": ("C", "D"),
    "DC": ("D", "C"),
    "DD": ("D", "D"),
}
state_to_last_moves_reversed = {
    "CC": ("C", "C"),
    "CD": ("D", "C"),
    "DC": ("C", "D"),
    "DD": ("D", "D"),
}