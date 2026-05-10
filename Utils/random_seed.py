# ---------------------------------------------------------------------------
# File: Utils/random_seed.py
# Purpose: Shared reproducibility helper for stochastic simulations.
# ---------------------------------------------------------------------------

import random

import numpy as np


DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python and NumPy RNGs, then return the seed used."""
    random.seed(seed)
    np.random.seed(seed)
    return seed
