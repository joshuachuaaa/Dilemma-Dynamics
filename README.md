# Dilemma Dynamics

## A Computational Research Framework for Evolutionary Cooperation

**Author: Joshua Chua Han Wei**

---

`Dilemma Dynamics` studies strategy selection, memory, noise, and cooperative
takeover in the Iterated Prisoner's Dilemma. The project combines analytical
Markov-chain evaluation, Monte-Carlo simulation, chromosome-encoded strategies,
and evolutionary selection experiments into a reproducible local research
workflow.

## 1  Project layout

```text
project_root/
│
├── Game/              # deterministic & Monte-Carlo engines
├── Markov/            # transition-matrix builders (m = 1–3)
├── Strategies/        # hand-coded & chromosome strategies
├── Utils/             # payoff, state, plotting, and quick-check helpers
├── Test/              # unittest coverage for games and strategy encoding
│
├── genetic.py         # evolutionary cooperation experiment
├── tournamentLean.py  # noise-sensitivity experiment
├── tournament.py      # full round-robin analysis driver
├── requirements.txt   # pip dependencies
└── README.md          # ← you are here
```

> **Packages** – every folder contains an `__init__.py`, so dotted imports like `from Game.game import MarkovGame` work regardless of the working directory.

---

## 2  Quick-start — reproduce analysis figures

```bash
# 1) create a fresh environment (Python ≥ 3.10)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2) install dependencies
pip install -r requirements.txt

# 3) generate the main analysis plots
python genetic.py          # ≈ takes about an hour
python tournamentLean.py   # ≈ 5-10 minutes
```

Figures appear in Matplotlib windows **and** `./figures/`.

---

## 3  File-by-file guide

| File                | Purpose                                                                                                                                                                                          | Typical runtime |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| `genetic.py`        | 15 replicates × 8 variants (trunc/proportional × k = 1‑4). Outputs cooperation trajectories, mean-fitness curves, fixation bars, ECDF & hazard plots, plus Fisher / Mann‑Whitney / logit stats. | ≈ 75 s          |
| `tournamentLean.py` | Monte-Carlo sweep over ε ∈ {0 %, 5 %, 10 %}. Produces performance lines, memory-size box-plots, cooperative-vs-exploitative plots, global μ ± σ and class-gap shrinkage.                         | ≈ 60 s          |
| `tournament.py`     | Full round-robin analysis for strategy comparison; optional heat-maps and rankings.                                                                                                              | 20 – 120 s      |
| `Utils/test.py`     | Tiny sanity check for sample deterministic vs Monte-Carlo pay-offs.                                                                                                                              | < 2 s           |

---

## 4  Performance awareness

| Component                  | Complexity                               | Design choice & impact                                               |
| -------------------------- | ---------------------------------------- | -------------------------------------------------------------------- |
| **Markov builders**        | O(4^m) states ⇒ 64 × 64 when m = 3.      | Project caps m ≤ 3 to keep transition matrices tractable.            |
| **Monte-Carlo engine**     | O(trials × rounds); default 10 000 × 50. | Gives SE ≈ 0.03 pts; trials can be halved for quick tests.           |
| **Evolutionary simulator** | O(GEN × POP × (POP − 1)/2 × MC_cost).    | With GEN = 20, POP = 8 wall-time ≤ 80 s; larger POP tested via flag. |
| **Memory footprint**       | Peak RAM ≈ 130 MB (NumPy arrays).        | Comfortable for typical local analysis runs.                         |

---

## 5  Running your own matches

```python
from Game.game import MarkovGame, MonteCarloGame
from Strategies.m1strategies import TitForTat, WinStayLoseShift

# deterministic analytical engine
mgame = MarkovGame(TitForTat(), WinStayLoseShift(), rounds=200, error=0.03)
score_A, score_B, _ = mgame.run()
print(score_A, score_B)
```

To add a custom strategy, subclass `Strategies.strategy.Strategy` and implement:

- `next_move(history, state_matrix)` → returns `'C'` or `'D'`.
- `move_probabilities(history, state_matrix)` → returns `{"C": p_c, "D": p_d}`.

---

## 6  Verification hooks

- `genetic.py` runs `verification_test()` before the main experiment to check population invariance when μ = 0, k = 0.
- `Utils/test.py` reproduces quick deterministic vs Monte-Carlo pay-off checks in < 2 s.
- `Test/` contains unittest coverage for Markov dynamics, chromosome conversion, memory strategies, and result symmetry.

---

This framework is intended for reproducible computational experiments in
evolutionary game dynamics.
