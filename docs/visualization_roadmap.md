# Visualization and Interactive Simulation Roadmap

`Dilemma Dynamics` already produces static Matplotlib figures. The next useful
step is an interactive research interface that makes strategy behavior,
transition dynamics, noise, and evolutionary takeover easier to inspect without
turning the simulation core into GUI code.

## Design Principle

Keep the architecture layered:

```text
Strategies / Game / Markov
        ↓
Experiments
        ↓
Analysis and visualization
        ↓
Optional GUI
```

The GUI should call stable experiment functions and render results. It should
not own game rules, payoff logic, strategy behavior, or mutation logic.

## Candidate Interfaces

### 1. Research Dashboard

Best fit: Streamlit or Panel.

Purpose:
- Select two strategies and compare Markov vs Monte-Carlo payoffs.
- Adjust rounds, trials, and trembling-hand error.
- Show payoff matrix, final Markov distribution, and score trajectories.
- Run a small tournament and display rankings/heatmaps.

Why it fits:
- Low engineering overhead.
- Works naturally with Pandas and Matplotlib.
- Good for independent-study/research demonstration.

### 2. Pairwise Simulation Viewer

Purpose:
- Animate one matchup round by round.
- Show each player's last memory window.
- Display current moves, cumulative payoff, and error flips.
- Explain how a selected strategy made its decision.

Architecture requirement:
- Add a non-random trace mode to `MonteCarloGame` or a new simulator function
  that returns per-round records instead of only aggregate scores.

### 3. Evolutionary Takeover Explorer

Purpose:
- Run a reduced population/generation experiment interactively.
- Plot cooperative share, mean fitness, mutation events, and fixation status.
- Let users compare truncation vs proportional selection.

Architecture requirement:
- Have `genetic.py` expose structured experiment records without plotting as a
  side effect. The current `run_experiment()` function is a good start.

## Recommended First GUI Milestone

Build a Streamlit dashboard with three tabs:

1. Pairwise Matchup
2. Tournament Analysis
3. Evolutionary Dynamics

Minimum viable controls:
- Strategy A and Strategy B selectors.
- Engine selector: Markov or Monte-Carlo.
- Rounds slider.
- Trials slider for Monte-Carlo.
- Error-rate slider.
- Run button.

Minimum viable outputs:
- Score summary.
- Payoff trajectory or expected payoff table.
- Tournament heatmap.
- Cooperation trajectory for evolutionary runs.

## Data Needed for Better Visuals

Current engines mostly return aggregate scores. Better interactive visuals need
structured trace data:

```text
round
state_before
intended_move_1
intended_move_2
actual_move_1
actual_move_2
error_flip_1
error_flip_2
payoff_1
payoff_2
state_after
```

Adding this as a separate trace function keeps existing batch experiments fast
and unchanged.

## Implementation Order

1. Add pure trace helpers for pairwise simulations.
2. Move plotting functions into an `Analysis/` module.
3. Add a minimal Streamlit app in `apps/strategy_dashboard.py`.
4. Add tests for trace output shape and payoff consistency.
5. Add screenshots or GIFs to the README once the dashboard exists.

## Technologies

Recommended:
- Streamlit for fastest research UI.
- Plotly for interactive charts if Matplotlib becomes limiting.

Avoid initially:
- A custom desktop GUI.
- A large web frontend.
- Putting visualization state inside `Game/` or `Strategies/`.
