# Visualization and Interactive Simulation Roadmap

`Dilemma Dynamics` already produces static Matplotlib figures for its
game-theoretic simulations. The next useful step is an interactive research
interface that makes strategy behavior, transition dynamics, noise, and
evolutionary takeover easier to inspect without turning the simulation core into
GUI code.

## Design Principle

Keep the architecture layered:

```text
Strategies / Game / Markov
        v
Experiments
        v
Analysis and visualization
        v
Optional GUI
```

The GUI should call stable experiment functions and render results. It should
not own game rules, payoff logic, strategy behavior, or mutation logic.

## Candidate Interfaces

### 0. Terminal Simulation Trace

Best fit: standard-library command-line/TUI workflow.

Purpose:
- Inspect a single matchup round by round.
- Show the current memory window, intended moves, actual moves, error flips,
  per-round payoff, and cumulative score.
- Provide a fast, dependency-free way to explain what the simulation is doing.

Status:
- `apps/strategy_tui.py` provides this first visualization layer.

Limit:
- This is enough for understanding a single simulated game, but not enough for
  comparing many strategies, viewing heatmaps, or exploring evolutionary runs.
  Those still benefit from Matplotlib figures or a GUI dashboard.

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

### 1a. Lightweight Desktop GUI

Best fit: Tkinter.

Purpose:
- Run one pairwise matchup from a small desktop window.
- Select strategies, rounds, error rate, starting state, and seed.
- Show final score, cooperation counts, a round table, and a simple score chart.

Status:
- `apps/simulation_gui.py` provides this lightweight GUI without adding a new
  dependency.

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
