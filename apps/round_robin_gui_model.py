"""Pure data helpers for a live round-robin tournament GUI."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations

from Experiments.strategy_registry import available_strategy_names, create_strategy
from Game.game import MarkovGame, MonteCarloGame
from Utils.payoff_matrix import payoff_matrix


ENGINE_NAMES = ("markov", "montecarlo")


@dataclass(frozen=True)
class MatchResult:
    """One completed pairwise matchup from a round-robin run."""

    match_index: int
    total_matches: int
    strategy_a: str
    strategy_b: str
    score_a: float
    score_b: float
    engine_type: str
    rounds: int
    trials: int
    error: float


@dataclass(frozen=True)
class RankingRow:
    """Aggregate tournament position for one selected strategy."""

    rank: int
    strategy: str
    total_score: float
    matches_played: int
    wins: int
    losses: int
    ties: int
    average_score_per_match: float


@dataclass(frozen=True)
class RoundRobinSummary:
    """Completed round-robin output for a GUI to render."""

    strategy_names: tuple[str, ...]
    engine_type: str
    rounds: int
    trials: int
    error: float
    match_results: tuple[MatchResult, ...]
    rankings: tuple[RankingRow, ...]


def list_strategy_names() -> tuple[str, ...]:
    """Return names that can be shown in strategy selection controls."""
    return tuple(available_strategy_names())


def list_engine_names() -> tuple[str, ...]:
    """Return engine names accepted by the round-robin model."""
    return ENGINE_NAMES


def select_strategy_names(strategy_names: Iterable[str]) -> tuple[str, ...]:
    """Validate and freeze a GUI strategy selection in display order."""
    selected = tuple(strategy_names)
    if len(selected) < 2:
        raise ValueError("Select at least two strategies for a round-robin tournament.")

    valid_names = set(list_strategy_names())
    seen: set[str] = set()
    for name in selected:
        if name not in valid_names:
            valid = ", ".join(list_strategy_names())
            raise ValueError(
                f"Invalid strategy name {name!r}. Available strategies: {valid}"
            )
        if name in seen:
            raise ValueError(f"Duplicate strategy name {name!r} in selection.")
        seen.add(name)

    return selected


def iter_pairwise_match_results(
    strategy_names: Iterable[str],
    *,
    engine_type: str = "markov",
    rounds: int = 50,
    trials: int = 10000,
    error: float = 0.0,
    initial_state: str = "CC",
) -> Iterator[MatchResult]:
    """
    Yield one pairwise result at a time for a live round-robin display.

    Validation happens before the iterator is returned so GUI controls can show
    configuration errors without waiting for the first match to start.
    """
    selected = select_strategy_names(strategy_names)
    engine = _validate_engine_type(engine_type)
    _validate_run_config(
        rounds=rounds,
        trials=trials,
        error=error,
        initial_state=initial_state,
    )
    pairings = tuple(combinations(selected, 2))
    total_matches = len(pairings)

    def _results() -> Iterator[MatchResult]:
        for match_index, (strategy_a, strategy_b) in enumerate(pairings, start=1):
            score_a, score_b = _run_pairwise_match(
                strategy_a,
                strategy_b,
                engine_type=engine,
                rounds=rounds,
                trials=trials,
                error=error,
                initial_state=initial_state,
            )
            yield MatchResult(
                match_index=match_index,
                total_matches=total_matches,
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                score_a=score_a,
                score_b=score_b,
                engine_type=engine,
                rounds=rounds,
                trials=trials,
                error=error,
            )

    return _results()


def iter_round_robin_results(
    strategy_names: Iterable[str],
    *,
    engine_type: str = "markov",
    rounds: int = 50,
    trials: int = 10000,
    error: float = 0.0,
    initial_state: str = "CC",
) -> Iterator[MatchResult]:
    """Alias with tournament wording for callers outside the GUI."""
    return iter_pairwise_match_results(
        strategy_names,
        engine_type=engine_type,
        rounds=rounds,
        trials=trials,
        error=error,
        initial_state=initial_state,
    )


def build_rankings(
    strategy_names: Iterable[str],
    match_results: Iterable[MatchResult],
) -> tuple[RankingRow, ...]:
    """Build rankings from complete or partial match results."""
    selected = select_strategy_names(strategy_names)
    selected_set = set(selected)
    stats = {
        name: {
            "total_score": 0.0,
            "matches_played": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
        for name in selected
    }

    for result in match_results:
        if result.strategy_a not in selected_set:
            raise ValueError(
                f"Match result includes unselected strategy {result.strategy_a!r}."
            )
        if result.strategy_b not in selected_set:
            raise ValueError(
                f"Match result includes unselected strategy {result.strategy_b!r}."
            )

        stats[result.strategy_a]["total_score"] += result.score_a
        stats[result.strategy_b]["total_score"] += result.score_b
        stats[result.strategy_a]["matches_played"] += 1
        stats[result.strategy_b]["matches_played"] += 1

        if result.score_a > result.score_b:
            stats[result.strategy_a]["wins"] += 1
            stats[result.strategy_b]["losses"] += 1
        elif result.score_b > result.score_a:
            stats[result.strategy_b]["wins"] += 1
            stats[result.strategy_a]["losses"] += 1
        else:
            stats[result.strategy_a]["ties"] += 1
            stats[result.strategy_b]["ties"] += 1

    ordered = sorted(
        selected,
        key=lambda name: (
            -stats[name]["total_score"],
            name,
        ),
    )

    rankings: list[RankingRow] = []
    previous_score: float | None = None
    previous_rank = 0
    for index, name in enumerate(ordered, start=1):
        score = stats[name]["total_score"]
        rank = previous_rank if previous_score == score else index
        matches_played = int(stats[name]["matches_played"])
        rankings.append(
            RankingRow(
                rank=rank,
                strategy=name,
                total_score=score,
                matches_played=matches_played,
                wins=int(stats[name]["wins"]),
                losses=int(stats[name]["losses"]),
                ties=int(stats[name]["ties"]),
                average_score_per_match=(
                    score / matches_played if matches_played else 0.0
                ),
            )
        )
        previous_score = score
        previous_rank = rank

    return tuple(rankings)


def run_round_robin(
    strategy_names: Iterable[str],
    *,
    engine_type: str = "markov",
    rounds: int = 50,
    trials: int = 10000,
    error: float = 0.0,
    initial_state: str = "CC",
) -> RoundRobinSummary:
    """Run every matchup and return final GUI-friendly tournament data."""
    selected = select_strategy_names(strategy_names)
    engine = _validate_engine_type(engine_type)
    results = tuple(
        iter_pairwise_match_results(
            selected,
            engine_type=engine,
            rounds=rounds,
            trials=trials,
            error=error,
            initial_state=initial_state,
        )
    )
    return RoundRobinSummary(
        strategy_names=selected,
        engine_type=engine,
        rounds=rounds,
        trials=trials,
        error=error,
        match_results=results,
        rankings=build_rankings(selected, results),
    )


def _run_pairwise_match(
    strategy_a: str,
    strategy_b: str,
    *,
    engine_type: str,
    rounds: int,
    trials: int,
    error: float,
    initial_state: str,
) -> tuple[float, float]:
    player_a = create_strategy(strategy_a)
    player_b = create_strategy(strategy_b)

    if engine_type == "markov":
        game = MarkovGame(
            player_a,
            player_b,
            rounds=rounds,
            error=error,
            initial_state=initial_state,
        )
        score_a, score_b, _ = game.run()
        return float(score_a), float(score_b)

    if engine_type == "montecarlo":
        game = MonteCarloGame(
            player_a,
            player_b,
            rounds=rounds,
            trials=trials,
            error=error,
            initial_state=initial_state,
        )
        score_a, score_b = game.run()
        return float(score_a), float(score_b)

    raise ValueError(f"Unsupported engine_type after validation: {engine_type!r}")


def _validate_engine_type(engine_type: str) -> str:
    engine = engine_type.lower().replace("-", "").replace("_", "")
    if engine not in ENGINE_NAMES:
        valid = ", ".join(ENGINE_NAMES)
        raise ValueError(
            f"Invalid engine name {engine_type!r}. Available engines: {valid}"
        )
    return engine


def _validate_run_config(
    *,
    rounds: int,
    trials: int,
    error: float,
    initial_state: str,
) -> None:
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not 0.0 <= error <= 1.0:
        raise ValueError("error must be between 0.0 and 1.0")
    if initial_state not in payoff_matrix:
        raise ValueError(
            f"initial_state must be one of {tuple(payoff_matrix)}, got {initial_state!r}"
        )
