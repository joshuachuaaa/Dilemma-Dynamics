from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps import round_robin_gui_model as model


SMALL_SELECTION = ("AlwaysCooperate", "AlwaysDefect", "TitForTat")


def test_strategy_subset_selection_preserves_gui_order():
    names = model.list_strategy_names()

    assert names == tuple(sorted(names))
    assert set(SMALL_SELECTION).issubset(names)
    assert model.select_strategy_names(SMALL_SELECTION) == SMALL_SELECTION


def test_pairwise_match_results_are_yielded_incrementally():
    results = model.iter_pairwise_match_results(
        SMALL_SELECTION,
        engine_type="markov",
        rounds=3,
        error=0.0,
    )

    first = next(results)

    assert first.match_index == 1
    assert first.total_matches == 3
    assert first.strategy_a == "AlwaysCooperate"
    assert first.strategy_b == "AlwaysDefect"
    assert first.score_a == 0.0
    assert first.score_b == 15.0

    assert [
        (result.match_index, result.strategy_a, result.strategy_b)
        for result in results
    ] == [
        (2, "AlwaysCooperate", "TitForTat"),
        (3, "AlwaysDefect", "TitForTat"),
    ]


def test_rankings_can_be_built_from_partial_results():
    first_result = next(
        model.iter_pairwise_match_results(
            SMALL_SELECTION,
            engine_type="markov",
            rounds=3,
            error=0.0,
        )
    )

    rankings = model.build_rankings(SMALL_SELECTION, [first_result])

    assert [
        (
            row.rank,
            row.strategy,
            row.total_score,
            row.matches_played,
            row.wins,
            row.losses,
            row.ties,
        )
        for row in rankings
    ] == [
        (1, "AlwaysDefect", 15.0, 1, 1, 0, 0),
        (2, "AlwaysCooperate", 0.0, 1, 0, 1, 0),
        (2, "TitForTat", 0.0, 0, 0, 0, 0),
    ]


def test_final_rankings_after_all_round_robin_matches():
    summary = model.run_round_robin(
        SMALL_SELECTION,
        engine_type="markov",
        rounds=3,
        error=0.0,
    )

    assert [
        (result.strategy_a, result.strategy_b, result.score_a, result.score_b)
        for result in summary.match_results
    ] == [
        ("AlwaysCooperate", "AlwaysDefect", 0.0, 15.0),
        ("AlwaysCooperate", "TitForTat", 9.0, 9.0),
        ("AlwaysDefect", "TitForTat", 7.0, 2.0),
    ]
    assert [
        (
            row.rank,
            row.strategy,
            row.total_score,
            row.matches_played,
            row.wins,
            row.losses,
            row.ties,
        )
        for row in summary.rankings
    ] == [
        (1, "AlwaysDefect", 22.0, 2, 2, 0, 0),
        (2, "TitForTat", 11.0, 2, 0, 1, 1),
        (3, "AlwaysCooperate", 9.0, 2, 0, 1, 1),
    ]


def test_invalid_engine_name_fails_clearly():
    with pytest.raises(ValueError) as excinfo:
        model.iter_pairwise_match_results(SMALL_SELECTION, engine_type="NotAnEngine")

    message = str(excinfo.value)
    assert "Invalid engine name 'NotAnEngine'" in message
    assert "Available engines: markov, montecarlo" in message


def test_invalid_strategy_name_fails_clearly():
    with pytest.raises(ValueError) as excinfo:
        model.select_strategy_names(("TitForTat", "NotARealStrategy"))

    message = str(excinfo.value)
    assert "Invalid strategy name 'NotARealStrategy'" in message
    assert "Available strategies:" in message
    assert "AlwaysDefect" in message
