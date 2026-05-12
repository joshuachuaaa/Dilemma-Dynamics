from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps import simulation_gui_model as model


def test_strategy_names_are_exposed_for_selection():
    names = model.list_strategy_names()

    assert names == tuple(sorted(names))
    assert {"AlwaysCooperate", "AlwaysDefect", "TitForTat"}.issubset(names)


def test_selected_strategies_produce_trace_summary():
    summary = model.build_trace_summary(
        "TitForTat",
        "AlwaysDefect",
        rounds=3,
        error=0.0,
        seed=123,
    )

    assert summary.strategy_a == "TitForTat"
    assert summary.strategy_b == "AlwaysDefect"
    assert summary.timeline == ("CD", "DD", "DD")
    assert summary.final_score_a == 2
    assert summary.final_score_b == 7


def test_invalid_strategy_name_fails_clearly():
    with pytest.raises(ValueError) as excinfo:
        model.build_trace_summary("TitForTat", "NotARealStrategy", rounds=1)

    message = str(excinfo.value)
    assert "Invalid strategy_b strategy name 'NotARealStrategy'" in message
    assert "Available strategies:" in message
    assert "AlwaysDefect" in message


def test_score_series_and_table_rows_are_headless_trace_views():
    summary = model.build_trace_summary(
        "TitForTat",
        "AlwaysDefect",
        rounds=3,
        error=0.0,
    )

    assert [
        (point.round_index, point.strategy_a_score, point.strategy_b_score)
        for point in summary.score_points
    ] == [(0, 0.0, 0.0), (1, 0, 5), (2, 1, 6), (3, 2, 7)]

    assert [
        (
            row.round_index,
            row.memory,
            row.strategy_a_move,
            row.strategy_b_move,
            row.outcome,
            row.payoff_a,
            row.payoff_b,
            row.cumulative_score_a,
            row.cumulative_score_b,
        )
        for row in summary.table_rows
    ] == [
        (1, "CC", "C", "D", "CD", 0, 5, 0, 5),
        (2, "CD", "D", "D", "DD", 1, 1, 1, 6),
        (3, "DD", "D", "D", "DD", 1, 1, 2, 7),
    ]
