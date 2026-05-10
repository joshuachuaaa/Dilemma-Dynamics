from apps.strategy_tui import format_trace_table
from Simulation.trace import simulate_match_trace
from Strategies.m0strategies import AlwaysDefect
from Strategies.m1strategies import TitForTat


def test_format_trace_table_includes_timeline_and_final_score():
    trace = simulate_match_trace(TitForTat(), AlwaysDefect(), rounds=3, error=0.0)
    output = format_trace_table("TitForTat", "AlwaysDefect", trace)

    assert "Dilemma Dynamics trace: TitForTat vs AlwaysDefect" in output
    assert "Timeline: CD DD DD" in output
    assert "Final score: TitForTat 2, AlwaysDefect 7" in output
