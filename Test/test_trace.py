import pytest

from Simulation.trace import simulate_match_trace, trace_to_dataframe
from Strategies.m0strategies import AlwaysDefect, RandomStrategy
from Strategies.m1strategies import TitForTat


def test_tit_for_tat_vs_always_defect_trace_scores():
    trace = simulate_match_trace(TitForTat(), AlwaysDefect(), rounds=3, error=0.0)

    assert [row.outcome for row in trace] == ["CD", "DD", "DD"]
    assert [(row.payoff_1, row.payoff_2) for row in trace] == [(0, 5), (1, 1), (1, 1)]
    assert trace[-1].cumulative_score_1 == 2
    assert trace[-1].cumulative_score_2 == 7
    assert trace[0].state_before == ("CC",)
    assert trace[0].state_after == ("CD",)


def test_trace_seed_reproducibility_with_random_strategy():
    first = simulate_match_trace(
        RandomStrategy(0.5), RandomStrategy(0.5), rounds=8, error=0.1, seed=123
    )
    second = simulate_match_trace(
        RandomStrategy(0.5), RandomStrategy(0.5), rounds=8, error=0.1, seed=123
    )

    assert [row.outcome for row in first] == [row.outcome for row in second]
    assert [row.error_flip_1 for row in first] == [row.error_flip_1 for row in second]


def test_trace_to_dataframe_shape():
    trace = simulate_match_trace(TitForTat(), AlwaysDefect(), rounds=2, error=0.0)
    df = trace_to_dataframe(trace)

    assert list(df["round_index"]) == [1, 2]
    assert list(df["outcome"]) == ["CD", "DD"]


@pytest.mark.parametrize("rounds,error,initial_state", [(-1, 0.0, "CC"), (2, 1.1, "CC"), (2, 0.0, "XX")])
def test_trace_config_validation(rounds, error, initial_state):
    with pytest.raises(ValueError):
        simulate_match_trace(
            TitForTat(),
            AlwaysDefect(),
            rounds=rounds,
            error=error,
            initial_state=initial_state,
        )
