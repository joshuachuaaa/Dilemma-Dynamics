"""Canonical strategy sets used by the research experiments."""

from Strategies.m0strategies import AlwaysDefect, RandomStrategy
from Strategies.m1strategies import TitForTat, WinStayLoseShift, ReverseTitForTat
from Strategies.m2strategies import (
    GenerousTwoTitForTwo,
    Pavlov2,
    SuspiciousTf2T,
    TitForTwoTats,
)
from Strategies.m3strategies import (
    Generous3,
    Pavlov3,
    PatternFollower3,
    ThreeGrudger,
    TwoForgiveOnePunish,
    UnforgivingPatternHunter,
)


def default_competitors():
    """Return fresh strategy instances for the baseline comparison set."""
    return [
        AlwaysDefect(),
        RandomStrategy(coop_prob=0.5),
        TitForTat(),
        WinStayLoseShift(),
        ReverseTitForTat(),
        TitForTwoTats(),
        Pavlov2(),
        GenerousTwoTitForTwo(),
        SuspiciousTf2T(),
        TwoForgiveOnePunish(),
        ThreeGrudger(),
        PatternFollower3(),
        Pavlov3(),
        Generous3(),
        UnforgivingPatternHunter(),
    ]
