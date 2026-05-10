"""Named strategy registry for scripts and interactive tools."""

from Strategies.m0strategies import AlwaysCooperate, AlwaysDefect, RandomStrategy
from Strategies.m1strategies import GrimTrigger, ReverseTitForTat, TitForTat, WinStayLoseShift
from Strategies.m2strategies import (
    ClearGrudger,
    GenerousTwoTitForTwo,
    Grim2,
    Pavlov2,
    Prober,
    SuspiciousTf2T,
    TitForTwoTats,
    Vindictive2,
)
from Strategies.m3strategies import (
    Generous3,
    Pavlov3,
    PatternFollower3,
    ThreeGrudger,
    TitForThreeTats,
    TwoForgiveOnePunish,
    UnforgivingPatternHunter,
)


STRATEGY_FACTORIES = {
    "AlwaysCooperate": AlwaysCooperate,
    "AlwaysDefect": AlwaysDefect,
    "RandomStrategy": lambda: RandomStrategy(0.5),
    "TitForTat": TitForTat,
    "WinStayLoseShift": WinStayLoseShift,
    "ReverseTitForTat": ReverseTitForTat,
    "GrimTrigger": GrimTrigger,
    "TitForTwoTats": TitForTwoTats,
    "ClearGrudger": ClearGrudger,
    "Pavlov2": Pavlov2,
    "GenerousTwoTitForTwo": GenerousTwoTitForTwo,
    "G2T2T": GenerousTwoTitForTwo,
    "SuspiciousTf2T": SuspiciousTf2T,
    "Prober": Prober,
    "Grim2": Grim2,
    "Vindictive2": Vindictive2,
    "TitForThreeTats": TitForThreeTats,
    "TwoForgiveOnePunish": TwoForgiveOnePunish,
    "ThreeGrudger": ThreeGrudger,
    "PatternFollower3": PatternFollower3,
    "Pavlov3": Pavlov3,
    "Generous3": Generous3,
    "UnforgivingPatternHunter": UnforgivingPatternHunter,
}


def available_strategy_names() -> list[str]:
    """Return strategy names accepted by scripts and apps."""
    return sorted(STRATEGY_FACTORIES)


def create_strategy(name: str):
    """Create a fresh strategy instance by registry name."""
    try:
        return STRATEGY_FACTORIES[name]()
    except KeyError as exc:
        valid = ", ".join(available_strategy_names())
        raise ValueError(f"Unknown strategy {name!r}. Available strategies: {valid}") from exc
