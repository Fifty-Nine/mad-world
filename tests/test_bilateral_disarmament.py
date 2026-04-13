"""Tests for Bilateral Disarmament Crisis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mad_world.actions import (
    InsufficientInfluenceError,
    InvalidInfluenceAmountError,
)
from mad_world.crises import (
    BilateralDisarmamentAction,
    BilateralDisarmamentCrisis,
    BilateralDisarmamentDefs,
)

if TYPE_CHECKING:
    from mad_world.core import GameState


def test_bilateral_disarmament_tie(basic_game: GameState) -> None:
    crisis = BilateralDisarmamentCrisis()
    bid = 5

    actions = {
        "Alpha": BilateralDisarmamentAction(investment=bid),
        "Omega": BilateralDisarmamentAction(investment=bid),
    }

    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[0].influence_delta == {"Alpha": -bid}
    assert events[1].influence_delta == {"Omega": -bid}
    assert events[2].clock_delta == -bid
    assert "true bilateral disarmament" in events[2].description
    assert events[2].influence_delta == {}
    assert events[2].gdp_delta == {}


def test_bilateral_disarmament_p1_wins(basic_game: GameState) -> None:
    crisis = BilateralDisarmamentCrisis()
    p1_bid = 2
    p2_bid = 5

    actions = {
        "Alpha": BilateralDisarmamentAction(investment=p1_bid),
        "Omega": BilateralDisarmamentAction(investment=p2_bid),
    }

    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[0].influence_delta == {"Alpha": -p1_bid}
    assert events[1].influence_delta == {"Omega": -p2_bid}
    assert events[2].clock_delta == -p1_bid
    assert events[2].influence_delta == {
        "Alpha": BilateralDisarmamentDefs.WINNER_INF
    }
    assert events[2].gdp_delta == {"Omega": BilateralDisarmamentDefs.LOSER_GDP}
    assert (
        "Alpha refused to make meaningful concessions" in events[2].description
    )


def test_bilateral_disarmament_p2_wins(basic_game: GameState) -> None:
    crisis = BilateralDisarmamentCrisis()
    p1_bid = 5
    p2_bid = 2

    actions = {
        "Alpha": BilateralDisarmamentAction(investment=p1_bid),
        "Omega": BilateralDisarmamentAction(investment=p2_bid),
    }

    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[0].influence_delta == {"Alpha": -p1_bid}
    assert events[1].influence_delta == {"Omega": -p2_bid}
    assert events[2].clock_delta == -p2_bid
    assert events[2].influence_delta == {
        "Omega": BilateralDisarmamentDefs.WINNER_INF
    }
    assert events[2].gdp_delta == {"Alpha": BilateralDisarmamentDefs.LOSER_GDP}
    assert (
        "Omega refused to make meaningful concessions" in events[2].description
    )


def test_bilateral_disarmament_defaults(basic_game: GameState) -> None:
    crisis = BilateralDisarmamentCrisis()

    # Aggressive player bids 0
    basic_game.players["Alpha"].influence = 10
    action_agg = crisis.get_default_action("Alpha", basic_game, aggressive=True)
    assert action_agg.investment == 0

    # Diplomatic player bids half of influence
    action_dip = crisis.get_default_action(
        "Alpha", basic_game, aggressive=False
    )
    assert action_dip.investment == 5


def test_bilateral_disarmament_action_validation(basic_game: GameState) -> None:
    basic_game.players["Alpha"].influence = 5

    action_valid = BilateralDisarmamentAction(investment=5)
    action_valid.validate_semantics(basic_game, "Alpha")

    action_insufficient = BilateralDisarmamentAction(investment=6)
    with pytest.raises(InsufficientInfluenceError):
        action_insufficient.validate_semantics(basic_game, "Alpha")

    action_invalid = BilateralDisarmamentAction(investment=-1)
    with pytest.raises(InvalidInfluenceAmountError):
        action_invalid.validate_semantics(basic_game, "Alpha")


def test_bilateral_disarmament_action_type() -> None:
    crisis = BilateralDisarmamentCrisis()
    assert crisis.action_type == BilateralDisarmamentAction
