from __future__ import annotations

import pytest

from mad_world.actions import (
    InsufficientGDPError,
    InvalidGDPAmountError,
)
from mad_world.core import GameState
from mad_world.crises import (
    DoomsdayAsteroidAction,
    DoomsdayAsteroidCrisis,
    DoomsdayAsteroidDefs,
)
from mad_world.rules import GameRules


@pytest.fixture
def basic_game() -> GameState:
    return GameState.new_game(
        rules=GameRules(max_clock_state=100),
        players=["Player1", "Player2"],
    )


def test_doomsday_asteroid_world_ending(basic_game: GameState) -> None:
    crisis = DoomsdayAsteroidCrisis()
    bid = DoomsdayAsteroidDefs.GDP_THRESHOLD // 4
    actions = {
        "Player1": DoomsdayAsteroidAction(investment=bid),
        "Player2": DoomsdayAsteroidAction(investment=bid),
    }
    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[0].gdp_delta == {"Player1": -bid}
    assert events[1].gdp_delta == {"Player2": -bid}
    assert events[2].world_ending is True
    assert "failed to deflect" in events[2].description


def test_doomsday_asteroid_tie(basic_game: GameState) -> None:
    crisis = DoomsdayAsteroidCrisis()
    bid = DoomsdayAsteroidDefs.GDP_THRESHOLD // 2 + 1
    actions = {
        "Player1": DoomsdayAsteroidAction(investment=bid),
        "Player2": DoomsdayAsteroidAction(investment=bid),
    }
    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[2].world_ending is False
    assert events[2].gdp_delta == {
        "Player1": DoomsdayAsteroidDefs.WINNER_GDP // 2,
        "Player2": DoomsdayAsteroidDefs.WINNER_GDP // 2,
    }
    assert events[2].influence_delta == {
        "Player1": DoomsdayAsteroidDefs.WINNER_INF // 2,
        "Player2": DoomsdayAsteroidDefs.WINNER_INF // 2,
    }


def test_doomsday_asteroid_p1_wins(basic_game: GameState) -> None:
    crisis = DoomsdayAsteroidCrisis()
    p1_bid = DoomsdayAsteroidDefs.GDP_THRESHOLD // 2 + 1
    p2_bid = DoomsdayAsteroidDefs.GDP_THRESHOLD - p1_bid
    actions = {
        "Player1": DoomsdayAsteroidAction(investment=p1_bid),
        "Player2": DoomsdayAsteroidAction(investment=p2_bid),
    }
    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[2].world_ending is False
    assert events[2].gdp_delta == {"Player1": DoomsdayAsteroidDefs.WINNER_GDP}
    assert events[2].influence_delta == {
        "Player1": DoomsdayAsteroidDefs.WINNER_INF
    }


def test_doomsday_asteroid_p2_wins(basic_game: GameState) -> None:
    crisis = DoomsdayAsteroidCrisis()
    p2_bid = DoomsdayAsteroidDefs.GDP_THRESHOLD // 2 + 1
    p1_bid = DoomsdayAsteroidDefs.GDP_THRESHOLD - p2_bid
    actions = {
        "Player1": DoomsdayAsteroidAction(investment=p1_bid),
        "Player2": DoomsdayAsteroidAction(investment=p2_bid),
    }
    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[2].world_ending is False
    assert events[2].gdp_delta == {"Player2": DoomsdayAsteroidDefs.WINNER_GDP}
    assert events[2].influence_delta == {
        "Player2": DoomsdayAsteroidDefs.WINNER_INF
    }


def test_doomsday_asteroid_get_default_action(basic_game: GameState) -> None:
    crisis = DoomsdayAsteroidCrisis()
    # Aggressive player bids 25% of threshold.
    action_agg = crisis.get_default_action(
        "Player1", basic_game, aggressive=True
    )
    expected_agg = (DoomsdayAsteroidDefs.GDP_THRESHOLD // 2) // 2
    assert action_agg.investment == expected_agg

    # Cautious player bids 50% of threshold.
    action_cautious = crisis.get_default_action(
        "Player1", basic_game, aggressive=False
    )
    expected_cautious = DoomsdayAsteroidDefs.GDP_THRESHOLD // 2
    assert action_cautious.investment == expected_cautious

    # Check max bid constraint.
    basic_game.players["Player1"].gdp = 1
    action_constrained = crisis.get_default_action(
        "Player1", basic_game, aggressive=False
    )
    assert action_constrained.investment == 1


def test_doomsday_asteroid_validate_semantics(basic_game: GameState) -> None:
    action = DoomsdayAsteroidAction(investment=10)
    basic_game.players["Player1"].gdp = 15
    # Should not raise
    action.validate_semantics(basic_game, "Player1")

    basic_game.players["Player1"].gdp = 5
    with pytest.raises(InsufficientGDPError):
        action.validate_semantics(basic_game, "Player1")

    action_invalid = DoomsdayAsteroidAction(investment=-1)
    basic_game.players["Player1"].gdp = 15
    with pytest.raises(InvalidGDPAmountError):
        action_invalid.validate_semantics(basic_game, "Player1")
