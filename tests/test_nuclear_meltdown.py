from __future__ import annotations

import pytest

from mad_world.actions import (
    InsufficientGDPError,
    InvalidGDPAmountError,
)
from mad_world.core import GameState
from mad_world.crises import (
    NuclearMeltdownAction,
    NuclearMeltdownCrisis,
    NuclearMeltdownDefs,
)
from mad_world.rules import GameRules


@pytest.fixture
def basic_game() -> GameState:
    return GameState.new_game(
        rules=GameRules(max_clock_state=100),
        players=["Player1", "Player2"],
    )


def test_nuclear_meltdown_world_ending(basic_game: GameState) -> None:
    crisis = NuclearMeltdownCrisis()
    bid = NuclearMeltdownDefs.GDP_THRESHOLD // 4
    actions = {
        "Player1": NuclearMeltdownAction(investment=bid),
        "Player2": NuclearMeltdownAction(investment=bid),
    }
    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[0].gdp_delta == {"Player1": -bid}
    assert events[1].gdp_delta == {"Player2": -bid}
    assert events[2].world_ending is True
    assert "underfunded" in events[2].description


def test_nuclear_meltdown_tie(basic_game: GameState) -> None:
    crisis = NuclearMeltdownCrisis()
    bid = NuclearMeltdownDefs.GDP_THRESHOLD // 2 + 1
    actions = {
        "Player1": NuclearMeltdownAction(investment=bid),
        "Player2": NuclearMeltdownAction(investment=bid),
    }
    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[2].world_ending is False
    total_investment = bid * 2
    expected_clock_impact = -(
        total_investment // NuclearMeltdownDefs.SCALING_FACTOR
    )
    assert events[2].clock_delta == expected_clock_impact
    assert (
        not hasattr(events[2], "influence_delta")
        or not events[2].influence_delta
    )


def test_nuclear_meltdown_p1_wins(basic_game: GameState) -> None:
    crisis = NuclearMeltdownCrisis()
    p1_bid = NuclearMeltdownDefs.GDP_THRESHOLD // 2 - 1
    p2_bid = NuclearMeltdownDefs.GDP_THRESHOLD - p1_bid + 2
    actions = {
        "Player1": NuclearMeltdownAction(investment=p1_bid),
        "Player2": NuclearMeltdownAction(investment=p2_bid),
    }
    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[2].world_ending is False
    total_investment = p1_bid + p2_bid
    expected_clock_impact = -(
        total_investment // NuclearMeltdownDefs.SCALING_FACTOR
    )
    assert events[2].clock_delta == expected_clock_impact
    assert events[2].influence_delta == {
        "Player1": NuclearMeltdownDefs.WINNER_INF
    }


def test_nuclear_meltdown_p2_wins(basic_game: GameState) -> None:
    crisis = NuclearMeltdownCrisis()
    p2_bid = NuclearMeltdownDefs.GDP_THRESHOLD // 2 - 1
    p1_bid = NuclearMeltdownDefs.GDP_THRESHOLD - p2_bid + 2
    actions = {
        "Player1": NuclearMeltdownAction(investment=p1_bid),
        "Player2": NuclearMeltdownAction(investment=p2_bid),
    }
    events = crisis.resolve(basic_game, actions)
    assert len(events) == 3
    assert events[2].world_ending is False
    total_investment = p1_bid + p2_bid
    expected_clock_impact = -(
        total_investment // NuclearMeltdownDefs.SCALING_FACTOR
    )
    assert events[2].clock_delta == expected_clock_impact
    assert events[2].influence_delta == {
        "Player2": NuclearMeltdownDefs.WINNER_INF
    }


def test_nuclear_meltdown_get_default_action(basic_game: GameState) -> None:
    crisis = NuclearMeltdownCrisis()

    basic_game.players["Player1"].gdp = 10

    action_agg = crisis.get_default_action(
        "Player1", basic_game, aggressive=True
    )
    assert action_agg.investment == 3

    action_cautious = crisis.get_default_action(
        "Player1", basic_game, aggressive=False
    )
    assert action_cautious.investment == 6

    # Check max bid constraint.
    basic_game.players["Player1"].gdp = 2
    action_constrained = crisis.get_default_action(
        "Player1", basic_game, aggressive=False
    )
    assert action_constrained.investment == 2


def test_nuclear_meltdown_validate_semantics(basic_game: GameState) -> None:
    action = NuclearMeltdownAction(investment=10)
    basic_game.players["Player1"].gdp = 15
    # Should not raise
    action.validate_semantics(basic_game, "Player1")

    basic_game.players["Player1"].gdp = 5
    with pytest.raises(InsufficientGDPError):
        action.validate_semantics(basic_game, "Player1")

    action_invalid = NuclearMeltdownAction(investment=-1)
    basic_game.players["Player1"].gdp = 15
    with pytest.raises(InvalidGDPAmountError):
        action_invalid.validate_semantics(basic_game, "Player1")
