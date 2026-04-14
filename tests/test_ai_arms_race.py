from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mad_world.actions import InsufficientGDPError, InvalidGDPAmountError
from mad_world.crises import AIArmsRaceAction, AIArmsRaceCrisis, AIArmsRaceDefs

if TYPE_CHECKING:
    from mad_world.core import GameState


@pytest.fixture
def crisis() -> AIArmsRaceCrisis:
    return AIArmsRaceCrisis()


def test_action_validation(basic_game: GameState) -> None:
    action = AIArmsRaceAction(investment=basic_game.players["Alpha"].gdp + 1)
    with pytest.raises(InsufficientGDPError):
        action.validate_semantics(basic_game, "Alpha")

    action = AIArmsRaceAction(investment=-1)
    with pytest.raises(InvalidGDPAmountError):
        action.validate_semantics(basic_game, "Alpha")

    action = AIArmsRaceAction(investment=basic_game.players["Alpha"].gdp)
    action.validate_semantics(basic_game, "Alpha")


def test_resolve_ai_arms_race_tied_bids(
    basic_game: GameState, crisis: AIArmsRaceCrisis
) -> None:
    actions = {
        "Alpha": AIArmsRaceAction(investment=10),
        "Omega": AIArmsRaceAction(investment=10),
    }

    results = crisis.resolve(basic_game, actions)
    assert len(results) == 3

    assert results[0].gdp_delta == {"Alpha": -10}
    assert results[1].gdp_delta == {"Omega": -10}
    assert results[2].description == (
        "Both superpowers achieved parity in AI research, "
        "with neither gaining a definitive advantage."
    )
    assert getattr(results[2], "world_ending", False) is False
    assert results[2].influence_delta == {}


def test_resolve_ai_arms_race_winner_p2(
    basic_game: GameState, crisis: AIArmsRaceCrisis
) -> None:
    actions = {
        "Alpha": AIArmsRaceAction(investment=10),
        "Omega": AIArmsRaceAction(investment=12),
    }

    results = crisis.resolve(basic_game, actions)
    assert len(results) == 3

    assert results[0].gdp_delta == {"Alpha": -10}
    assert results[1].gdp_delta == {"Omega": -12}
    assert results[2].description == "Omega secured AI dominance."
    assert results[2].influence_delta == {"Omega": AIArmsRaceDefs.WINNER_INF}
    assert getattr(results[2], "world_ending", False) is False


def test_resolve_ai_arms_race_winner_p1(
    basic_game: GameState, crisis: AIArmsRaceCrisis
) -> None:
    actions = {
        "Alpha": AIArmsRaceAction(investment=12),
        "Omega": AIArmsRaceAction(investment=10),
    }
    actions = {
        "Alpha": AIArmsRaceAction(investment=12),
        "Omega": AIArmsRaceAction(investment=10),
    }

    results = crisis.resolve(basic_game, actions)
    assert len(results) == 3

    assert results[0].gdp_delta == {"Alpha": -12}
    assert results[1].gdp_delta == {"Omega": -10}
    assert results[2].description == "Alpha secured AI dominance."
    assert results[2].influence_delta == {"Alpha": AIArmsRaceDefs.WINNER_INF}
    assert getattr(results[2], "world_ending", False) is False


def test_resolve_ai_arms_race_rogue_ai(
    basic_game: GameState, crisis: AIArmsRaceCrisis
) -> None:
    actions = {
        "Alpha": AIArmsRaceAction(investment=16),
        "Omega": AIArmsRaceAction(investment=15),
    }

    results = crisis.resolve(basic_game, actions)
    assert len(results) == 3

    assert results[0].gdp_delta == {"Alpha": -16}
    assert results[1].gdp_delta == {"Omega": -15}
    assert getattr(results[2], "world_ending", False) is True


def test_properties(crisis: AIArmsRaceCrisis) -> None:
    assert crisis.action_type is AIArmsRaceAction


def test_default_action(
    basic_game: GameState, crisis: AIArmsRaceCrisis
) -> None:
    agg_action = crisis.get_default_action("Alpha", basic_game, aggressive=True)
    dip_action = crisis.get_default_action(
        "Alpha", basic_game, aggressive=False
    )

    assert agg_action.investment == AIArmsRaceDefs.GDP_THRESHOLD // 2
    assert dip_action.investment == AIArmsRaceDefs.GDP_THRESHOLD // 4
