"""Tests for the Blame Game crisis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mad_world.crises import (
    BLAME_BOTH_SHOULDER_CLOCK,
    BLAME_BOTH_SHOULDER_INF,
    BLAME_SINGLE_CLOCK,
    BLAME_SINGLE_DEFLECT_GDP_PENALTY,
    BLAME_SINGLE_DEFLECT_INF,
    BLAME_SINGLE_SHOULDER_INF,
    BlameGameAction,
    BlameGameCrisis,
)
from mad_world.enums import BlameGamePosture
from mad_world.events import PlayerActor
from mad_world.trivial_players import ParetoEfficientPlayer

if TYPE_CHECKING:
    from mad_world.core import GameState


def test_blame_game_both_shoulder(basic_game: GameState) -> None:
    crisis = BlameGameCrisis()
    events = crisis.resolve(
        basic_game,
        {
            "Alpha": BlameGameAction(posture=BlameGamePosture.SHOULDER),
            "Omega": BlameGameAction(posture=BlameGamePosture.SHOULDER),
        },
    )

    assert len(events) == 1
    event = events[0]
    assert event.world_ending is False
    assert event.clock_delta == BLAME_BOTH_SHOULDER_CLOCK
    assert event.influence_delta["Alpha"] == BLAME_BOTH_SHOULDER_INF
    assert event.influence_delta["Omega"] == BLAME_BOTH_SHOULDER_INF
    assert not event.gdp_delta


def test_blame_game_one_shoulders_one_deflects_no_penalty(
    basic_game: GameState,
) -> None:
    # Ensure deflector (Alpha) has <= debt than shoulderer (Omega).
    # Basic game starts with Alpha and Omega having 0 debt.
    crisis = BlameGameCrisis()
    events = crisis.resolve(
        basic_game,
        {
            "Alpha": BlameGameAction(posture=BlameGamePosture.DEFLECT),
            "Omega": BlameGameAction(posture=BlameGamePosture.SHOULDER),
        },
    )

    assert len(events) == 1
    event = events[0]
    assert event.world_ending is False
    assert event.clock_delta == BLAME_SINGLE_CLOCK
    assert event.influence_delta["Alpha"] == BLAME_SINGLE_DEFLECT_INF
    assert event.influence_delta["Omega"] == BLAME_SINGLE_SHOULDER_INF
    assert not event.gdp_delta


def test_blame_game_one_shoulders_one_deflects_with_penalty(
    basic_game: GameState,
) -> None:
    # Ensure deflector (Alpha) has > debt than shoulderer (Omega).
    # Manually modify the state to give Alpha some debt
    for i in range(10):
        basic_game.escalation_track[i] = PlayerActor(name="Alpha")

    crisis = BlameGameCrisis()
    events = crisis.resolve(
        basic_game,
        {
            "Alpha": BlameGameAction(posture=BlameGamePosture.DEFLECT),
            "Omega": BlameGameAction(posture=BlameGamePosture.SHOULDER),
        },
    )

    assert len(events) == 1
    event = events[0]
    assert event.world_ending is False
    assert event.clock_delta == BLAME_SINGLE_CLOCK
    assert event.influence_delta["Alpha"] == BLAME_SINGLE_DEFLECT_INF
    assert event.influence_delta["Omega"] == BLAME_SINGLE_SHOULDER_INF
    assert event.gdp_delta["Alpha"] == BLAME_SINGLE_DEFLECT_GDP_PENALTY
    assert "Omega" not in event.gdp_delta


def test_blame_game_both_deflect(basic_game: GameState) -> None:
    crisis = BlameGameCrisis()
    events = crisis.resolve(
        basic_game,
        {
            "Alpha": BlameGameAction(posture=BlameGamePosture.DEFLECT),
            "Omega": BlameGameAction(posture=BlameGamePosture.DEFLECT),
        },
    )

    assert len(events) == 1
    event = events[0]
    assert event.world_ending is True
    assert event.clock_delta == 0
    assert not event.influence_delta
    assert not event.gdp_delta


def test_blame_game_get_default_action() -> None:
    crisis = BlameGameCrisis()
    assert (
        crisis.get_default_action(aggressive=True).posture
        == BlameGamePosture.DEFLECT
    )
    assert (
        crisis.get_default_action(aggressive=False).posture
        == BlameGamePosture.SHOULDER
    )


@pytest.mark.asyncio
async def test_pareto_crisis_message(basic_game: GameState) -> None:
    player = ParetoEfficientPlayer("Alpha")
    crisis = BlameGameCrisis()

    msg = await player.crisis_message(basic_game, crisis)
    assert msg.message_to_opponent is not None
    assert "DEFLECT" in msg.message_to_opponent

    basic_game.escalation_track[0] = PlayerActor(name="Alpha")

    msg2 = await player.crisis_message(basic_game, crisis)
    assert msg2.message_to_opponent is not None
    assert "SHOULDER" in msg2.message_to_opponent


@pytest.mark.asyncio
async def test_pareto_crisis_action(basic_game: GameState) -> None:
    player = ParetoEfficientPlayer("Alpha")
    crisis = BlameGameCrisis()

    action = await player.crisis(basic_game, crisis)
    assert action.posture == BlameGamePosture.DEFLECT

    basic_game.escalation_track[0] = PlayerActor(name="Alpha")
    action2 = await player.crisis(basic_game, crisis)
    assert action2.posture == BlameGamePosture.SHOULDER
