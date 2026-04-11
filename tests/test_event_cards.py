from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mad_world.core import resolve_round_events
from mad_world.decks import Deck
from mad_world.enums import GamePhase
from mad_world.event_cards import (
    BaseEventCard,
    ClockDownEvent,
    ClockUpEvent,
    GDPEvent,
    InfluenceBothEvent,
    InfluenceEvent,
    create_event_deck,
)

if TYPE_CHECKING:
    from mad_world.core import GameState


def test_clock_up_event(basic_game: GameState) -> None:
    event = ClockUpEvent()
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].clock_delta == 1


def test_clock_down_event(basic_game: GameState) -> None:
    event = ClockDownEvent()
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].clock_delta == -1


def test_influence_p1_event(basic_game: GameState) -> None:
    event = InfluenceEvent(player_idx=0)
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].influence_delta == {"Alpha": 3}


def test_influence_p2_event(basic_game: GameState) -> None:
    event = InfluenceEvent(player_idx=1)
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].influence_delta == {"Omega": 3}


def test_gdp_p1_event(basic_game: GameState) -> None:
    event = GDPEvent(player_idx=0)
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].gdp_delta == {"Alpha": event.gdp_bonus}


def test_gdp_p2_event(basic_game: GameState) -> None:
    event = GDPEvent(player_idx=1)
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].gdp_delta == {"Omega": event.gdp_bonus}


def test_influence_both_event(basic_game: GameState) -> None:
    event = InfluenceBothEvent()
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].influence_delta == {"Alpha": 2, "Omega": 2}


def test_create_event_deck(basic_game: GameState) -> None:
    deck = create_event_deck(basic_game.rng)
    assert len(deck) == 30


@pytest.mark.asyncio
async def test_resolve_round_events(basic_game: GameState) -> None:
    basic_game.event_deck = Deck[BaseEventCard].create(
        [ClockUpEvent()], basic_game.rng
    )
    basic_game.current_phase = GamePhase.ROUND_EVENTS

    initial_clock = basic_game.doomsday_clock

    new_game = await resolve_round_events(basic_game)

    assert new_game.doomsday_clock == initial_clock + 1
    assert len(new_game.event_deck.discard_pile) == 1
    assert new_game.current_phase == GamePhase.BIDDING_MESSAGING
