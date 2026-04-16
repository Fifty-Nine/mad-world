from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mad_world.core import resolve_round_events
from mad_world.decks import Deck
from mad_world.enums import GamePhase
from mad_world.event_cards import (
    BaseEventCard,
    BasePlayerEffectCard,
    ClockChangeEvent,
    GDPEvent,
    GlobalSanctionsEvent,
    InfluenceBothEvent,
    InfluenceChangeEvent,
    RFInterferenceEvent,
    UNPeacekeepingEvent,
    create_event_deck,
)
from mad_world.events import BaseGameEvent
from mad_world.util import gain_or_lose

if TYPE_CHECKING:
    from collections.abc import Callable

    from mad_world.core import GameState


def test_clock_up_event(basic_game: GameState) -> None:
    event = ClockChangeEvent(amount=1)
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].clock_delta == 1
    assert "increases by 1" in game_events[0].description


def test_clock_down_event(basic_game: GameState) -> None:
    event = ClockChangeEvent(amount=-1)
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].clock_delta == -1
    assert "decreases by 1" in game_events[0].description


@pytest.mark.parametrize(
    ("card_type", "idx", "amount"),
    [
        (InfluenceChangeEvent, 0, 1),
        (InfluenceChangeEvent, 0, -2),
        (InfluenceChangeEvent, 1, 3),
        (InfluenceChangeEvent, 1, -4),
        (GDPEvent, 0, -5),
        (GDPEvent, 0, 6),
        (GDPEvent, 1, -7),
        (GDPEvent, 1, 8),
    ],
)
def test_player_effect_cards(
    card_type: Callable[..., BasePlayerEffectCard],
    idx: int,
    amount: int,
    basic_game: GameState,
) -> None:
    event_card = card_type(player_idx=idx, amount=amount)
    game_events = event_card.run(basic_game)
    assert len(game_events) == 1

    event = game_events[0]
    player = basic_game.player_names[idx]
    key = event_card.effect_key()
    units = event_card.effect_units()
    desc = gain_or_lose(amount)

    assert key in BaseGameEvent.model_fields

    assert getattr(event, key) == {player: amount}
    assert f"{desc}s {abs(amount)}{units}" in event.description

    event_card.model_dump_json(indent=2)


def test_influence_both_event(basic_game: GameState) -> None:
    event = InfluenceBothEvent()
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert game_events[0].influence_delta == {"Alpha": 2, "Omega": 2}


def test_un_peacekeeping_event(basic_game: GameState) -> None:
    event = UNPeacekeepingEvent()
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert len(game_events[0].new_effects) == 1
    assert game_events[0].new_effects[0].card_kind == "un_peacekeeping"


def test_global_sanctions_event(basic_game: GameState) -> None:
    event = GlobalSanctionsEvent()
    game_events = event.run(basic_game)
    assert len(game_events) == 1
    assert len(game_events[0].new_effects) == 1
    assert game_events[0].new_effects[0].card_kind == "global_sanctions"


def test_create_event_deck(basic_game: GameState) -> None:
    deck = create_event_deck(basic_game.rng)
    assert len(deck) == 49


@pytest.mark.asyncio
async def test_resolve_round_events(basic_game: GameState) -> None:
    basic_game.event_deck = Deck[BaseEventCard].create(
        [ClockChangeEvent(amount=1)], basic_game.rng
    )
    basic_game.current_phase = GamePhase.ROUND_EVENTS

    initial_clock = basic_game.doomsday_clock

    new_game = await resolve_round_events(basic_game)

    assert new_game.doomsday_clock == initial_clock + 1
    assert len(new_game.event_deck.discard_pile) == 1
    assert new_game.current_phase == GamePhase.BIDDING_MESSAGING


def test_rf_interference_event() -> None:
    e = RFInterferenceEvent()
    assert e.title == "RF Interference"
    assert "garbles" in e.description
    assert e.effect_type().__name__ == "RFInterferenceEffect"
