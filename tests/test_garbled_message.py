from __future__ import annotations

from typing import TYPE_CHECKING

from mad_world.actions import ChatAction, MessagingAction
from mad_world.effects import RFInterferenceEffect
from mad_world.events import GarbledMessageEvent, MessageEvent, PlayerActor

if TYPE_CHECKING:
    from mad_world.core import GameState


def test_garbled_message_log(basic_game: GameState) -> None:
    effect = RFInterferenceEffect(duration=2)
    basic_game.active_effects.append(effect)
    action = ChatAction(
        chat_message="This is a completely normal message.", end_channel=False
    )
    basic_game.log_message("Alpha", "Omega", action)
    last_event = basic_game.event_log[-1].event
    assert isinstance(last_event, GarbledMessageEvent)
    assert last_event.original_message == "This is a completely normal message."
    assert last_event.message != last_event.original_message


def test_done_by_player_garbled_message() -> None:
    e = GarbledMessageEvent(
        description="Test",
        actor=PlayerActor(name="Alpha"),
        message="Garbled",
        original_message="Original",
        channel_message=False,
    )
    assert e.done_by_player("Alpha") is True
    assert e.done_by_player("Omega") is False


def test_non_garbled_message_log(basic_game: GameState) -> None:
    action = MessagingAction(message_to_opponent="Normal")
    basic_game.log_message("Alpha", "Omega", action)
    last_event = basic_game.event_log[-1].event
    assert isinstance(last_event, MessageEvent)
    assert not isinstance(last_event, GarbledMessageEvent)
    assert last_event.message == "Normal"
