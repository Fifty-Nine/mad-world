from __future__ import annotations

from typing import TYPE_CHECKING

from mad_world.effects import BaseEffect
from mad_world.events import SystemEvent

if TYPE_CHECKING:
    from mad_world.core import GameState


def test_apply_event_new_effect(basic_game: GameState) -> None:
    class TestEffect(BaseEffect):
        card_kind = "test_effect_new"
        title = "Test"
        description = "Test"
        mechanics = "Test"

        def run(self, game: GameState) -> list[SystemEvent]:
            return []

    e = TestEffect(duration=2)
    ev = SystemEvent(description="Test event", new_effects=[e])

    basic_game.current_round = 5
    basic_game.apply_event(ev)

    assert len(basic_game.active_effects) == 1
    assert basic_game.active_effects[0].start_round == 5


def test_expire_effects(basic_game: GameState) -> None:
    class TestEffectExp2(BaseEffect):
        card_kind = "test_effect_exp2"
        title = "Test Expire"
        description = "Test"
        mechanics = "Test"

        def run(self, game: GameState) -> list[SystemEvent]:
            return []

    e = TestEffectExp2(duration=2)
    ev = SystemEvent(description="Test event", new_effects=[e])

    basic_game.current_round = 1
    basic_game.apply_event(ev)
    assert len(basic_game.active_effects) == 1

    # Not expired yet
    basic_game._expire_effects()
    assert len(basic_game.active_effects) == 1

    # Advance to end of duration (start + duration = 1 + 2 = 3)
    basic_game.current_round = 3
    basic_game._expire_effects()
    assert len(basic_game.active_effects) == 0

    # Check that the expire event was logged
    last_event = basic_game.event_log[-1].event
    assert "has expired" in last_event.description
