from __future__ import annotations

import json

import mad_world.event_cards as ec
from mad_world.core import GameState


def test_game_state_serialization(basic_game: GameState) -> None:
    """Test GameState can be serialized to JSON and completely deserialized."""
    json_data = basic_game.model_dump_json()

    # Ensure it's valid JSON
    data = json.loads(json_data)

    # Reconstruct the game state
    restored_game = GameState.model_validate(data)

    # Ensure attributes are restored correctly
    assert restored_game.current_round == basic_game.current_round
    assert restored_game.current_phase == basic_game.current_phase
    assert len(restored_game.players) == len(basic_game.players)

    orig_draw_pile = basic_game.event_deck.draw_pile
    rest_draw_pile = restored_game.event_deck.draw_pile
    assert rest_draw_pile[0].card_kind == orig_draw_pile[0].card_kind

    for card, restored_card in zip(orig_draw_pile, rest_draw_pile, strict=True):
        if isinstance(card, ec.BasePlayerEffectCard):
            assert isinstance(restored_card, ec.BasePlayerEffectCard)
            assert card.player_idx == restored_card.player_idx
            assert card.amount == restored_card.amount
