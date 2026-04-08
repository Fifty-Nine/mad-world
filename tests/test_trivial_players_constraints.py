from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mad_world.core import game_loop
from mad_world.trivial_players import (
    Capitalist,
    CrazyIvan,
    Diplomat,
    Pacifist,
    Saboteur,
)

if TYPE_CHECKING:
    from mad_world.players import GamePlayer
    from mad_world.rules import GameRules


@pytest.mark.asyncio
async def test_trivial_players_with_constraints(
    stable_rules: GameRules,
) -> None:
    # Reduce allowed bids and operations directly via active_effects
    # We can inject these effects at the start of the game to ensure
    # the bots hit their fallback logic.

    # Capitalist fallback (no domestic investment)
    # Pacifist fallback (no 0 bid)
    # CrazyIvan fallback (no first-strike) - actually we can just manually
    # restrict the operations dict.
    stable_rules.round_count = 1

    # Empty operations and bids
    stable_rules.allowed_bids = [10]
    stable_rules.allowed_operations = {}

    players: list[GamePlayer] = [
        Capitalist("Alpha"),
        CrazyIvan("Omega"),
    ]

    _, _, game = await game_loop(stable_rules, players)
    # Game should complete without crashing from KeyError
    assert game.current_round == 2

    players2: list[GamePlayer] = [
        Pacifist("Alpha"),
        Saboteur("Omega"),
    ]

    _, _, game2 = await game_loop(stable_rules, players2)
    # Game should complete without crashing
    assert game2.current_round == 2

    players3: list[GamePlayer] = [
        Diplomat("Alpha"),
        Diplomat("Omega"),
    ]

    _, _, game3 = await game_loop(stable_rules, players3)
    assert game3.current_round == 2
