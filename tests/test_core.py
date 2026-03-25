"""Tests for the core module."""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from mad_world.core import GameOverReason, GamePlayer, game_loop
from mad_world.rules import GameRules
from mad_world.trivial_players import (
    Capitalist,
    CrazyIvan,
    Diplomat,
    Pacifist,
    Saboteur,
)


@dataclass
class Scenario:
    alpha: Callable[[str], GamePlayer]
    omega: Callable[[str], GamePlayer]
    winner: str | None
    reason: GameOverReason


TEST_CASES = [
    Scenario(CrazyIvan, CrazyIvan, None, GameOverReason.WORLD_DESTROYED),
    Scenario(Pacifist, Pacifist, None, GameOverReason.STALEMATE),
    Scenario(Capitalist, Capitalist, None, GameOverReason.WORLD_DESTROYED),
    Scenario(CrazyIvan, Pacifist, None, GameOverReason.WORLD_DESTROYED),
    Scenario(CrazyIvan, Capitalist, None, GameOverReason.WORLD_DESTROYED),
    Scenario(Pacifist, Capitalist, "Omega", GameOverReason.ECONOMIC_VICTORY),
    Scenario(Saboteur, Pacifist, "Alpha", GameOverReason.ECONOMIC_VICTORY),
    Scenario(Saboteur, Capitalist, None, GameOverReason.WORLD_DESTROYED),
    Scenario(Saboteur, Saboteur, None, GameOverReason.WORLD_DESTROYED),
    Scenario(Saboteur, CrazyIvan, None, GameOverReason.WORLD_DESTROYED),
    Scenario(Diplomat, Pacifist, None, GameOverReason.STALEMATE),
    Scenario(Diplomat, Capitalist, None, GameOverReason.WORLD_DESTROYED),
    Scenario(Diplomat, Saboteur, "Omega", GameOverReason.ECONOMIC_VICTORY),
    Scenario(Diplomat, CrazyIvan, None, GameOverReason.WORLD_DESTROYED),
    Scenario(Diplomat, Diplomat, None, GameOverReason.STALEMATE),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    TEST_CASES,
    ids=[f"{tc.alpha.__name__}_vs_{tc.omega.__name__}" for tc in TEST_CASES],
)
async def test_game_outcomes(scenario: Scenario) -> None:
    winner, reason, _event_log = await game_loop(
        GameRules(), [scenario.alpha("Alpha"), scenario.omega("Omega")]
    )

    assert winner == scenario.winner
    assert reason == scenario.reason
