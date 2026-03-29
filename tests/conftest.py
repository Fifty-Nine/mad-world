"""Shared pytest fixtures."""

import pytest

from mad_world.core import GameState, PlayerState
from mad_world.enums import GamePhase
from mad_world.rules import GameRules


@pytest.fixture
def basic_game() -> GameState:
    """Provides a basic game state for testing."""
    rules = GameRules()
    players = {
        "Alpha": PlayerState(name="Alpha", gdp=50, influence=5),
        "Omega": PlayerState(name="Omega", gdp=50, influence=5),
    }
    return GameState(
        players=players,
        rules=rules,
        doomsday_clock=0,
        current_round=1,
        current_phase=GamePhase.BIDDING,
    )
