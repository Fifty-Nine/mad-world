"""Shared pytest fixtures."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from mad_world.core import GameState
from mad_world.enums import GamePhase
from mad_world.rules import GameRules

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def basic_game() -> GameState:
    """Provides a basic game state for testing."""
    rules = GameRules()
    return GameState.new_game(
        players=["Alpha", "Omega"],
        rules=rules,
        current_round=1,
        current_phase=GamePhase.BIDDING,
    )


@pytest.fixture
def seeded_rng() -> random.Random:
    return random.Random(0)


@pytest.fixture
def stable_rng(
    seeded_rng: random.Random,
) -> Generator[random.Random, None, None]:
    def fixed_shuffle(values: list[Any]) -> None:
        values.sort(reverse=True)

    with patch.object(seeded_rng, "shuffle", side_effect=fixed_shuffle):
        yield seeded_rng
