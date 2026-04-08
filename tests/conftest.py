"""Shared pytest fixtures."""

from __future__ import annotations

import builtins
import io
import os
import random
import tempfile
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from mad_world.core import GameState
from mad_world.crises import StandoffCrisis
from mad_world.enums import GamePhase
from mad_world.rules import GameRules

if TYPE_CHECKING:
    from collections.abc import Generator


class BadTestWriteException(Exception):
    def __init__(self, path: Any) -> None:
        super().__init__(f"Test tried to write to forbidden path: {path}")


def is_write_mode(mode: str) -> bool:
    return any(m in mode for m in "wa+x")


@cache
def is_tmpdir(file: Any) -> bool:
    if isinstance(file, int):
        return True

    try:
        path = Path(os.fspath(os.fsdecode(file))).resolve()
    except TypeError:
        return False

    temp_dir = Path(tempfile.gettempdir()).resolve()
    cache_dir = Path("./.pytest_cache").resolve()
    return (
        path.is_relative_to(temp_dir)
        or path == Path(os.devnull).resolve()
        or path.is_relative_to(cache_dir)
    )


@pytest.fixture(autouse=True)
def forbid_write(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_open = builtins.open

    def patched_open(
        file: Any, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if is_write_mode(mode) and not is_tmpdir(file):
            raise BadTestWriteException(file)

        return orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", patched_open)
    monkeypatch.setattr(io, "open", patched_open)


@pytest.fixture(autouse=True)
def mock_random(monkeypatch: pytest.MonkeyPatch) -> None:
    random.seed(0)

    class PatchedRandom(random.Random):
        def __init__(self, x: Any = 0) -> None:
            super().__init__(x)

    monkeypatch.setattr(random, "Random", PatchedRandom)


@pytest.fixture
def stable_rules() -> GameRules:
    return GameRules(
        seed=0,
        initial_crisis_deck=[StandoffCrisis()],
        initial_event_deck=[],
    )


@pytest.fixture
def basic_game(stable_rules: GameRules) -> GameState:
    """Provides a basic game state for testing."""
    return GameState.new_game(
        players=["Alpha", "Omega"],
        rules=stable_rules,
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
