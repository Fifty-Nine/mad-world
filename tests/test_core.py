"""Tests for the core module."""

from mad_world.core import GameOverReason, GameRules, game_loop
from mad_world.trivial_players import CrazyIvan


def test_oops_all_ivans() -> None:
    assert game_loop(GameRules(), [CrazyIvan("Alpha"), CrazyIvan("Omega")])[
        :-1
    ] == (
        None,
        GameOverReason.WORLD_DESTROYED,
    )
