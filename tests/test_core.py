"""Tests for the core module."""

from mad_world.core import CrazyIvan, GameOverReason, GameRules, game_loop


def test_oops_all_ivans() -> None:
    assert game_loop(GameRules(), [CrazyIvan("Alpha"), CrazyIvan("Omega")])[
        :-1
    ] == (
        None,
        GameOverReason.WORLD_DESTROYED,
    )
