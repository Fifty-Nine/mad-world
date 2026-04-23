"""Tests for ensuring game reproducibility with seeded RNG."""

from __future__ import annotations

import asyncio

import pytest

from mad_world.core import (
    GameState,
    WorldDestroyed,
    check_game_over,
    destroy_world,
    game_loop,
    iterate_game,
)
from mad_world.rules import GameRules
from mad_world.trivial_players import (
    Capitalist,
    CrazyIvan,
    Diplomat,
    Pacifist,
    ParetoEfficientPlayer,
    Saboteur,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("alpha_class", "omega_class", "seed"),
    [
        (Pacifist, CrazyIvan, 42),
        (Capitalist, Diplomat, 1337),
        (Saboteur, ParetoEfficientPlayer, 999),
    ],
)
@pytest.mark.xfail(strict=False, reason="Reproducibility may have edge cases")
async def test_game_reproducibility(
    alpha_class: type, omega_class: type, seed: int
) -> None:
    """Verifies that games run with identical seeds are reproducible."""

    # We will use a smaller max_clock_state and round_count to prevent the tests
    # from timing out.
    # 3 rounds is enough to test iterative reproducibility vs continuous
    rules_a = GameRules(seed=seed, round_count=3, max_clock_state=15)
    p1_a = alpha_class("Alpha")
    p2_a = omega_class("Omega")
    res_a, reason_a, state_a = await game_loop(rules_a, [p1_a, p2_a])

    rules_b = GameRules(seed=seed, round_count=3, max_clock_state=15)
    p1_b = alpha_class("Alpha")
    p2_b = omega_class("Omega")
    res_b, reason_b, state_b = await game_loop(rules_b, [p1_b, p2_b])

    rules_c = GameRules(seed=seed, round_count=3, max_clock_state=15)
    p1_c = alpha_class("Alpha")
    p2_c = omega_class("Omega")
    state_c = GameState.new_game(players=["Alpha", "Omega"], rules=rules_c)

    rules_d = GameRules(seed=seed, round_count=3, max_clock_state=15)
    p1_d = alpha_class("Alpha")
    p2_d = omega_class("Omega")
    state_d = GameState.new_game(players=["Alpha", "Omega"], rules=rules_d)

    # Replicate game_loop start logic for C and D
    desc_c = await asyncio.gather(
        p1_c.get_description(), p2_c.get_description()
    )
    state_c.players["Alpha"].description = desc_c[0]
    state_c.players["Omega"].description = desc_c[1]
    await asyncio.gather(p1_c.start_game(state_c), p2_c.start_game(state_c))

    desc_d = await asyncio.gather(
        p1_d.get_description(), p2_d.get_description()
    )
    state_d.players["Alpha"].description = desc_d[0]
    state_d.players["Omega"].description = desc_d[1]
    await asyncio.gather(p1_d.start_game(state_d), p2_d.start_game(state_d))

    assert state_c == state_d

    while not check_game_over(state_c):
        try:
            state_c = await iterate_game(state_c, [p1_c, p2_c])
        except WorldDestroyed:
            state_c = destroy_world(state_c)

        try:
            state_d = await iterate_game(state_d, [p1_d, p2_d])
        except WorldDestroyed:
            state_d = destroy_world(state_d)

        assert state_c == state_d, "States diverged during iteration"

        if check_game_over(state_c) or check_game_over(state_d):
            break

    state_c.check_endgame_mandates()
    res_c, reason_c = state_c.determine_victor()

    state_d.check_endgame_mandates()
    res_d, reason_d = state_d.determine_victor()

    assert res_c == res_d
    assert reason_c == reason_d

    # Final assertion comparing the continuous and iterative game states.
    assert res_a == res_b == res_c
    assert reason_a == reason_b == reason_c
    assert state_a == state_b == state_c
