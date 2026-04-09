from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mad_world.core import GameManager, check_game_over
from mad_world.enums import GamePhase
from mad_world.rules import GameRules
from mad_world.trivial_players import CrazyIvan

if TYPE_CHECKING:
    from mad_world.events import SystemActor
    from mad_world.players import GamePlayer
from typing import cast


@pytest.mark.asyncio
async def test_game_manager_basic_run() -> None:
    rules = GameRules(max_clock_state=2, round_count=2, initial_clock_state=1)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]

    manager = GameManager(rules, players)
    await manager.start()

    while not manager.game_over:
        await manager.step()

    assert manager.game_over is True
    assert manager.reason is not None


@pytest.mark.asyncio
async def test_game_manager_step_after_over() -> None:
    rules = GameRules(max_clock_state=2, round_count=1, initial_clock_state=1)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]

    manager = GameManager(rules, players)
    await manager.start()

    while not manager.game_over:
        await manager.step()

    assert manager.game_over is True

    # Extra step does nothing
    await manager.step()
    assert manager.game_over is True


@pytest.mark.asyncio
async def test_game_manager_world_destroyed() -> None:
    # Setup for quick world destruction
    rules = GameRules(max_clock_state=1, round_count=2, initial_clock_state=1)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]

    manager = GameManager(rules, players)
    await manager.start()

    while not manager.game_over:
        await manager.step()

    assert manager.game_over is True


@pytest.mark.asyncio
async def test_game_manager_game_over_middle_of_stepping() -> None:
    # Make a game that is almost over, but not quite
    rules = GameRules(max_clock_state=2, round_count=1, initial_clock_state=1)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]

    manager = GameManager(rules, players)
    await manager.start()

    # We step the game until the state becomes game over but before we re-check
    # By manipulating the current round to end the game on the next step
    manager.game.current_round = rules.round_count

    # This step should advance the game, notice game is over, and call _handle_game_over  # noqa: E501
    await manager.step()

    # It might take a few steps to reach GamePhase check condition
    while not manager.game_over:
        manager.game.current_round = rules.round_count + 1
        await manager.step()

    assert manager.game_over is True


@pytest.mark.asyncio
async def test_game_manager_game_over_immediately_after_iterate() -> None:
    rules = GameRules(max_clock_state=2, round_count=1, initial_clock_state=1)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]

    manager = GameManager(rules, players)
    await manager.start()

    # Advance the phase to something that does not trigger crisis, but then force game over by current round  # noqa: E501
    manager.game.current_round = rules.round_count + 1

    await manager.step()
    assert manager.game_over is True


@pytest.mark.asyncio
async def test_game_manager_game_over_from_iteration() -> None:
    rules = GameRules(max_clock_state=2, round_count=1, initial_clock_state=1)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]

    manager = GameManager(rules, players)
    await manager.start()

    # Setup so that after iterating, check_game_over is True
    # If the phase is Opening, it won't check game over until later.
    # To hit line 802, the top check_game_over MUST be False,
    # then iterate_game runs and returns a game where check_game_over is True.
    # We can do this by fast forwarding to GamePhase.ROUND_EVENTS (where it increments round)  # noqa: E501
    manager.game.current_phase = GamePhase.OPERATIONS
    manager.game.current_round = rules.round_count

    # Operations will advance the phase to ROUND_EVENTS and increment current_round!  # noqa: E501
    await manager.step()

    assert manager.game_over is True


@pytest.mark.asyncio
async def test_game_manager_game_over_after_iterate_only() -> None:
    rules = GameRules(max_clock_state=2, round_count=2, initial_clock_state=1)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]

    manager = GameManager(rules, players)
    await manager.start()

    # We want line 802 to run!
    # Line 790 is: if check_game_over(self.game): return
    # Line 801 is: if check_game_over(self.game): await self._handle_game_over()
    # So we must make check_game_over FALSE before iterate_game
    # and TRUE after iterate_game!

    manager.game.current_round = rules.round_count
    manager.game.current_phase = GamePhase.OPERATIONS
    # operations changes to ROUND_EVENTS and current_round += 1
    # which makes check_game_over TRUE

    assert check_game_over(manager.game) is False
    await manager.step()
    assert check_game_over(manager.game) is True
    assert manager.game_over is True


@pytest.mark.asyncio
async def test_game_manager_game_over_after_iterate_only_proper_fix() -> None:
    rules = GameRules(max_clock_state=5, round_count=1, initial_clock_state=0)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]

    manager = GameManager(rules, players)
    await manager.start()

    manager.game.current_phase = GamePhase.OPERATIONS
    manager.game.current_round = 1

    # We remove clock to avoid triggering a crisis when advancing phase
    manager.game.escalate(
        cast("SystemActor", manager.game.escalation_track[0]),
        -manager.game.doomsday_clock,
    )

    assert check_game_over(manager.game) is False

    await manager.step()

    assert check_game_over(manager.game) is True
    assert manager.game_over is True


@pytest.mark.asyncio
async def test_game_manager_game_over_after_iterate_only_target() -> None:
    rules = GameRules(max_clock_state=2, round_count=1, initial_clock_state=1)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]

    manager = GameManager(rules, players)
    await manager.start()

    # Fast forward to the last phase before incrementing round
    manager.game.current_phase = GamePhase.OPERATIONS
    manager.game.current_round = 1

    # Reset clock so it doesn't trigger crisis
    manager.game.escalate(
        cast("SystemActor", manager.game.escalation_track[0]),
        -manager.game.doomsday_clock,
    )

    # We want iterate_game to make it GamePhase.ROUND_EVENTS and current_round=2
    # check_game_over logic:
    # game.current_round > game.rules.round_count
    # So 2 > 1 is True

    # Before step, check_game_over is False
    assert check_game_over(manager.game) is False

    # Because check_game_over is False, it will skip line 790 and go to iterate_game  # noqa: E501
    # iterate_game will return game where check_game_over is True
    # then line 802 will trigger
    await manager.step()

    assert manager.game_over is True


@pytest.mark.asyncio
async def test_game_manager_802_hit() -> None:
    rules = GameRules(max_clock_state=2, round_count=1, initial_clock_state=1)
    players: list[GamePlayer] = [CrazyIvan("P1"), CrazyIvan("P2")]
    manager = GameManager(rules, players)
    await manager.start()

    # Start in OPERATIONS, round 1.
    manager.game.current_phase = GamePhase.OPERATIONS

    # iterate_game will advance to ROUND_EVENTS and round 2.
    # At this exact point, check_game_over(self.game) will evaluate to True
    # Let's ensure doomsday_clock doesn't trigger crisis early
    manager.game.escalate(
        cast("SystemActor", manager.game.escalation_track[0]),
        -manager.game.doomsday_clock,
    )

    # Pre-condition: check_game_over is False, so we bypass line 790
    assert check_game_over(manager.game) is False

    # Perform step, iterate_game executes, self.game gets updated
    # check_game_over evaluates True at line 801, runs 802
    await manager.step()

    # Post-condition
    assert check_game_over(manager.game) is True
    assert manager.game_over is True
