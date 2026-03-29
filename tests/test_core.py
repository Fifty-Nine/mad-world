"""Tests for the core module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from mad_world.actions import (
    BaseAction,
    BiddingAction,
    InvalidActionError,
    InvalidBiddingActionError,
    OperationsAction,
)
from mad_world.core import (
    GameState,
    format_results,
    game_loop,
    iterate_game,
    process_bid,
    resolve_operation,
)
from mad_world.enums import GameOverReason, GamePhase
from mad_world.events import (
    ActorKind,
    GameEvent,
    PlayerActor,
    SystemActor,
)
from mad_world.rules import GameRules
from mad_world.trivial_players import (
    Capitalist,
    CrazyIvan,
    Diplomat,
    Pacifist,
    Saboteur,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mad_world.players import GamePlayer


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
    Scenario(Diplomat, Capitalist, "Omega", GameOverReason.ECONOMIC_VICTORY),
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
        GameRules(),
        [scenario.alpha("Alpha"), scenario.omega("Omega")],
    )

    assert winner == scenario.winner
    assert reason == scenario.reason


def test_validate_operation_invalid_name(basic_game: GameState) -> None:
    with pytest.raises(
        InvalidActionError,
        match="INVALID OPERATION: 'invalid-op' is not a valid operation",
    ):
        basic_game.validate_operation("invalid-op", "Alpha")


def test_validate_operation_insufficient_influence(
    basic_game: GameState,
) -> None:
    basic_game.players["Alpha"].influence = 2
    with pytest.raises(InvalidActionError, match="INSUFFICIENT INFLUENCE"):
        basic_game.validate_operation("domestic-investment", "Alpha")


def test_bidding_action_validate_semantics_invalid_bid(
    basic_game: GameState,
) -> None:
    basic_game.rules.allowed_bids = [0, 1, 2, 3]
    action = BiddingAction(bid=4)
    with pytest.raises(InvalidBiddingActionError, match="INVALID BID"):
        action.validate_semantics(basic_game, "Alpha")


def test_operations_action_validate_semantics_insufficient_influence(
    basic_game: GameState,
) -> None:
    basic_game.rules.allowed_operations[
        "domestic-investment"
    ].influence_cost = 3
    action = OperationsAction(
        operations=["domestic-investment", "domestic-investment"],
    )
    with pytest.raises(InvalidActionError, match="INSUFFICIENT INFLUENCE"):
        action.validate_semantics(basic_game, "Alpha")


def test_process_bid_invalid_bid(basic_game: GameState) -> None:
    basic_game.rules.allowed_bids = [0, 1]
    process_bid(basic_game, "Alpha", -999)
    assert basic_game.players["Alpha"].influence == 6
    assert basic_game.doomsday_clock == 1
    assert (
        "corrected to the maximum possible value"
        in basic_game.event_log[-1].description
    )


def test_resolve_operation_invalid_operation(basic_game: GameState) -> None:
    event = resolve_operation(basic_game, "Alpha", "Omega", "invalid-op")
    assert "was rejected" in event.description
    assert event.actor.actor_kind == ActorKind.PLAYER
    assert event.actor.name == "Alpha"


def test_game_event_defaults() -> None:
    event = GameEvent(actor=SystemActor(), description="Test event")
    assert event.clock_delta == 0
    assert event.gdp_delta == {}
    assert event.influence_delta == {}
    assert event.secret is False
    assert event.current_round is None
    assert event.current_phase is None


def test_describe_state_critical_clock(basic_game: GameState) -> None:
    basic_game.rules.max_clock_state = 30
    basic_game.doomsday_clock = 25
    description = basic_game.describe_state()
    assert "(CRITICAL)" in description


def test_iterate_game_end_phase(basic_game: GameState) -> None:
    basic_game.current_phase = GamePhase.END
    new_game = pytest.importorskip("asyncio").run(iterate_game(basic_game, []))
    assert new_game == basic_game


def test_actor_models() -> None:
    pa = PlayerActor(name="Test")
    assert pa.actor_kind == ActorKind.PLAYER
    assert pa.name == "Test"

    sa = SystemActor()
    assert sa.actor_kind == ActorKind.SYSTEM


def test_format_results(basic_game: GameState) -> None:
    res = format_results("Alpha", GameOverReason.ECONOMIC_VICTORY, basic_game)
    assert "Winner: Alpha" in res
    assert "Reason: ECONOMIC_VICTORY" in res

    res_mad = format_results(None, GameOverReason.WORLD_DESTROYED, basic_game)
    assert "Winner: no one" in res_mad
    assert "Reason: WORLD_DESTROYED" in res_mad
    assert "(before MAD)" in res_mad


def test_recent_events(basic_game: GameState) -> None:
    basic_game.last_round = 2
    basic_game.last_phase = GamePhase.BIDDING
    event = GameEvent(
        actor=SystemActor(),
        description="test",
        current_round=2,
        current_phase=GamePhase.BIDDING,
    )
    basic_game.event_log.append(event)
    basic_game.event_log.append(
        GameEvent(
            actor=SystemActor(),
            description="other",
            current_round=1,
            current_phase=GamePhase.OPERATIONS,
        ),
    )

    recent = basic_game.recent_events()
    assert len(recent) == 1
    assert recent[0].description == "test"


def test_base_action_validate_semantics(basic_game: GameState) -> None:
    action = BaseAction()
    action.validate_semantics(basic_game, "Alpha")
