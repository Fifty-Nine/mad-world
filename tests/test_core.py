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
    get_bid_impact,
    iterate_game,
    resolve_operation,
)
from mad_world.crises import (
    SANCTIONS_TIE_GDP_EFFECT,
    DoomsdayAsteroidCrisis,
    DoomsdayAsteroidDefs,
    InternationalSanctionsCrisis,
)
from mad_world.enums import GameOverReason, GamePhase
from mad_world.events import (
    ActorKind,
    GameEvent,
    PlayerActor,
    SystemActor,
)
from mad_world.trivial_players import (
    Capitalist,
    CrazyIvan,
    Diplomat,
    Pacifist,
    ParetoEfficientPlayer,
    Saboteur,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mad_world.players import GamePlayer
    from mad_world.rules import GameRules


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
    Scenario(
        ParetoEfficientPlayer,
        ParetoEfficientPlayer,
        None,
        GameOverReason.STALEMATE,
    ),
    Scenario(
        ParetoEfficientPlayer,
        CrazyIvan,
        None,
        GameOverReason.WORLD_DESTROYED,
    ),
    Scenario(
        ParetoEfficientPlayer,
        Pacifist,
        "Alpha",
        GameOverReason.ECONOMIC_VICTORY,
    ),
    Scenario(
        ParetoEfficientPlayer, Capitalist, None, GameOverReason.WORLD_DESTROYED
    ),
    Scenario(
        ParetoEfficientPlayer, Saboteur, None, GameOverReason.WORLD_DESTROYED
    ),
    Scenario(
        ParetoEfficientPlayer,
        Diplomat,
        "Alpha",
        GameOverReason.ECONOMIC_VICTORY,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    TEST_CASES,
    ids=[f"{tc.alpha.__name__}_vs_{tc.omega.__name__}" for tc in TEST_CASES],
)
async def test_game_outcomes(
    scenario: Scenario, stable_rules: GameRules
) -> None:
    winner, reason, _event_log = await game_loop(
        stable_rules,
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


def test_get_bid_impact_invalid_bid(basic_game: GameState) -> None:
    basic_game.rules.allowed_bids = [0, 1]
    bid, impact, desc = get_bid_impact(basic_game, "Alpha", -999)
    assert bid == 1
    assert impact == 1
    assert "corrected to the maximum possible value" in desc


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
    basic_game.escalation_track = [SystemActor()] * 25
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


@pytest.mark.asyncio
async def test_survived_crisis(stable_rules: GameRules) -> None:
    stable_rules.initial_clock_state = 29
    stable_rules.max_clock_state = 30
    winner, reason, _game = await game_loop(
        stable_rules,
        [Diplomat("Alpha"), Diplomat("Omega")],
    )

    assert winner is None
    assert reason == GameOverReason.STALEMATE


@pytest.mark.asyncio
async def test_international_sanctions_integration(
    stable_rules: GameRules,
) -> None:
    """Test an InternationalSanctionsCrisis through the full game loop."""
    stable_rules.initial_clock_state = 48
    stable_rules.max_clock_state = 50
    stable_rules.round_count = 1
    stable_rules.initial_crisis_deck = [InternationalSanctionsCrisis()]

    # Diplomats tend to bid 1.
    # Round 1:
    # BIDDING: Alpha bids 1, Omega bids 1.
    # Clock impact: 1 + 1 = 2.
    # Clock: 48 -> 50.
    # Crisis triggers!
    # Both players have 1 escalation token.
    # InternationalSanctionsCrisis (Tie): Clock -10, GDP -10 each.
    # Clock: 50 -> 40.
    # GDP: 50 -> 40.
    # After crisis, returns to OPERATIONS phase.
    # Clock is 40, so they can still do operations.
    # Diplomat has 5 + 1 = 6 influence.
    # unilateral-drawdown costs 6.
    # Round ends after operations.
    _, reason, game = await game_loop(
        stable_rules,
        [Diplomat("Alpha"), Diplomat("Omega")],
    )

    # Sanctions reduce clock/GDP by 10 each.
    # Drawdowns reduce clock by 9 each.
    assert game.doomsday_clock == 22
    assert (
        game.players["Alpha"].gdp
        == stable_rules.initial_gdp + SANCTIONS_TIE_GDP_EFFECT
    )
    assert (
        game.players["Omega"].gdp
        == stable_rules.initial_gdp + SANCTIONS_TIE_GDP_EFFECT
    )

    assert reason == GameOverReason.STALEMATE


def test_clock_limits(basic_game: GameState) -> None:
    """Ensure clock state can't go negative due to an event."""
    basic_game.apply_event(
        GameEvent(actor=SystemActor(), description="", clock_delta=-1)
    )
    assert basic_game.doomsday_clock == 0

    basic_game.apply_event(
        GameEvent(
            actor=SystemActor(),
            description="",
            clock_delta=basic_game.rules.max_clock_state + 1,
        ),
    )
    assert basic_game.doomsday_clock == basic_game.rules.max_clock_state


def test_escalation_debt(basic_game: GameState) -> None:
    # Alpha escalates by 5
    basic_game.apply_event(
        GameEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha escalates",
            clock_delta=5,
        )
    )
    assert basic_game.escalation_debt("Alpha") == 5
    assert basic_game.escalation_debt("Omega") == 0

    # Alpha de-escalates by 8 -> net -3. Should spill over to Omega.
    basic_game.apply_event(
        GameEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha de-escalates heavily",
            clock_delta=-8,
        )
    )
    assert basic_game.escalation_debt("Alpha") == 0
    assert basic_game.escalation_debt("Omega") == 0

    # Alpha de-escalates by 2 -> Omega is already 0, goes to -2.
    # Then clamped to 0.
    basic_game.apply_event(
        GameEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha de-escalates again",
            clock_delta=-2,
        )
    )
    assert basic_game.escalation_debt("Alpha") == 0
    assert basic_game.escalation_debt("Omega") == 0

    # Omega escalates
    basic_game.apply_event(
        GameEvent(
            actor=PlayerActor(name="Omega"),
            description="Omega escalates",
            clock_delta=4,
        )
    )
    assert basic_game.escalation_debt("Alpha") == 0
    assert basic_game.escalation_debt("Omega") == 4


def test_state_round_trip(basic_game: GameState) -> None:
    GameState.model_validate(basic_game.model_dump())


@pytest.mark.asyncio
async def test_doomsday_asteroid_integration(
    stable_rules: GameRules,
) -> None:
    """Test a DoomsdayAsteroidCrisis through the full game loop."""
    stable_rules.initial_clock_state = 48
    stable_rules.max_clock_state = 50
    stable_rules.round_count = 1
    stable_rules.initial_crisis_deck = [DoomsdayAsteroidCrisis()]

    # Diplomats tend to bid 1.
    # Round 1:
    # BIDDING: Alpha bids 1, Omega bids 1.
    # Clock impact: 1 + 1 = 2.
    # Clock: 48 -> 50.
    # Crisis triggers!
    # Both players are cautious, so they will default to 50% of threshold.
    # 50% of threshold = 15
    # Both bid 15. Sum is 30. Threshold is met.
    # Clock is reduced by 20.
    # 50 -> 30.
    # GDP is reduced by 15 each, but increased by WINNER_GDP // 2 = 5.
    # GDP: 50 -> 35 -> 40.
    # After crisis, returns to OPERATIONS phase.
    # Diplomat has 5 + 1 = 6 influence.
    # unilateral-drawdown costs 6.
    # Drawdowns reduce clock by 9 each.
    # Round ends after operations.
    _, reason, game = await game_loop(
        stable_rules,
        [Diplomat("Alpha"), Diplomat("Omega")],
    )

    assert game.doomsday_clock == 12
    assert game.players["Alpha"].gdp == stable_rules.initial_gdp - 15 + (
        DoomsdayAsteroidDefs.WINNER_GDP // 2
    )
    assert game.players["Omega"].gdp == stable_rules.initial_gdp - 15 + (
        DoomsdayAsteroidDefs.WINNER_GDP // 2
    )

    assert reason == GameOverReason.STALEMATE
