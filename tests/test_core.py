"""Tests for the core module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.unimplemented_player import UnimplementedPlayer

if TYPE_CHECKING:
    from pathlib import Path

    from mad_world.players import GamePlayer


from mad_world.actions import (
    BaseAction,
    BiddingAction,
    ChatAction,
    InvalidActionError,
    InvalidBiddingActionError,
    InvalidChannelRequestError,
    MessagingAction,
    OperationsAction,
)
from mad_world.core import (
    GameState,
    format_results,
    game_loop,
    get_bid_impact,
    iterate_game,
    resolve_chat_channel,
    resolve_operation,
    resolve_operations,
    resolve_round_events,
)
from mad_world.crises import (
    SANCTIONS_TIE_GDP_EFFECT,
    DoomsdayAsteroidCrisis,
    DoomsdayAsteroidDefs,
    InternationalSanctionsCrisis,
)
from mad_world.enums import GameOverReason, GamePhase, OpenChannelPreference
from mad_world.events import (
    ActionEvent,
    ActorKind,
    ChannelOpenedEvent,
    OperationConductedEvent,
    PlayerActor,
    SystemActor,
    SystemEvent,
)
from mad_world.rules import GameRules, OperationDefinition
from mad_world.trivial_players import (
    Capitalist,
    CrazyIvan,
    Diplomat,
    Pacifist,
    ParetoEfficientPlayer,
    Saboteur,
)


@dataclass
class Scenario:
    alpha: type[GamePlayer]
    omega: type[GamePlayer]
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
    # Empty mandate deck so mandates don't unexpectedly save or end worlds in
    # these generic baseline scenario tests
    stable_rules.initial_mandate_deck = []

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


@pytest.mark.asyncio
async def test_resolve_chat_channel(basic_game: GameState) -> None:
    alpha_msg = MessagingAction(
        message_to_opponent="Let's talk",
        channel_preference=OpenChannelPreference.REQUEST,
    )
    omega_msg = MessagingAction(
        message_to_opponent="Sure",
        channel_preference=OpenChannelPreference.ACCEPT,
    )

    class TestAlphaPlayer(UnimplementedPlayer):
        async def chat(
            self, game: GameState, remaining_messages: int
        ) -> ChatAction:
            return ChatAction(
                message=f"Alpha chat {remaining_messages}", end_channel=False
            )

    class TestOmegaPlayer(UnimplementedPlayer):
        async def chat(
            self, game: GameState, remaining_messages: int
        ) -> ChatAction:
            return ChatAction(message="Omega done", end_channel=True)

    players: list[GamePlayer] = [
        TestAlphaPlayer("Alpha"),
        TestOmegaPlayer("Omega"),
    ]
    await resolve_chat_channel(basic_game, players, alpha_msg, omega_msg)

    assert basic_game.players["Alpha"].channels_opened == 1
    assert basic_game.players["Omega"].channels_opened == 0

    events = basic_game.event_log
    assert any(isinstance(e, ChannelOpenedEvent) for e in events)


@pytest.mark.asyncio
async def test_resolve_chat_channel_double_request(
    basic_game: GameState,
) -> None:
    alpha_msg = MessagingAction(
        channel_preference=OpenChannelPreference.REQUEST,
    )
    omega_msg = MessagingAction(
        channel_preference=OpenChannelPreference.REQUEST,
    )

    class TestPlayer(UnimplementedPlayer):
        async def chat(
            self, game: GameState, remaining_messages: int
        ) -> ChatAction:
            return ChatAction(message="test", end_channel=True)

    players: list[GamePlayer] = [TestPlayer("Alpha"), TestPlayer("Omega")]
    await resolve_chat_channel(basic_game, players, alpha_msg, omega_msg)

    assert basic_game.players["Alpha"].channels_opened == 1
    assert basic_game.players["Omega"].channels_opened == 1


@pytest.mark.asyncio
async def test_resolve_chat_channel_omega_request(
    basic_game: GameState,
) -> None:
    alpha_msg = MessagingAction(
        channel_preference=OpenChannelPreference.ACCEPT,
    )
    omega_msg = MessagingAction(
        channel_preference=OpenChannelPreference.REQUEST,
    )

    class TestPlayer(UnimplementedPlayer):
        async def chat(
            self, game: GameState, remaining_messages: int
        ) -> ChatAction:
            return ChatAction(message="test", end_channel=True)

    players: list[GamePlayer] = [TestPlayer("Alpha"), TestPlayer("Omega")]
    await resolve_chat_channel(basic_game, players, alpha_msg, omega_msg)

    assert basic_game.players["Alpha"].channels_opened == 0
    assert basic_game.players["Omega"].channels_opened == 1


@pytest.mark.asyncio
async def test_resolve_chat_channel_max_messages(
    basic_game: GameState,
) -> None:
    alpha_msg = MessagingAction(
        channel_preference=OpenChannelPreference.REQUEST,
    )
    omega_msg = MessagingAction(
        channel_preference=OpenChannelPreference.ACCEPT,
    )

    class TestPlayer(UnimplementedPlayer):
        async def chat(
            self, game: GameState, remaining_messages: int
        ) -> ChatAction:
            return ChatAction(message="test", end_channel=False)

    players: list[GamePlayer] = [TestPlayer("Alpha"), TestPlayer("Omega")]
    await resolve_chat_channel(basic_game, players, alpha_msg, omega_msg)

    assert basic_game.players["Alpha"].channels_opened == 1
    assert basic_game.players["Omega"].channels_opened == 0

    events = basic_game.event_log
    assert any(
        "The communication channel was closed after reaching the limit."
        in e.description
        for e in events
    )


@pytest.mark.asyncio
async def test_resolve_chat_channel_no_consent(basic_game: GameState) -> None:
    alpha_msg = MessagingAction(
        channel_preference=OpenChannelPreference.REQUEST,
    )
    omega_msg = MessagingAction(
        channel_preference=OpenChannelPreference.REJECT,
    )

    class TestPlayer(UnimplementedPlayer):
        async def chat(
            self, game: GameState, remaining_messages: int
        ) -> ChatAction:
            return ChatAction(message="Never called", end_channel=True)

    players: list[GamePlayer] = [TestPlayer("Alpha"), TestPlayer("Omega")]
    await resolve_chat_channel(basic_game, players, alpha_msg, omega_msg)

    assert basic_game.players["Alpha"].channels_opened == 0


def test_messaging_action_validate_semantics_invalid_channel_request(
    basic_game: GameState,
) -> None:
    basic_game.rules.max_channels_per_game = 0
    action = MessagingAction(channel_preference=OpenChannelPreference.REQUEST)
    with pytest.raises(InvalidChannelRequestError):
        action.validate_semantics(basic_game, "Alpha")


@pytest.mark.parametrize("bid", [0, 1, 3, 5, 10])
def test_bidding_action_validate_semantics_valid(
    basic_game: GameState, bid: int
) -> None:
    action = BiddingAction(bid=bid)
    # Should not raise
    action.validate_semantics(basic_game, "Alpha")


def test_bidding_action_validate_semantics_invalid_bid(
    basic_game: GameState,
) -> None:
    basic_game.rules.allowed_bids = [0, 1, 2, 3]
    action = BiddingAction(bid=4)
    with pytest.raises(InvalidBiddingActionError, match="INVALID BID"):
        action.validate_semantics(basic_game, "Alpha")


def test_operations_action_validate_semantics_valid(
    basic_game: GameState,
) -> None:
    # Alpha has 5 Inf. domestic-investment (3) + aggressive-extraction (2) = 5
    action = OperationsAction(
        operations=["domestic-investment", "aggressive-extraction"],
    )
    # Should not raise
    action.validate_semantics(basic_game, "Alpha")


def test_operations_action_validate_semantics_insufficient_influence(
    basic_game: GameState,
) -> None:
    domestic_investment = basic_game.rules.allowed_operations[
        "domestic-investment"
    ].model_copy(update={"influence_cost": 3})
    basic_game.rules.allowed_operations = {
        "domestic-investment": domestic_investment
    }
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
    assert isinstance(event.actor, PlayerActor)
    assert event.actor.name == "Alpha"


def test_game_event_defaults() -> None:
    event = SystemEvent(description="Test event")
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
    # Set the 'last' state to what we're looking for
    basic_game.last_round = 2
    basic_game.last_phase = GamePhase.BIDDING

    # Older event (should be skipped by break)
    basic_game.event_log.append(
        SystemEvent(
            description="old",
            current_round=1,
            current_phase=GamePhase.OPERATIONS,
        ),
    )
    # Target events
    basic_game.event_log.append(
        SystemEvent(
            description="target1",
            current_round=2,
            current_phase=GamePhase.BIDDING,
        )
    )
    basic_game.event_log.append(
        SystemEvent(
            description="target2",
            current_round=2,
            current_phase=GamePhase.BIDDING,
        )
    )
    # Newer events (should be skipped initially)
    basic_game.event_log.append(
        SystemEvent(
            description="new",
            current_round=2,
            current_phase=GamePhase.OPERATIONS,
        ),
    )

    recent = basic_game.recent_events()
    assert [e.description for e in recent] == ["target1", "target2"]


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
    stable_rules.initial_mandate_deck = []
    stable_rules.aggressor_tax_clock_threshold = 100

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
    # Because mandates are dealt randomly, CoolerHeadsMandate might trigger.
    # We emptied the deck to avoid test flakiness.
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
    basic_game.apply_event(SystemEvent(description="", clock_delta=-1))
    assert basic_game.doomsday_clock == 0

    basic_game.apply_event(
        SystemEvent(
            description="",
            clock_delta=basic_game.rules.max_clock_state + 1,
        ),
    )
    assert basic_game.doomsday_clock == basic_game.rules.max_clock_state


def test_escalation_debt(basic_game: GameState) -> None:
    # Alpha escalates by 5
    basic_game.apply_event(
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha escalates",
            clock_delta=5,
        )
    )
    assert basic_game.escalation_debt("Alpha") == 5
    assert basic_game.escalation_debt("Omega") == 0

    # Alpha de-escalates by 8 -> net -3. Should spill over to Omega.
    basic_game.apply_event(
        ActionEvent(
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
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha de-escalates again",
            clock_delta=-2,
        )
    )
    assert basic_game.escalation_debt("Alpha") == 0
    assert basic_game.escalation_debt("Omega") == 0

    # Omega escalates
    basic_game.apply_event(
        ActionEvent(
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
    stable_rules.aggressor_tax_clock_threshold = 100

    # Empty mandate deck so mandates don't interfere with this test
    stable_rules.initial_mandate_deck = []
    stable_rules.initial_mandate_deck = []

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


def test_crisis_trigger_logging() -> None:
    """Test that a crisis trigger is correctly logged."""
    rules = GameRules(max_clock_state=10, initial_mandate_deck=[])
    game = GameState.new_game(rules=rules, players=["Alpha", "Omega"])

    # Manually set the clock to trigger a crisis on next advance_phase
    game.escalate(SystemActor(), 10)

    # The crisis deck should have at least one crisis
    assert len(game.crisis_deck) > 0

    game.advance_phase()

    assert game.pending_crisis is not None
    crisis_title = game.pending_crisis.title

    # Check the event in the log. advance_phase appends a state description
    # event at the end, so the crisis event should be second to last.
    crisis_event = game.event_log[-2]

    assert isinstance(crisis_event.actor, SystemActor)
    expected_description = (
        "Time has run out and a global crisis has been "
        f"triggered: {crisis_title}"
    )
    assert crisis_event.description == expected_description

    # Also check the very last event is indeed the state description
    last_event = game.event_log[-1]
    assert "ROUND" in last_event.description
    assert "PHASE" in last_event.description
    assert last_event.secret is True


@pytest.mark.asyncio
async def test_autosave(tmp_path: Path) -> None:
    rules = GameRules()
    players = ["alpha", "omega"]
    game = GameState.new_game(rules=rules, players=players, log_dir=tmp_path)
    await game.autosave()
    assert (tmp_path / "game_state.json").exists()


@pytest.mark.asyncio
@patch("anyio.Path.write_text", new_callable=AsyncMock)
async def test_autosave_no_log_dir(mock_write: AsyncMock) -> None:
    rules = GameRules()
    players = ["alpha", "omega"]
    game = GameState.new_game(rules=rules, players=players, log_dir=None)
    await game.autosave()
    assert not mock_write.called


@pytest.mark.asyncio
@patch("anyio.Path.write_text", new_callable=AsyncMock)
async def test_autosave_exception(
    mock_write: AsyncMock, tmp_path: Path
) -> None:
    rules = GameRules()
    players = ["alpha", "omega"]
    game = GameState.new_game(rules=rules, players=players, log_dir=tmp_path)

    mock_write.side_effect = OSError()

    with patch.object(
        logging.getLogger("mad_world"), "exception", new_callable=MagicMock
    ) as log_except:
        await game.autosave()
        assert log_except.called
        assert log_except.called


@pytest.mark.asyncio
async def test_operations_negative_influence_resolution(
    basic_game: GameState,
) -> None:
    # Give both players 3 influence
    basic_game.players["Alpha"].influence = basic_game.allowed_operations[
        "stand-down"
    ].influence_cost
    basic_game.players["Omega"].influence = basic_game.allowed_operations[
        "domestic-investment"
    ].influence_cost

    # If stand-down is rebalanced to remove the enemy influence penalty this
    # test will need to be updated.
    assert (
        basic_game.allowed_operations["stand-down"].enemy_influence_effect <= -1
    )

    alpha_action = OperationsAction(operations=["domestic-investment"])
    omega_action = OperationsAction(operations=["stand-down"])

    class TestAlphaPlayer(Diplomat):
        async def operations(self, game: GameState) -> OperationsAction:
            return alpha_action

    class TestOmegaPlayer(Diplomat):
        async def operations(self, game: GameState) -> OperationsAction:
            return omega_action

    new_game = await resolve_operations(
        basic_game,
        [TestAlphaPlayer(name="alpha"), TestOmegaPlayer(name="omega")],
    )

    # Validate Omega's operation was rejected
    # The last 3 events should be:
    # 1. Omega's stand-down
    # 2. Alpha's rejected operation
    # 3. System message advancing phase

    events = new_game.event_log
    rejected_event = next(
        e for e in reversed(events) if "rejected" in e.description
    )

    assert "Alpha" in rejected_event.description
    assert "domestic-investment" in rejected_event.description
    assert "INSUFFICIENT INFLUENCE" in rejected_event.description


def test_resolve_operation_diplomatic_maneuvering(
    basic_game: GameState,
) -> None:
    basic_game.players["Alpha"].influence = 1
    assert basic_game.doomsday_clock == 0
    assert basic_game.rules.max_clock_state > 4

    # When Alpha has an escalation token
    basic_game.escalate(PlayerActor(name="Alpha"), 1)
    basic_game.escalate(PlayerActor(name="Omega"), 1)
    basic_game.escalate(SystemActor(), 1)
    event = resolve_operation(
        basic_game, "Alpha", "Omega", "diplomatic-maneuvering"
    )
    assert isinstance(event, OperationConductedEvent)
    assert event.shift_blame == (PlayerActor(name="Omega"), 1)
    assert basic_game.doomsday_clock == 3

    # When Alpha only has a system token to swap
    basic_game.reset_escalation()
    assert basic_game.doomsday_clock == 0
    basic_game.escalate(PlayerActor(name="Omega"), 1)
    basic_game.escalate(SystemActor(), 1)
    assert basic_game.doomsday_clock == 2

    event2 = resolve_operation(
        basic_game, "Alpha", "Omega", "diplomatic-maneuvering"
    )
    assert isinstance(event2, OperationConductedEvent)
    assert event2.shift_blame == (PlayerActor(name="Omega"), 1)
    assert basic_game.doomsday_clock == 2

    # Test apply_event for the swap
    basic_game.apply_event(event2)
    assert basic_game.escalation_debt("Omega") == 2
    assert basic_game.doomsday_clock == 2


def test_resolve_operation_diplomatic_maneuvering_no_tokens(
    basic_game: GameState,
) -> None:
    basic_game.players["Alpha"].influence = 1

    assert basic_game.doomsday_clock == 0
    assert basic_game.rules.max_clock_state > 3
    basic_game.escalate(PlayerActor(name="Omega"), 2)

    # When Alpha has no token, and no system token exists
    event = resolve_operation(
        basic_game, "Alpha", "Omega", "diplomatic-maneuvering"
    )
    assert isinstance(event, OperationConductedEvent)
    # The event resolves with a system swap fallback, but because
    # the system cube isn't on the track, the swap will do nothing when applied.
    assert event.shift_blame == (PlayerActor(name="Omega"), 1)
    assert event.influence_delta["Alpha"] >= -1
    assert basic_game.doomsday_clock == 2


def test_apply_event_track_swap_not_found(basic_game: GameState) -> None:
    # Test that apply_event gracefully handles when the actor to swap
    # is not actually found in the track.
    basic_game.reset_escalation()
    assert basic_game.doomsday_clock == 0
    basic_game.escalate(PlayerActor(name="Omega"), 1)
    assert basic_game.doomsday_clock == 1

    event = ActionEvent(
        actor=PlayerActor(name="Alpha"),
        description="test",
        shift_blame=(PlayerActor(name="Omega"), 2),
    )
    basic_game.apply_event(event)

    assert basic_game.doomsday_clock == 1
    assert basic_game.escalation_debt("Omega") == 1


def test_resolve_operation_defaults(basic_game: GameState) -> None:
    basic_game.rules.allowed_operations = {
        "dummy": OperationDefinition(
            name="dummy", description="desc", influence_cost=0
        )
    }

    result = resolve_operation(basic_game, "Alpha", "Omega", "dummy")
    assert result.clock_delta == 0
    assert not result.gdp_delta
    assert not result.influence_delta
    assert not result.secret
    assert not result.world_ending
    assert not result.new_effects
    assert not result.shift_blame


def test_resolve_operation_scaling_rewards(basic_game: GameState) -> None:
    basic_game.rules.escalation_reward_clock_threshold = 20
    basic_game.rules.escalation_reward_gdp = 3

    # Below threshold: no scaling
    basic_game.escalation_track = [SystemActor()] * 19
    event1 = resolve_operation(
        basic_game, "Alpha", "Omega", "aggressive-extraction"
    )
    assert event1.gdp_delta["Alpha"] == 3  # op_def.friendly_gdp_effect
    assert event1.gdp_delta.get("Omega", 0) == 0  # op_def.enemy_gdp_effect

    # At/Above threshold: scaling applied (clock_effect = 1 > 0)
    basic_game.escalation_track = [SystemActor()] * 20
    event2 = resolve_operation(
        basic_game, "Alpha", "Omega", "aggressive-extraction"
    )
    assert event2.gdp_delta["Alpha"] == 6  # 3 + 3
    assert event2.gdp_delta["Omega"] == -3  # 0 - 3

    # At/Above threshold, but clock_effect <= 0: no scaling
    event3 = resolve_operation(basic_game, "Alpha", "Omega", "stand-down")
    assert event3.gdp_delta["Alpha"] == -5
    assert event3.gdp_delta.get("Omega", 0) == 0


@pytest.mark.asyncio
async def test_resolve_round_events_aggressor_tax(
    basic_game: GameState,
) -> None:
    basic_game.rules.aggressor_tax_clock_threshold = 20
    basic_game.rules.aggressor_tax_inf_cost = 1
    basic_game.rules.aggressor_tax_gdp_cost = 2

    # Below threshold: no tax
    basic_game.escalation_track = [SystemActor()] * 19
    new_game = await resolve_round_events(basic_game)
    events = new_game.recent_events()
    assert not any("Aggressor Tax applied" in e.description for e in events)

    # At threshold, alpha debt > omega debt: Alpha pays Inf
    basic_game.escalation_track = [SystemActor()] * 20
    basic_game.escalation_track[0] = PlayerActor(name="Alpha")
    basic_game.players["Alpha"].influence = 2

    new_game = await resolve_round_events(basic_game)
    events = new_game.recent_events()
    tax_event = next(
        e for e in events if "Aggressor Tax applied" in e.description
    )
    assert "Alpha" in tax_event.description
    assert tax_event.influence_delta["Alpha"] == -1
    assert tax_event.gdp_delta["Alpha"] == 0

    # Alpha has 0 influence: Alpha pays GDP
    basic_game.players["Alpha"].influence = 0
    new_game = await resolve_round_events(basic_game)
    events = new_game.recent_events()
    tax_event = next(
        e for e in events if "Aggressor Tax applied" in e.description
    )
    assert tax_event.influence_delta["Alpha"] == 0
    assert tax_event.gdp_delta["Alpha"] == -2

    # Tied debt: both pay
    basic_game.escalation_track[1] = PlayerActor(name="Omega")
    basic_game.players["Alpha"].influence = 5
    basic_game.players["Omega"].influence = 5
    new_game = await resolve_round_events(basic_game)
    events = new_game.recent_events()
    tax_events = [e for e in events if "Aggressor Tax applied" in e.description]
    assert len(tax_events) == 2
