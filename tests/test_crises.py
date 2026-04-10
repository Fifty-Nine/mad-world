"""Unit tests for crisis implementations."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, override
from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_world.actions import (
    BaseAction,
    InsufficientInfluenceError,
    InvalidInfluenceAmountError,
)
from mad_world.core import GameState, resolve_crisis
from mad_world.crises import (
    SANCTIONS_MIN_EFFECT,
    SANCTIONS_TIE_CLOCK_EFFECT,
    SANCTIONS_TIE_GDP_EFFECT,
    STANDOFF_LOSER_GDP_EFFECT,
    STANDOFF_LOSER_INF_EFFECT,
    STANDOFF_TIE_CLOCK_EFFECT,
    STANDOFF_TIE_GDP_EFFECT,
    STANDOFF_TIE_INF_EFFECT,
    STANDOFF_WINNER_CLOCK_EFFECT,
    BaseCrisis,
    GenericCrisis,
    InternationalSanctionsCrisis,
    ProxyWarAction,
    ProxyWarCrisis,
    ProxyWarDefs,
    StandoffAction,
    StandoffCrisis,
)
from mad_world.enums import StandoffPosture
from mad_world.events import (
    GameEvent,
    PlayerActor,
    SystemActor,
    SystemEvent,
)
from mad_world.rules import GameRules


class CrisisTestBase[T: BaseAction, C: GenericCrisis[Any]]:
    """Base class for testing crises."""

    @pytest.fixture
    def basic_game(self) -> GameState:
        return GameState.new_game(
            rules=GameRules(max_clock_state=100),
            players=["Alpha", "Omega"],
        )

    @pytest.fixture
    def crisis(self) -> C:
        raise NotImplementedError("Subclasses must provide a crisis fixture.")

    @pytest.mark.asyncio
    async def test_run_generic(
        self,
        crisis: C,
        basic_game: GameState,
    ) -> None:
        """Test the run method which gathers player actions."""
        p1 = MagicMock()
        p1.name = "Alpha"
        p1_action = crisis.get_default_action(
            p1.name, basic_game, aggressive=True
        )
        p1.crisis = AsyncMock(return_value=p1_action)

        p2 = MagicMock()
        p2.name = "Omega"
        p2_action = crisis.get_default_action(
            p2.name, basic_game, aggressive=False
        )
        p2.crisis = AsyncMock(return_value=p2_action)

        events = await crisis.run(basic_game, [p1, p2])

        assert isinstance(events, list)
        p1.crisis.assert_called_once_with(basic_game, crisis)
        p2.crisis.assert_called_once_with(basic_game, crisis)


class TestInternationalSanctionsCrisis:
    """Tests for the InternationalSanctionsCrisis implementation."""

    @pytest.fixture
    def crisis(self) -> InternationalSanctionsCrisis:
        return InternationalSanctionsCrisis()

    @pytest.fixture
    def basic_game(self) -> GameState:
        return GameState.new_game(
            rules=GameRules(max_clock_state=100),
            players=["Alpha", "Omega"],
        )

    @pytest.mark.asyncio
    async def test_run_tie(
        self,
        crisis: InternationalSanctionsCrisis,
        basic_game: GameState,
    ) -> None:
        """Test resolution when both players have equal debt."""
        basic_game.escalate(PlayerActor(name="Alpha"), 1)
        basic_game.escalate(PlayerActor(name="Omega"), 1)

        events = await crisis.run(basic_game, [])

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert isinstance(event.actor, SystemActor)
        assert event.clock_delta == SANCTIONS_TIE_CLOCK_EFFECT
        assert event.gdp_delta["Alpha"] == SANCTIONS_TIE_GDP_EFFECT
        assert event.gdp_delta["Omega"] == SANCTIONS_TIE_GDP_EFFECT

    @pytest.mark.asyncio
    async def test_run_p1_sanctioned(
        self,
        crisis: InternationalSanctionsCrisis,
        basic_game: GameState,
    ) -> None:
        """Test resolution when Player1 has more debt."""
        debt1, debt2 = 3, 1
        basic_game.escalate(PlayerActor(name="Alpha"), debt1)
        basic_game.escalate(PlayerActor(name="Omega"), debt2)

        events = await crisis.run(basic_game, [])

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert isinstance(event.actor, SystemActor)
        expected_amount = max(abs(debt1 - debt2), abs(SANCTIONS_MIN_EFFECT))
        assert event.clock_delta == -expected_amount
        assert event.gdp_delta["Alpha"] == -expected_amount
        assert "Omega" not in event.gdp_delta

    @pytest.mark.asyncio
    async def test_run_p2_sanctioned_large_debt(
        self,
        crisis: InternationalSanctionsCrisis,
        basic_game: GameState,
    ) -> None:
        """Test resolution when Player2 has much more debt."""
        debt1, debt2 = 1, 15
        basic_game.escalate(PlayerActor(name="Omega"), debt2)
        basic_game.escalate(PlayerActor(name="Alpha"), debt1)

        events = await crisis.run(basic_game, [])

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert isinstance(event.actor, SystemActor)
        expected_amount = max(abs(debt1 - debt2), abs(SANCTIONS_MIN_EFFECT))
        assert event.clock_delta == -expected_amount
        assert event.gdp_delta["Omega"] == -expected_amount
        assert "Alpha" not in event.gdp_delta


class TestStandoffCrisis(CrisisTestBase[StandoffAction, StandoffCrisis]):
    """Tests for the StandoffCrisis implementation."""

    @pytest.fixture
    def crisis(self) -> StandoffCrisis:
        return StandoffCrisis()

    def test_get_default_action(self, crisis: StandoffCrisis) -> None:
        """Test that default actions are correctly returned."""
        aggressive = crisis.get_default_action(
            MagicMock(), MagicMock(), aggressive=True
        )
        assert aggressive.posture == StandoffPosture.STAND_FIRM

        cautious = crisis.get_default_action(
            MagicMock(), MagicMock(), aggressive=False
        )
        assert cautious.posture == StandoffPosture.BACK_DOWN

    def test_resolve_doomsday(
        self,
        crisis: StandoffCrisis,
        basic_game: GameState,
    ) -> None:
        """Test resolution when both players stand firm."""
        actions = {
            "Alpha": StandoffAction(posture=StandoffPosture.STAND_FIRM),
            "Omega": StandoffAction(posture=StandoffPosture.STAND_FIRM),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event.actor, SystemActor)
        assert "nuclear exchange" in event.description
        assert event.world_ending

    def test_resolve_tie(
        self,
        crisis: StandoffCrisis,
        basic_game: GameState,
    ) -> None:
        """Test resolution when both players back down."""
        actions = {
            "Alpha": StandoffAction(posture=StandoffPosture.BACK_DOWN),
            "Omega": StandoffAction(posture=StandoffPosture.BACK_DOWN),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event.actor, SystemActor)
        assert "cooler heads" in event.description
        assert event.clock_delta == STANDOFF_TIE_CLOCK_EFFECT
        assert event.gdp_delta["Alpha"] == STANDOFF_TIE_GDP_EFFECT
        assert event.gdp_delta["Omega"] == STANDOFF_TIE_GDP_EFFECT
        assert event.influence_delta["Alpha"] == STANDOFF_TIE_INF_EFFECT
        assert event.influence_delta["Omega"] == STANDOFF_TIE_INF_EFFECT

    def test_resolve_tie_clamping(
        self,
        crisis: StandoffCrisis,
        basic_game: GameState,
    ) -> None:
        """Test resolution when both players back down, but clock is high."""
        basic_game.escalation_track = [SystemActor()] * 150
        actions = {
            "Alpha": StandoffAction(posture=StandoffPosture.BACK_DOWN),
            "Omega": StandoffAction(posture=StandoffPosture.BACK_DOWN),
        }

        events = crisis.resolve(basic_game, actions)
        assert len(events) == 1
        event = events[0]
        # Target clock is max_clock_state - 1.
        target_clock = basic_game.rules.max_clock_state - 1
        expected_delta = target_clock - basic_game.doomsday_clock
        assert event.clock_delta == expected_delta

    def test_resolve_winner_clamping(
        self,
        crisis: StandoffCrisis,
        basic_game: GameState,
    ) -> None:
        """Test resolution when one player stands firm, and clock is high."""
        basic_game.escalation_track = [SystemActor()] * 150
        actions = {
            "Alpha": StandoffAction(posture=StandoffPosture.STAND_FIRM),
            "Omega": StandoffAction(posture=StandoffPosture.BACK_DOWN),
        }

        events = crisis.resolve(basic_game, actions)
        assert len(events) == 1
        event = events[0]
        # Target clock is max_clock_state - 1.
        target_clock = basic_game.rules.max_clock_state - 1
        expected_delta = target_clock - basic_game.doomsday_clock
        assert event.clock_delta == expected_delta

    def test_resolve_winner_p1(
        self,
        crisis: StandoffCrisis,
        basic_game: GameState,
    ) -> None:
        """Test resolution when Player1 stands firm and Player2 backs down."""
        actions = {
            "Alpha": StandoffAction(posture=StandoffPosture.STAND_FIRM),
            "Omega": StandoffAction(posture=StandoffPosture.BACK_DOWN),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event.actor, PlayerActor)
        assert event.actor.name == "Alpha"
        assert event.clock_delta == STANDOFF_WINNER_CLOCK_EFFECT
        assert event.gdp_delta["Omega"] == STANDOFF_LOSER_GDP_EFFECT
        assert event.influence_delta["Omega"] == STANDOFF_LOSER_INF_EFFECT

    def test_resolve_winner_p2(
        self,
        crisis: StandoffCrisis,
        basic_game: GameState,
    ) -> None:
        """Test resolution when Player2 stands firm and Player1 backs down."""
        actions = {
            "Alpha": StandoffAction(posture=StandoffPosture.BACK_DOWN),
            "Omega": StandoffAction(posture=StandoffPosture.STAND_FIRM),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event.actor, PlayerActor)
        assert event.actor.name == "Omega"
        assert event.clock_delta == STANDOFF_WINNER_CLOCK_EFFECT
        assert event.gdp_delta["Alpha"] == STANDOFF_LOSER_GDP_EFFECT
        assert event.influence_delta["Alpha"] == STANDOFF_LOSER_INF_EFFECT


def test_crisis_default_additional_prompt() -> None:
    class DummyCrisis(BaseCrisis):
        card_kind: ClassVar[Literal["dummy"]] = "dummy"

        @override
        async def run(self, *_args: Any, **_kwargs: Any) -> Any: ...

    # Ensure no attribute errors when subclasses do not define additional_prompt
    assert DummyCrisis.additional_prompt is None
    assert DummyCrisis().additional_prompt is None


@pytest.mark.asyncio
async def test_resolve_crisis_consumable(basic_game: GameState) -> None:
    class ConsumableCrisis(BaseCrisis):
        card_kind: ClassVar[Literal["consumable-dummy"]] = "consumable-dummy"
        consumable: ClassVar[bool] = True

        @override
        async def run(self, *_args: Any, **_kwargs: Any) -> list[GameEvent]:
            return [
                SystemEvent(
                    description="Test consumable",
                    clock_delta=0,
                )
            ]

    crisis_instance = ConsumableCrisis()
    basic_game.pending_crisis = crisis_instance
    basic_game.crisis_deck.in_play.append(crisis_instance)

    p1 = MagicMock()
    p1.name = "Alpha"
    p2 = MagicMock()
    p2.name = "Omega"

    new_game = await resolve_crisis(basic_game, [p1, p2])

    assert crisis_instance not in new_game.crisis_deck.discard_pile
    assert crisis_instance not in new_game.crisis_deck.in_play
    assert crisis_instance not in new_game.crisis_deck.draw_pile


class TestProxyWarAction:
    def test_validate_semantics_success(self, basic_game: GameState) -> None:
        action = ProxyWarAction(investment=5)
        # Assuming basic_game starts with enough influence, e.g. 10
        basic_game.players["Alpha"].influence = 10
        action.validate_semantics(basic_game, "Alpha")

    def test_validate_semantics_insufficient_influence(
        self, basic_game: GameState
    ) -> None:
        action = ProxyWarAction(investment=15)
        basic_game.players["Alpha"].influence = 10
        with pytest.raises(InsufficientInfluenceError):
            action.validate_semantics(basic_game, "Alpha")

    def test_validate_semantics_negative_investment(
        self, basic_game: GameState
    ) -> None:
        action = ProxyWarAction(investment=-1)
        basic_game.players["Alpha"].influence = 10
        with pytest.raises(InvalidInfluenceAmountError):
            action.validate_semantics(basic_game, "Alpha")


class TestProxyWarCrisis:
    @pytest.fixture
    def crisis(self) -> ProxyWarCrisis:
        return ProxyWarCrisis()

    def test_get_default_action(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        basic_game.players["Alpha"].influence = 10

        # Aggressive default
        aggressive_action = crisis.get_default_action(
            "Alpha", basic_game, aggressive=True
        )
        assert aggressive_action.investment == 5

        # Cautious default
        cautious_action = crisis.get_default_action(
            "Alpha", basic_game, aggressive=False
        )
        assert cautious_action.investment == 0

        # Aggressive when inf < 5
        basic_game.players["Alpha"].influence = 3
        aggressive_action = crisis.get_default_action(
            "Alpha", basic_game, aggressive=True
        )
        assert aggressive_action.investment == 3

    def test_resolve_tie(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        actions = {
            "Alpha": ProxyWarAction(investment=4),
            "Omega": ProxyWarAction(investment=4),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert "stalemate" in event.description
        assert event.clock_delta == 8
        assert event.influence_delta == {"Alpha": -4, "Omega": -4}
        assert not event.gdp_delta

    def test_resolve_winner_p1(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        actions = {
            "Alpha": ProxyWarAction(investment=6),
            "Omega": ProxyWarAction(investment=3),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert "decisive victory" in event.description
        assert event.clock_delta == 9
        assert event.influence_delta == {"Alpha": -6, "Omega": -3}
        assert event.gdp_delta == {"Alpha": ProxyWarDefs.WINNER_GDP}

    def test_resolve_winner_p2(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        actions = {
            "Alpha": ProxyWarAction(investment=2),
            "Omega": ProxyWarAction(investment=7),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert "decisive victory" in event.description
        assert event.clock_delta == 9
        assert event.influence_delta == {"Alpha": -2, "Omega": -7}
        assert event.gdp_delta == {"Omega": ProxyWarDefs.WINNER_GDP}
