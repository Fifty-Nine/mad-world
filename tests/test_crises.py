"""Unit tests for crisis implementations."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, override
from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_world.actions import BaseAction
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
    StandoffAction,
    StandoffCrisis,
)
from mad_world.enums import StandoffPosture
from mad_world.events import GameEvent, PlayerActor, SystemActor


class CrisisTestBase[T: BaseAction, C: GenericCrisis[Any]]:
    """Base class for testing crises."""

    @pytest.fixture
    def mock_game_state(self) -> MagicMock:
        mock = MagicMock()
        mock.doomsday_clock = 50
        mock.rules.max_clock_state = 100
        mock.player_names = MagicMock(return_value=["Player1", "Player2"])
        return mock

    @pytest.fixture
    def crisis(self) -> C:
        raise NotImplementedError("Subclasses must provide a crisis fixture.")

    @pytest.mark.asyncio
    async def test_run_generic(
        self,
        crisis: C,
        mock_game_state: MagicMock,
    ) -> None:
        """Test the run method which gathers player actions."""
        p1 = MagicMock()
        p1.name = "Player1"
        p1_action = crisis.get_default_action(aggressive=True)
        p1.crisis = AsyncMock(return_value=p1_action)

        p2 = MagicMock()
        p2.name = "Player2"
        p2_action = crisis.get_default_action(aggressive=False)
        p2.crisis = AsyncMock(return_value=p2_action)

        events = await crisis.run(mock_game_state, [p1, p2])

        assert isinstance(events, list)
        p1.crisis.assert_called_once_with(mock_game_state, crisis)
        p2.crisis.assert_called_once_with(mock_game_state, crisis)


class TestInternationalSanctionsCrisis:
    """Tests for the InternationalSanctionsCrisis implementation."""

    @pytest.fixture
    def crisis(self) -> InternationalSanctionsCrisis:
        return InternationalSanctionsCrisis()

    @pytest.fixture
    def mock_game_state(self) -> MagicMock:
        mock = MagicMock()
        mock.doomsday_clock = 50
        mock.rules.max_clock_state = 100
        mock.player_names = MagicMock(return_value=["Player1", "Player2"])
        return mock

    @pytest.mark.asyncio
    async def test_run_tie(
        self,
        crisis: InternationalSanctionsCrisis,
        mock_game_state: MagicMock,
    ) -> None:
        """Test resolution when both players have equal debt."""
        mock_game_state.escalation_track = [
            PlayerActor(name="Player1"),
            PlayerActor(name="Player2"),
        ]

        events = await crisis.run(mock_game_state, [])

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, GameEvent)
        assert isinstance(event.actor, SystemActor)
        assert event.clock_delta == SANCTIONS_TIE_CLOCK_EFFECT
        assert event.gdp_delta["Player1"] == SANCTIONS_TIE_GDP_EFFECT
        assert event.gdp_delta["Player2"] == SANCTIONS_TIE_GDP_EFFECT

    @pytest.mark.asyncio
    async def test_run_p1_sanctioned(
        self,
        crisis: InternationalSanctionsCrisis,
        mock_game_state: MagicMock,
    ) -> None:
        """Test resolution when Player1 has more debt."""
        # Player1 debt: 3, Player2 debt: 1. Difference: 2.
        # min effect is 5, so Player1 should lose 5 GDP.
        mock_game_state.escalation_track = [
            PlayerActor(name="Player1"),
            PlayerActor(name="Player1"),
            PlayerActor(name="Player1"),
            PlayerActor(name="Player2"),
        ]

        events = await crisis.run(mock_game_state, [])

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, GameEvent)
        assert isinstance(event.actor, SystemActor)
        assert event.clock_delta == SANCTIONS_MIN_EFFECT
        assert event.gdp_delta["Player1"] == SANCTIONS_MIN_EFFECT
        assert "Player2" not in event.gdp_delta

    @pytest.mark.asyncio
    async def test_run_p2_sanctioned_large_debt(
        self,
        crisis: InternationalSanctionsCrisis,
        mock_game_state: MagicMock,
    ) -> None:
        """Test resolution when Player2 has much more debt."""
        # Player2 debt: 15, Player1 debt: 1. Difference: 14.
        # Should lose 14 GDP.
        mock_game_state.escalation_track = [
            PlayerActor(name="Player2")
        ] * 15 + [PlayerActor(name="Player1")]

        events = await crisis.run(mock_game_state, [])

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, GameEvent)
        assert isinstance(event.actor, SystemActor)
        assert event.clock_delta == -14
        assert event.gdp_delta["Player2"] == -14
        assert "Player1" not in event.gdp_delta


class TestStandoffCrisis(CrisisTestBase[StandoffAction, StandoffCrisis]):
    """Tests for the StandoffCrisis implementation."""

    @pytest.fixture
    def crisis(self) -> StandoffCrisis:
        return StandoffCrisis()

    def test_get_default_action(self, crisis: StandoffCrisis) -> None:
        """Test that default actions are correctly returned."""
        aggressive = crisis.get_default_action(aggressive=True)
        assert aggressive.posture == StandoffPosture.STAND_FIRM

        cautious = crisis.get_default_action(aggressive=False)
        assert cautious.posture == StandoffPosture.BACK_DOWN

    def test_resolve_doomsday(
        self,
        crisis: StandoffCrisis,
        mock_game_state: MagicMock,
    ) -> None:
        """Test resolution when both players stand firm."""
        actions = {
            "Player1": StandoffAction(posture=StandoffPosture.STAND_FIRM),
            "Player2": StandoffAction(posture=StandoffPosture.STAND_FIRM),
        }

        events = crisis.resolve(mock_game_state, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event.actor, SystemActor)
        assert "nuclear exchange" in event.description
        assert event.world_ending

    def test_resolve_tie(
        self,
        crisis: StandoffCrisis,
        mock_game_state: MagicMock,
    ) -> None:
        """Test resolution when both players back down."""
        actions = {
            "Player1": StandoffAction(posture=StandoffPosture.BACK_DOWN),
            "Player2": StandoffAction(posture=StandoffPosture.BACK_DOWN),
        }

        events = crisis.resolve(mock_game_state, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event.actor, SystemActor)
        assert "cooler heads" in event.description
        assert event.clock_delta == STANDOFF_TIE_CLOCK_EFFECT
        assert event.gdp_delta["Player1"] == STANDOFF_TIE_GDP_EFFECT
        assert event.gdp_delta["Player2"] == STANDOFF_TIE_GDP_EFFECT
        assert event.influence_delta["Player1"] == STANDOFF_TIE_INF_EFFECT
        assert event.influence_delta["Player2"] == STANDOFF_TIE_INF_EFFECT

    def test_resolve_tie_clamping(
        self,
        crisis: StandoffCrisis,
        mock_game_state: MagicMock,
    ) -> None:
        """Test resolution when both players back down, but clock is high."""
        mock_game_state.escalation_track = ["System"] * 150
        mock_game_state.doomsday_clock = 150
        mock_game_state.rules.max_clock_state = 100
        actions = {
            "Player1": StandoffAction(posture=StandoffPosture.BACK_DOWN),
            "Player2": StandoffAction(posture=StandoffPosture.BACK_DOWN),
        }

        events = crisis.resolve(mock_game_state, actions)
        assert len(events) == 1
        event = events[0]
        assert event.clock_delta == -51

    def test_resolve_winner_clamping(
        self,
        crisis: StandoffCrisis,
        mock_game_state: MagicMock,
    ) -> None:
        """Test resolution when one player stands firm, and clock is high."""
        mock_game_state.escalation_track = ["System"] * 150
        mock_game_state.doomsday_clock = 150
        mock_game_state.rules.max_clock_state = 100
        actions = {
            "Player1": StandoffAction(posture=StandoffPosture.STAND_FIRM),
            "Player2": StandoffAction(posture=StandoffPosture.BACK_DOWN),
        }

        events = crisis.resolve(mock_game_state, actions)
        assert len(events) == 1
        event = events[0]
        assert event.clock_delta == -51

    def test_resolve_winner_p1(
        self,
        crisis: StandoffCrisis,
        mock_game_state: MagicMock,
    ) -> None:
        """Test resolution when Player1 stands firm and Player2 backs down."""
        actions = {
            "Player1": StandoffAction(posture=StandoffPosture.STAND_FIRM),
            "Player2": StandoffAction(posture=StandoffPosture.BACK_DOWN),
        }

        events = crisis.resolve(mock_game_state, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event.actor, PlayerActor)
        assert event.actor.name == "Player1"
        assert event.clock_delta == STANDOFF_WINNER_CLOCK_EFFECT
        assert event.gdp_delta["Player2"] == STANDOFF_LOSER_GDP_EFFECT
        assert event.influence_delta["Player2"] == STANDOFF_LOSER_INF_EFFECT

    def test_resolve_winner_p2(
        self,
        crisis: StandoffCrisis,
        mock_game_state: MagicMock,
    ) -> None:
        """Test resolution when Player2 stands firm and Player1 backs down."""
        actions = {
            "Player1": StandoffAction(posture=StandoffPosture.BACK_DOWN),
            "Player2": StandoffAction(posture=StandoffPosture.STAND_FIRM),
        }

        events = crisis.resolve(mock_game_state, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event.actor, PlayerActor)
        assert event.actor.name == "Player2"
        assert event.clock_delta == STANDOFF_WINNER_CLOCK_EFFECT
        assert event.gdp_delta["Player1"] == STANDOFF_LOSER_GDP_EFFECT
        assert event.influence_delta["Player1"] == STANDOFF_LOSER_INF_EFFECT


def test_crisis_default_additional_prompt() -> None:
    class DummyCrisis(BaseCrisis):
        card_kind: ClassVar[Literal["dummy"]] = "dummy"

        @override
        async def run(self, *_args: Any, **_kwargs: Any) -> Any: ...

    # Ensure no attribute errors when subclasses do not define additional_prompt
    assert DummyCrisis.additional_prompt is None
    assert DummyCrisis().additional_prompt is None
