"""Unit tests for crisis implementations."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, override
from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_world.actions import (
    BaseAction,
    InsufficientGDPError,
    InsufficientInfluenceError,
    InvalidGDPAmountError,
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
    GlobalFamineAction,
    GlobalFamineCrisis,
    GlobalFamineDefs,
    InternationalSanctionsCrisis,
    ProxyWarAction,
    ProxyWarCrisis,
    ProxyWarDefs,
    RogueProliferationAction,
    RogueProliferationCrisis,
    RogueProliferationDefs,
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

    def test_action_type(self, crisis: InternationalSanctionsCrisis) -> None:
        assert crisis.action_type is BaseAction

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

        @property
        def action_type(self) -> type[BaseAction]:
            return BaseAction

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

        @property
        def action_type(self) -> type[BaseAction]:
            return BaseAction

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

    def test_action_type(self, crisis: ProxyWarCrisis) -> None:
        assert crisis.action_type is ProxyWarAction

    def test_get_default_action(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        basic_game.players["Alpha"].influence = 10

        # Aggressive default
        aggressive_action = crisis.get_default_action(
            "Alpha", basic_game, aggressive=True
        )
        assert aggressive_action.investment == 3

        # Cautious default
        cautious_action = crisis.get_default_action(
            "Alpha", basic_game, aggressive=False
        )
        assert cautious_action.investment == 7

        # Aggressive when inf < 3
        basic_game.players["Alpha"].influence = 2
        aggressive_action = crisis.get_default_action(
            "Alpha", basic_game, aggressive=True
        )
        assert aggressive_action.investment == 2

    def test_resolve_mad_dynamic_threshold(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        basic_game.players["Alpha"].influence = 2
        basic_game.players["Omega"].influence = 3
        # Threshold is 5. Total bid is 4.
        actions = {
            "Alpha": ProxyWarAction(investment=2),
            "Omega": ProxyWarAction(investment=2),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 3
        assert events[0].influence_delta == {"Alpha": -2}
        assert events[1].influence_delta == {"Omega": -2}

        mad_event = events[2]
        assert isinstance(mad_event, SystemEvent)
        assert mad_event.world_ending

    def test_resolve_success_dynamic_threshold(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        basic_game.players["Alpha"].influence = 2
        basic_game.players["Omega"].influence = 3
        # Threshold is 5. Total bid is 5.
        actions = {
            "Alpha": ProxyWarAction(investment=2),
            "Omega": ProxyWarAction(investment=3),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 3
        assert events[0].influence_delta == {"Alpha": -2}
        assert events[1].influence_delta == {"Omega": -3}

        win_event = events[2]
        assert isinstance(win_event, SystemEvent)
        assert "geopolitical victory" in win_event.description
        assert win_event.clock_delta == ProxyWarDefs.CLOCK_IMPACT
        assert win_event.gdp_delta == {"Alpha": ProxyWarDefs.WINNER_GDP}

    def test_resolve_mad(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        actions = {
            "Alpha": ProxyWarAction(investment=4),
            "Omega": ProxyWarAction(investment=3),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 3
        assert events[0].influence_delta == {"Alpha": -4}
        assert events[1].influence_delta == {"Omega": -3}

        mad_event = events[2]
        assert isinstance(mad_event, SystemEvent)
        assert mad_event.world_ending

    def test_resolve_tie(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        actions = {
            "Alpha": ProxyWarAction(investment=5),
            "Omega": ProxyWarAction(investment=5),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 3
        assert events[0].influence_delta == {"Alpha": -5}
        assert events[1].influence_delta == {"Omega": -5}

        tie_event = events[2]
        assert isinstance(tie_event, SystemEvent)
        assert "contributed equally" in tie_event.description
        assert tie_event.clock_delta == ProxyWarDefs.CLOCK_IMPACT
        assert not tie_event.gdp_delta

    def test_resolve_winner_p1(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        actions = {
            "Alpha": ProxyWarAction(investment=3),
            "Omega": ProxyWarAction(investment=8),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 3
        assert events[0].influence_delta == {"Alpha": -3}
        assert events[1].influence_delta == {"Omega": -8}

        win_event = events[2]
        assert isinstance(win_event, SystemEvent)
        assert "geopolitical victory" in win_event.description
        assert win_event.clock_delta == ProxyWarDefs.CLOCK_IMPACT
        assert win_event.gdp_delta == {"Alpha": ProxyWarDefs.WINNER_GDP}

    def test_resolve_winner_p2(
        self, crisis: ProxyWarCrisis, basic_game: GameState
    ) -> None:
        actions = {
            "Alpha": ProxyWarAction(investment=8),
            "Omega": ProxyWarAction(investment=3),
        }

        events = crisis.resolve(basic_game, actions)

        assert len(events) == 3
        assert events[0].influence_delta == {"Alpha": -8}
        assert events[1].influence_delta == {"Omega": -3}

        win_event = events[2]
        assert isinstance(win_event, SystemEvent)
        assert "geopolitical victory" in win_event.description
        assert win_event.clock_delta == ProxyWarDefs.CLOCK_IMPACT
        assert win_event.gdp_delta == {"Omega": ProxyWarDefs.WINNER_GDP}


class TestRogueProliferationAction:
    def test_validate_semantics_success(self, basic_game: GameState) -> None:
        action = RogueProliferationAction(investment=5)
        # Should not raise
        action.validate_semantics(basic_game, "Alpha")

    def test_validate_semantics_insufficient_influence(
        self, basic_game: GameState
    ) -> None:
        action = RogueProliferationAction(investment=20)
        with pytest.raises(InsufficientInfluenceError):
            action.validate_semantics(basic_game, "Alpha")

    def test_validate_semantics_negative_investment(
        self, basic_game: GameState
    ) -> None:
        action = RogueProliferationAction(investment=-5)
        with pytest.raises(InvalidInfluenceAmountError):
            action.validate_semantics(basic_game, "Alpha")


class TestRogueProliferationCrisis(
    CrisisTestBase[RogueProliferationAction, RogueProliferationCrisis]
):
    @pytest.fixture
    @override
    def crisis(self) -> RogueProliferationCrisis:
        return RogueProliferationCrisis()

    @pytest.fixture
    def default_action(self) -> RogueProliferationAction:
        return RogueProliferationAction(investment=5)

    def test_action_type(self, crisis: RogueProliferationCrisis) -> None:
        assert crisis.action_type is RogueProliferationAction

    def test_resolve_success_winner(
        self, basic_game: GameState, crisis: RogueProliferationCrisis
    ) -> None:
        events = crisis.resolve(
            basic_game,
            {
                "Alpha": RogueProliferationAction(investment=6),
                "Omega": RogueProliferationAction(investment=4),
            },
        )

        assert len(events) == 3
        # Cost events
        assert events[0].influence_delta == {"Alpha": -6}
        assert events[1].influence_delta == {"Omega": -4}

        # Resolution event
        assert not events[2].world_ending
        assert events[2].clock_delta == RogueProliferationDefs.CLOCK_IMPACT
        assert events[2].gdp_delta == {
            "Alpha": RogueProliferationDefs.WINNER_GDP,
            "Omega": RogueProliferationDefs.LOSER_GDP,
        }
        assert events[2].influence_delta == {
            "Alpha": RogueProliferationDefs.WINNER_INF
        }

    def test_resolve_success_tie(
        self, basic_game: GameState, crisis: RogueProliferationCrisis
    ) -> None:
        events = crisis.resolve(
            basic_game,
            {
                "Alpha": RogueProliferationAction(investment=5),
                "Omega": RogueProliferationAction(investment=5),
            },
        )

        assert len(events) == 3
        assert events[0].influence_delta == {"Alpha": -5}
        assert events[1].influence_delta == {"Omega": -5}

        assert not events[2].world_ending
        assert events[2].clock_delta == RogueProliferationDefs.CLOCK_IMPACT
        assert events[2].gdp_delta == {}
        assert events[2].influence_delta == {}

    def test_resolve_failure(
        self, basic_game: GameState, crisis: RogueProliferationCrisis
    ) -> None:
        events = crisis.resolve(
            basic_game,
            {
                "Alpha": RogueProliferationAction(investment=2),
                "Omega": RogueProliferationAction(investment=2),
            },
        )

        assert len(events) == 3
        assert events[0].influence_delta == {"Alpha": -2}
        assert events[1].influence_delta == {"Omega": -2}

        assert events[2].world_ending

    def test_resolve_insufficient_total_inf(
        self, basic_game: GameState, crisis: RogueProliferationCrisis
    ) -> None:
        basic_game.players["Alpha"].influence = 2
        basic_game.players["Omega"].influence = 2

        events = crisis.resolve(
            basic_game,
            {
                "Alpha": RogueProliferationAction(investment=2),
                "Omega": RogueProliferationAction(investment=2),
            },
        )

        assert len(events) == 3
        assert not events[2].world_ending
        assert events[2].clock_delta == RogueProliferationDefs.CLOCK_IMPACT
        assert events[2].gdp_delta == {}
        assert events[2].influence_delta == {}

    def test_get_default_action(
        self, basic_game: GameState, crisis: RogueProliferationCrisis
    ) -> None:
        basic_game.players["Alpha"].influence = 10

        aggressive = crisis.get_default_action(
            "Alpha", basic_game, aggressive=True
        )
        assert aggressive.investment == 7

        cautious = crisis.get_default_action(
            "Alpha", basic_game, aggressive=False
        )
        assert cautious.investment == 5

        # Test capping by available influence
        basic_game.players["Alpha"].influence = 3
        capped = crisis.get_default_action("Alpha", basic_game, aggressive=True)
        assert capped.investment == 3


class TestGlobalFamineAction:
    def test_validation_valid(self, basic_game: GameState) -> None:
        action = GlobalFamineAction(investment=20)
        action.validate_semantics(basic_game, "Alpha")

    def test_validation_insufficient_gdp(self, basic_game: GameState) -> None:
        action = GlobalFamineAction(investment=60)
        with pytest.raises(InsufficientGDPError):
            action.validate_semantics(basic_game, "Alpha")

    def test_validation_negative(self, basic_game: GameState) -> None:
        action = GlobalFamineAction(investment=-5)
        with pytest.raises(InvalidGDPAmountError):
            action.validate_semantics(basic_game, "Alpha")


class TestGlobalFamineCrisis(
    CrisisTestBase[GlobalFamineAction, GlobalFamineCrisis]
):
    @pytest.fixture
    @override
    def crisis(self) -> GlobalFamineCrisis:
        return GlobalFamineCrisis()

    def test_properties(self, crisis: GlobalFamineCrisis) -> None:
        assert crisis.card_kind == "global-famine"
        assert crisis.action_type is GlobalFamineAction

    def test_get_default_action(
        self, basic_game: GameState, crisis: GlobalFamineCrisis
    ) -> None:
        # Initial GDP is 50, half_bid for GlobalFamine is 15 // 2 = 7
        aggressive = crisis.get_default_action(
            "Alpha", basic_game, aggressive=True
        )
        assert aggressive.investment == 3  # 7 // 2

        cautious = crisis.get_default_action(
            "Alpha", basic_game, aggressive=False
        )
        assert cautious.investment == 9  # 7 + 2

        # Test capping by available gdp
        basic_game.players["Alpha"].gdp = 2
        capped = crisis.get_default_action("Alpha", basic_game, aggressive=True)
        assert capped.investment == 2

    def test_resolve_success_uneven_bids(
        self, basic_game: GameState, crisis: GlobalFamineCrisis
    ) -> None:
        # Threshold is 15, Alpha bids 10, Omega bids 6 -> total 16 > 15
        actions = {
            "Alpha": GlobalFamineAction(investment=10),
            "Omega": GlobalFamineAction(investment=6),
        }

        events = crisis.resolve(basic_game, actions)
        assert len(events) == 3

        # First two events are CrisisResolutionEvent for deducting investments
        assert events[0].gdp_delta == {"Alpha": -10}
        assert events[1].gdp_delta == {"Omega": -6}

        # Last event is a SystemEvent for resolving the crisis
        assert isinstance(events[2], SystemEvent)
        assert events[2].gdp_delta == {"Alpha": GlobalFamineDefs.WINNER_GDP}
        assert events[2].influence_delta == {
            "Alpha": GlobalFamineDefs.WINNER_INF
        }
        assert events[2].clock_delta == 0
        assert "securing exclusive trade agreements" in events[2].description

    def test_resolve_success_tie(
        self, basic_game: GameState, crisis: GlobalFamineCrisis
    ) -> None:
        # Threshold is 15, Alpha bids 8, Omega bids 8 -> total 16 > 15
        actions = {
            "Alpha": GlobalFamineAction(investment=8),
            "Omega": GlobalFamineAction(investment=8),
        }

        events = crisis.resolve(basic_game, actions)
        assert len(events) == 3

        assert isinstance(events[2], SystemEvent)
        assert not events[2].gdp_delta
        assert not events[2].influence_delta
        assert "contributed equally" in events[2].description

    def test_resolve_failure(
        self, basic_game: GameState, crisis: GlobalFamineCrisis
    ) -> None:
        # Threshold is 15, Alpha bids 5, Omega bids 5 -> total 10 < 15
        actions = {
            "Alpha": GlobalFamineAction(investment=5),
            "Omega": GlobalFamineAction(investment=5),
        }

        events = crisis.resolve(basic_game, actions)
        assert len(events) == 3

        assert isinstance(events[2], SystemEvent)
        assert events[2].clock_delta == GlobalFamineDefs.FAIL_CLOCK_IMPACT
        assert events[2].influence_delta == {
            "Alpha": GlobalFamineDefs.FAIL_INF_PENALTY,
            "Omega": GlobalFamineDefs.FAIL_INF_PENALTY,
        }
        assert "failed to provide adequate relief" in events[2].description
