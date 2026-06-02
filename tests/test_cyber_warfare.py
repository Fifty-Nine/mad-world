from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest

from mad_world.crises import (
    CyberWarfareAction,
    CyberWarfareCrisis,
    CyberWarfareDefs,
)
from mad_world.enums import CyberWarfarePosture
from mad_world.events import SystemEvent

from .test_crises import CrisisTestBase

if TYPE_CHECKING:
    from mad_world.core import GameState


class TestCyberWarfareAction:
    def test_validate_semantics(self, basic_game: GameState) -> None:
        action = CyberWarfareAction(posture=CyberWarfarePosture.ATTACK)
        # Should not raise
        action.validate_semantics(basic_game, "Alpha")


class TestCyberWarfareCrisis(
    CrisisTestBase[CyberWarfareAction, CyberWarfareCrisis]
):
    @pytest.fixture
    @override
    def crisis(self) -> CyberWarfareCrisis:
        return CyberWarfareCrisis()

    @pytest.fixture
    def default_action(self) -> CyberWarfareAction:
        return CyberWarfareAction(posture=CyberWarfarePosture.ATTACK)

    def test_action_type(self, crisis: CyberWarfareCrisis) -> None:
        assert crisis.action_type is CyberWarfareAction

    def test_get_default_action(
        self, basic_game: GameState, crisis: CyberWarfareCrisis
    ) -> None:
        aggressive = crisis.get_default_action(
            "Alpha", basic_game, aggressive=True
        )
        assert aggressive.posture == CyberWarfarePosture.ATTACK

        cautious = crisis.get_default_action(
            "Alpha", basic_game, aggressive=False
        )
        assert cautious.posture == CyberWarfarePosture.DEFEND

    def test_resolve_both_attack(
        self, basic_game: GameState, crisis: CyberWarfareCrisis
    ) -> None:
        actions = {
            "Alpha": CyberWarfareAction(posture=CyberWarfarePosture.ATTACK),
            "Omega": CyberWarfareAction(posture=CyberWarfarePosture.ATTACK),
        }
        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert event.clock_delta == CyberWarfareDefs.DOUBLE_ATTACK_CLOCK_PENALTY
        assert event.gdp_delta == {
            "Alpha": CyberWarfareDefs.DOUBLE_ATTACK_GDP_PENALTY,
            "Omega": CyberWarfareDefs.DOUBLE_ATTACK_GDP_PENALTY,
        }
        assert not event.influence_delta

    def test_resolve_both_defend(
        self, basic_game: GameState, crisis: CyberWarfareCrisis
    ) -> None:
        actions = {
            "Alpha": CyberWarfareAction(posture=CyberWarfarePosture.DEFEND),
            "Omega": CyberWarfareAction(posture=CyberWarfarePosture.DEFEND),
        }
        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert event.clock_delta == CyberWarfareDefs.DOUBLE_DEFEND_CLOCK_REWARD
        assert event.influence_delta == {
            "Alpha": CyberWarfareDefs.DOUBLE_DEFEND_INF_PENALTY,
            "Omega": CyberWarfareDefs.DOUBLE_DEFEND_INF_PENALTY,
        }
        assert not event.gdp_delta

    def test_resolve_attack_defend(
        self, basic_game: GameState, crisis: CyberWarfareCrisis
    ) -> None:
        actions = {
            "Alpha": CyberWarfareAction(posture=CyberWarfarePosture.ATTACK),
            "Omega": CyberWarfareAction(posture=CyberWarfarePosture.DEFEND),
        }
        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert not event.clock_delta
        assert event.gdp_delta == {
            "Alpha": CyberWarfareDefs.ATTACK_DEFEND_GDP_PENALTY
        }
        assert event.influence_delta == {
            "Alpha": CyberWarfareDefs.ATTACK_DEFEND_INF_REWARD
        }

    def test_resolve_defend_attack(
        self, basic_game: GameState, crisis: CyberWarfareCrisis
    ) -> None:
        actions = {
            "Alpha": CyberWarfareAction(posture=CyberWarfarePosture.DEFEND),
            "Omega": CyberWarfareAction(posture=CyberWarfarePosture.ATTACK),
        }
        events = crisis.resolve(basic_game, actions)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SystemEvent)
        assert not event.clock_delta
        assert event.gdp_delta == {
            "Omega": CyberWarfareDefs.ATTACK_DEFEND_GDP_PENALTY
        }
        assert event.influence_delta == {
            "Omega": CyberWarfareDefs.ATTACK_DEFEND_INF_REWARD
        }
