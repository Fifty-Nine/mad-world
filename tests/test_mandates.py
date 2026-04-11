"""Tests for mandate cards and logic."""

from __future__ import annotations

import contextlib
import random
from typing import ClassVar

from mad_world.core import GameState
from mad_world.enums import GamePhase
from mad_world.events import ActionEvent, GameEvent, PlayerActor, SystemEvent
from mad_world.mandates import (
    AccelerationistMandate,
    BaseMandate,
    CoolerHeadsMandate,
    CounterIntelligenceMandate,
    MilitaryIndustrialComplexMandate,
    PacifistUtopiaMandate,
    PopularJingoismMandate,
    SleepingGiantMandate,
    SpaceRaceMandate,
    WarProfiteerMandate,
    create_mandate_deck,
)
from mad_world.rules import GameRules


def test_check_endgame_mandates() -> None:
    game = GameState.new_game(rules=GameRules(), players=["Alpha", "Omega"])

    # Verify check_endgame_mandates works properly with a returned value
    class MockEndgameMandate(BaseMandate):
        card_kind = "mock_end"
        is_instant: ClassVar[bool] = False
        title: ClassVar[str] = "Mock"
        description: ClassVar[str] = "Mock description"

        def is_met(self, game: GameState, player_name: str) -> bool:
            with contextlib.suppress(NotImplementedError):
                BaseMandate.is_met(self, game, player_name)
            return True

        def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
            with contextlib.suppress(NotImplementedError):
                BaseMandate.reward(self, game, player_name)
            return [SystemEvent(description="test")]

    m2 = MockEndgameMandate()
    game.players["Alpha"].mandates.append(m2)
    game.check_endgame_mandates()
    assert m2 in game.players["Alpha"].completed_mandates


def test_create_mandate_deck() -> None:
    rng = random.Random(42)
    deck = create_mandate_deck(rng)
    assert len(deck) == 9


def test_sleeping_giant_mandate() -> None:
    mandate = SleepingGiantMandate()
    game = GameState.new_game(rules=GameRules(), players=["Alpha", "Omega"])

    # Alpha is behind by 15 GDP.
    game.players["Alpha"].gdp = 35
    game.players["Omega"].gdp = 50
    assert mandate.is_met(game, "Alpha") is True
    assert mandate.is_met(game, "Omega") is False

    # Alpha is behind by 14 (not enough)
    game.players["Alpha"].gdp = 36
    assert mandate.is_met(game, "Alpha") is False

    # Alpha is behind by 20 (too much)
    game.players["Alpha"].gdp = 30
    assert mandate.is_met(game, "Alpha") is False

    events = mandate.reward(game, "Alpha")
    assert len(events) == 1
    assert events[0].gdp_delta == {"Alpha": 20}


def test_accelerationist_mandate() -> None:
    mandate = AccelerationistMandate()
    game = GameState.new_game(
        rules=GameRules(max_clock_state=30), players=["Alpha", "Omega"]
    )

    game.escalate(PlayerActor(name="Alpha"), 28)
    assert game.doomsday_clock == 28
    assert mandate.is_met(game, "Alpha") is False

    game.escalate(PlayerActor(name="Alpha"), 1)
    assert game.doomsday_clock == 29
    assert mandate.is_met(game, "Alpha") is True

    events = mandate.reward(game, "Alpha")
    assert len(events) == 1
    assert events[0].gdp_delta == {"Alpha": 15}


def test_pacifist_utopia_mandate() -> None:
    mandate = PacifistUtopiaMandate()
    game = GameState.new_game(
        rules=GameRules(max_clock_state=30), players=["Alpha", "Omega"]
    )

    # Max clock is 30, 20% is 6. Debt must be < 6 (so 5 or less).
    for _ in range(5):
        game.escalate(PlayerActor(name="Alpha"), 1)

    assert mandate.is_met(game, "Alpha") is True

    game.escalate(PlayerActor(name="Alpha"), 1)
    assert mandate.is_met(game, "Alpha") is False

    game.players["Alpha"].gdp = 60
    events = mandate.reward(game, "Alpha")
    assert len(events) == 1
    assert events[0].gdp_delta == {"Alpha": 30}


def test_war_profiteer_mandate() -> None:
    mandate = WarProfiteerMandate()
    game = GameState.new_game(
        rules=GameRules(max_clock_state=30), players=["Alpha", "Omega"]
    )

    # Condition: both > 60 GDP, clock >= 25/30
    game.players["Alpha"].gdp = 61
    game.players["Omega"].gdp = 61
    game.escalate(PlayerActor(name="Alpha"), 25)

    assert mandate.is_met(game, "Alpha") is True

    # One player has <= 60 GDP
    game.players["Omega"].gdp = 60
    assert mandate.is_met(game, "Alpha") is False

    # Restore GDP, but lower clock
    game.players["Omega"].gdp = 61
    game = GameState.new_game(
        rules=GameRules(max_clock_state=30), players=["Alpha", "Omega"]
    )
    game.players["Alpha"].gdp = 61
    game.players["Omega"].gdp = 61
    game.escalate(PlayerActor(name="Alpha"), 24)
    assert mandate.is_met(game, "Alpha") is False

    events = mandate.reward(game, "Alpha")
    assert len(events) == 1
    assert events[0].gdp_delta == {"Alpha": 10, "Omega": -10}


def test_popular_jingoism_mandate() -> None:
    mandate = PopularJingoismMandate()
    game = GameState.new_game(rules=GameRules(), players=["Alpha", "Omega"])

    # Wrong round
    game.current_round = 4
    game.current_phase = GamePhase.OPERATIONS_MESSAGING
    assert mandate.is_met(game, "Alpha") is False

    # Right round, wrong phase
    game.current_round = 5
    game.current_phase = GamePhase.OPERATIONS
    assert mandate.is_met(game, "Alpha") is False

    # Right round and phase, no matching event
    game.current_phase = GamePhase.OPERATIONS_MESSAGING
    assert mandate.is_met(game, "Alpha") is False

    # Add matching event
    game.event_log.append(
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha bid 5 for influence.",
            current_round=5,
            current_phase=GamePhase.BIDDING,
        )
    )
    assert mandate.is_met(game, "Alpha") is True

    # Check logic if opponent bid
    game.event_log.append(
        ActionEvent(
            actor=PlayerActor(name="Omega"),
            description="Omega bid 5 for influence.",
            current_round=5,
            current_phase=GamePhase.BIDDING,
        )
    )
    assert mandate.is_met(game, "Omega") is True

    # Check invalid string gracefully fails
    game.event_log.append(
        ActionEvent(
            actor=PlayerActor(name="Omega"),
            description="Omega bid too_many for influence.",
            current_round=5,
            current_phase=GamePhase.BIDDING,
        )
    )
    assert mandate.is_met(game, "Omega") is True

    # Check loop breaking on older rounds
    game.event_log.insert(
        0,
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha bid 5 for influence.",
            current_round=4,
            current_phase=GamePhase.BIDDING,
        ),
    )
    game.event_log.insert(
        0,
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha bid 5 for influence.",
            current_round=5,
            current_phase=GamePhase.BIDDING,
        ),
    )
    # Still matches the recent round 5 valid one
    assert mandate.is_met(game, "Alpha") is True

    # Now check loop break on empty event log or log with no matches at all
    game.event_log = []
    assert mandate.is_met(game, "Alpha") is False

    # Check early exit when hitting a different round entirely
    game.event_log.append(
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha bid 5 for influence.",
            current_round=4,  # different round
            current_phase=GamePhase.BIDDING,
        )
    )
    game.event_log.append(
        ActionEvent(
            actor=PlayerActor(name="Omega"),
            description="Omega conducted proxy-subversion.",
            current_round=6,  # future round (continue test)
            current_phase=GamePhase.OPERATIONS,
        )
    )
    game.current_round = 5
    assert mandate.is_met(game, "Alpha") is False

    events = mandate.reward(game, "Alpha")
    assert len(events) == 1
    assert events[0].gdp_delta == {"Alpha": 10}


def test_space_race_mandate() -> None:
    mandate = SpaceRaceMandate()
    game = GameState.new_game(rules=GameRules(), players=["Alpha", "Omega"])

    # Test early phase exit
    game.current_phase = GamePhase.OPERATIONS
    assert mandate.is_met(game, "Alpha") is False

    game.current_phase = GamePhase.ROUND_EVENTS

    # Right phase, but no operations
    game.current_phase = GamePhase.ROUND_EVENTS
    game.current_round = 2
    assert mandate.is_met(game, "Alpha") is False

    for _ in range(5):
        game.event_log.append(
            ActionEvent(
                actor=PlayerActor(name="Alpha"),
                description="Alpha conducted domestic-investment.",
                current_round=1,
                current_phase=GamePhase.OPERATIONS,
            )
        )

    assert mandate.is_met(game, "Alpha") is True

    # Test loop break on different round
    game.event_log.insert(
        0,
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha conducted domestic-investment.",
            current_round=0,
            current_phase=GamePhase.OPERATIONS,
        ),
    )
    assert mandate.is_met(game, "Alpha") is True

    # Check loop break on empty event log
    game.event_log = []
    assert mandate.is_met(game, "Alpha") is False

    game.current_round = 1
    game.event_log.append(
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha conducted domestic-investment.",
            current_round=0,  # earlier round triggers early break
            current_phase=GamePhase.OPERATIONS,
        )
    )
    assert mandate.is_met(game, "Alpha") is False

    # Check early exit when hitting a different round entirely
    game.event_log.append(
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="Alpha conducted domestic-investment.",
            current_round=2,  # different round
            current_phase=GamePhase.OPERATIONS,
        )
    )
    game.current_round = 3
    assert mandate.is_met(game, "Alpha") is False

    game.players["Omega"].influence = 10
    events = mandate.reward(game, "Alpha")
    assert len(events) == 1
    assert events[0].gdp_delta == {"Alpha": 5}
    assert events[0].influence_delta == {"Omega": -10}


def test_counter_intelligence_mandate() -> None:
    mandate = CounterIntelligenceMandate()
    game = GameState.new_game(rules=GameRules(), players=["Alpha", "Omega"])

    game.current_phase = GamePhase.ROUND_EVENTS
    game.current_round = 2
    assert mandate.is_met(game, "Alpha") is False

    game.event_log.append(
        ActionEvent(
            actor=PlayerActor(name="Omega"),
            description="Omega conducted proxy-subversion.",
            current_round=1,
            current_phase=GamePhase.OPERATIONS,
        )
    )

    assert mandate.is_met(game, "Alpha") is True

    # Test early exit and loop break coverage
    game.current_phase = GamePhase.BIDDING
    assert mandate.is_met(game, "Alpha") is False
    game.current_phase = GamePhase.OPERATIONS

    # We clear out the log and test an early break on a different round entirely
    game.event_log = []
    game.current_round = 5
    game.event_log.insert(
        0,
        ActionEvent(
            actor=PlayerActor(name="Omega"),
            description="Omega conducted proxy-subversion.",
            current_round=4,  # different round early break
            current_phase=GamePhase.OPERATIONS,
        ),
    )
    assert mandate.is_met(game, "Alpha") is False

    events = mandate.reward(game, "Alpha")
    assert len(events) == 1
    assert events[0].gdp_delta == {"Alpha": 5, "Omega": -5}


def test_cooler_heads_mandate() -> None:
    mandate = CoolerHeadsMandate()
    game = GameState.new_game(
        rules=GameRules(max_clock_state=30), players=["Alpha", "Omega"]
    )

    game.escalate(PlayerActor(name="Alpha"), 29)
    assert mandate.is_met(game, "Alpha") is False

    game.escalate(PlayerActor(name="Alpha"), 1)
    assert mandate.is_met(game, "Alpha") is True

    events = mandate.reward(game, "Alpha")
    assert len(events) == 1
    assert events[0].clock_delta == -4
    assert events[0].influence_delta == {"Alpha": 5}


def test_military_industrial_complex_mandate() -> None:
    game = GameState.new_game(rules=GameRules(), players=["Alpha", "Omega"])
    mandate = MilitaryIndustrialComplexMandate()

    # Not met by default
    assert mandate.is_met(game, "Alpha") is False

    game.current_phase = GamePhase.ROUND_EVENTS
    game.current_round = 2

    # Still not met
    assert mandate.is_met(game, "Alpha") is False

    # Simulate player doing the operation, but not opponent
    game.event_log.extend(
        [
            ActionEvent(
                description="Alpha conducted conventional-offensive",
                current_phase=GamePhase.OPERATIONS,
                current_round=1,
                actor=PlayerActor(name="Alpha"),
            )
        ]
    )
    assert mandate.is_met(game, "Alpha") is False

    # Simulate opponent doing the operation, but not player
    game.event_log.clear()
    game.event_log.extend(
        [
            ActionEvent(
                description="Omega conducted stand-down",
                current_phase=GamePhase.OPERATIONS,
                current_round=1,
                actor=PlayerActor(name="Omega"),
            )
        ]
    )
    assert mandate.is_met(game, "Alpha") is False

    # Both conditions met
    game.event_log.clear()
    game.event_log.extend(
        [
            ActionEvent(
                description="Alpha conducted conventional-offensive",
                current_phase=GamePhase.OPERATIONS,
                current_round=1,
                actor=PlayerActor(name="Alpha"),
            ),
            ActionEvent(
                description="Omega conducted stand-down",
                current_phase=GamePhase.OPERATIONS,
                current_round=1,
                actor=PlayerActor(name="Omega"),
            ),
        ]
    )
    assert mandate.is_met(game, "Alpha") is True

    # Reverse roles, shouldn't work for Alpha
    game.event_log.clear()
    game.event_log.extend(
        [
            ActionEvent(
                description="Omega conducted conventional-offensive",
                current_phase=GamePhase.OPERATIONS,
                current_round=1,
                actor=PlayerActor(name="Omega"),
            ),
            ActionEvent(
                description="Alpha conducted stand-down",
                current_phase=GamePhase.OPERATIONS,
                current_round=1,
                actor=PlayerActor(name="Alpha"),
            ),
        ]
    )
    assert mandate.is_met(game, "Alpha") is False

    # Check rewards
    events = mandate.reward(game, "Alpha")
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SystemEvent)
    assert event.gdp_delta == {"Alpha": 10}
    assert event.influence_delta == {"Alpha": 2}
