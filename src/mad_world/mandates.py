"""Mandate cards and mechanic definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import dropwhile, takewhile
from typing import TYPE_CHECKING, ClassVar

from mad_world.cards import BaseCard
from mad_world.decks import Deck
from mad_world.enums import GamePhase
from mad_world.events import (
    BiddingEvent,
    OperationConductedEvent,
    PlayerActor,
    SystemEvent,
)

if TYPE_CHECKING:
    import random

    from mad_world.core import GameState
    from mad_world.events import GameEvent


class BaseMandate(BaseCard, ABC):
    """Base class for all mandate cards."""

    title: ClassVar[str]
    description: ClassVar[str]
    is_instant: ClassVar[bool]

    @abstractmethod
    def is_met(self, game: GameState, player_name: str) -> bool:
        """Check if the condition for this mandate has been met."""
        raise NotImplementedError

    @abstractmethod
    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        """Return the rewards (as GameEvents) for completing the mandate."""
        raise NotImplementedError


class InstantMandate(BaseMandate, ABC):
    """Mandates that can be claimed immediately when their condition is met."""

    # Marker to indicate this is an instant mandate
    is_instant: ClassVar[bool] = True


class EndgameMandate(BaseMandate, ABC):
    """Mandates that are only revealed and claimed at the end of the game."""

    # Marker to indicate this is an endgame mandate
    is_instant: ClassVar[bool] = False


class SleepingGiantDefs:
    GDP_BEHIND_MIN: ClassVar[int] = 15
    GDP_BEHIND_MAX: ClassVar[int] = 19
    REWARD_GDP: ClassVar[int] = 20


class SleepingGiantMandate(EndgameMandate):
    card_kind: ClassVar[str] = "sleeping_giant"
    title: ClassVar[str] = "Sleeping Giant"
    description: ClassVar[str] = (
        f"Endgame - If exactly {SleepingGiantDefs.GDP_BEHIND_MIN}-"
        f"{SleepingGiantDefs.GDP_BEHIND_MAX} GDP behind opponent, "
        f"gain {SleepingGiantDefs.REWARD_GDP} GDP."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        player_gdp = game.players[player_name].gdp
        opponent_name = next(p for p in game.players if p != player_name)
        opponent_gdp = game.players[opponent_name].gdp

        diff = opponent_gdp - player_gdp
        return (
            SleepingGiantDefs.GDP_BEHIND_MIN
            <= diff
            <= SleepingGiantDefs.GDP_BEHIND_MAX
        )

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Gaining {SleepingGiantDefs.REWARD_GDP} GDP."
                ),
                gdp_delta={player_name: SleepingGiantDefs.REWARD_GDP},
            )
        ]


class AccelerationistDefs:
    CLOCK_BUFFER: ClassVar[int] = 1
    REWARD_GDP: ClassVar[int] = 15


class AccelerationistMandate(EndgameMandate):
    card_kind: ClassVar[str] = "accelerationist"
    title: ClassVar[str] = "Accelerationist"
    description: ClassVar[str] = (
        f"Endgame - If at the end of the game, the clock is exactly "
        f"(max - {AccelerationistDefs.CLOCK_BUFFER}), gain "
        f"{AccelerationistDefs.REWARD_GDP} GDP."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        return game.doomsday_clock == (
            game.rules.max_clock_state - AccelerationistDefs.CLOCK_BUFFER
        )

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Gaining {AccelerationistDefs.REWARD_GDP} GDP."
                ),
                gdp_delta={player_name: AccelerationistDefs.REWARD_GDP},
            )
        ]


class PacifistUtopiaDefs:
    MAX_CLOCK_PERCENTAGE: ClassVar[float] = 0.2
    GDP_MULTIPLIER: ClassVar[float] = 0.5  # 50% bonus = 1.5x total


class PacifistUtopiaMandate(EndgameMandate):
    card_kind: ClassVar[str] = "pacifist_utopia"
    title: ClassVar[str] = "Pacifist Utopia"
    description: ClassVar[str] = (
        f"Endgame - If you have < "
        f"({int(PacifistUtopiaDefs.MAX_CLOCK_PERCENTAGE * 100)}% * max) "
        f"escalation debt, multiply your GDP by "
        f"{1 + PacifistUtopiaDefs.GDP_MULTIPLIER}."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        debt = game.escalation_track.count(PlayerActor(name=player_name))
        return debt < (
            PacifistUtopiaDefs.MAX_CLOCK_PERCENTAGE * game.rules.max_clock_state
        )

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        player_gdp = game.players[player_name].gdp
        bonus = int(player_gdp * PacifistUtopiaDefs.GDP_MULTIPLIER)
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Gaining {bonus} GDP."
                ),
                gdp_delta={player_name: bonus},
            )
        ]


class WarProfiteerDefs:
    GDP_THRESHOLD: ClassVar[int] = 60
    CLOCK_RATIO: ClassVar[float] = 25.0 / 30.0
    STEAL_AMOUNT: ClassVar[int] = 10


class WarProfiteerMandate(EndgameMandate):
    card_kind: ClassVar[str] = "war_profiteer"
    title: ClassVar[str] = "War Profiteer"
    description: ClassVar[str] = (
        f"Endgame - If both players have >{WarProfiteerDefs.GDP_THRESHOLD} "
        f"GDP and the doomsday clock is near maximum "
        f"(>= {WarProfiteerDefs.CLOCK_RATIO * 100:.1f}% proportion), steal "
        f"{WarProfiteerDefs.STEAL_AMOUNT} GDP from opponent."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        opponent_name = next(p for p in game.players if p != player_name)
        if (
            game.players[player_name].gdp <= WarProfiteerDefs.GDP_THRESHOLD
            or game.players[opponent_name].gdp <= WarProfiteerDefs.GDP_THRESHOLD
        ):
            return False

        ratio = game.doomsday_clock / float(game.rules.max_clock_state)
        return ratio >= WarProfiteerDefs.CLOCK_RATIO

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        opponent_name = next(p for p in game.players if p != player_name)
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Stealing {WarProfiteerDefs.STEAL_AMOUNT} GDP from "
                    f"{opponent_name}."
                ),
                gdp_delta={
                    player_name: WarProfiteerDefs.STEAL_AMOUNT,
                    opponent_name: -WarProfiteerDefs.STEAL_AMOUNT,
                },
            )
        ]


class PopularJingoismDefs:
    ALLOWED_ROUNDS: ClassVar[tuple[int, ...]] = (5, 6, 7)
    MIN_BID: ClassVar[int] = 5
    REWARD_GDP: ClassVar[int] = 10


class PopularJingoismMandate(InstantMandate):
    card_kind: ClassVar[str] = "popular_jingoism"
    title: ClassVar[str] = "Popular Jingoism"
    description: ClassVar[str] = (
        f"Place a bid of {PopularJingoismDefs.MIN_BID} or more during "
        f"round {', '.join(map(str, PopularJingoismDefs.ALLOWED_ROUNDS[:-1]))}"
        f" or {PopularJingoismDefs.ALLOWED_ROUNDS[-1]}. "
        f"Gain {PopularJingoismDefs.REWARD_GDP} GDP."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        if game.current_round not in PopularJingoismDefs.ALLOWED_ROUNDS:
            return False

        if game.current_phase != GamePhase.OPERATIONS_MESSAGING:
            return False

        def _check_event(e: GameEvent) -> bool:
            return (
                isinstance(e, BiddingEvent)
                and e.current_phase == GamePhase.BIDDING
                and e.done_by_player(player_name)
                and e.bid >= PopularJingoismDefs.MIN_BID
            )

        for event in reversed(game.event_log):
            if event.current_round != game.current_round:
                break  # Prune search early
            if _check_event(event):
                return True

        return False

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Gaining {PopularJingoismDefs.REWARD_GDP} GDP."
                ),
                gdp_delta={player_name: PopularJingoismDefs.REWARD_GDP},
            )
        ]


class SpaceRaceDefs:
    REQUIRED_OPS: ClassVar[int] = 5
    TARGET_OP: ClassVar[str] = "domestic-investment"
    REWARD_GDP: ClassVar[int] = 5


class SpaceRaceMandate(InstantMandate):
    card_kind: ClassVar[str] = "space_race"
    title: ClassVar[str] = "Space Race"
    description: ClassVar[str] = (
        f"Conduct {SpaceRaceDefs.REQUIRED_OPS} {SpaceRaceDefs.TARGET_OP} "
        f"operations in a single round. "
        f"Gain {SpaceRaceDefs.REWARD_GDP} GDP and set opponent's influence "
        "to zero."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        if game.current_phase != GamePhase.ROUND_EVENTS:
            return False

        # When the phase advances from OPERATIONS to ROUND_EVENTS, the
        # round number is incremented.
        target_round = game.current_round - 1

        def _check_event(e: GameEvent) -> bool:
            return (
                isinstance(e, OperationConductedEvent)
                and e.current_phase == GamePhase.OPERATIONS
                and e.done_by_player(player_name)
                and e.operation == SpaceRaceDefs.TARGET_OP
            )

        def _correct_phase(e: GameEvent) -> bool:
            return (
                e.current_phase == GamePhase.OPERATIONS
                and e.current_round == target_round
            )

        count = sum(
            1
            for e in takewhile(
                _correct_phase,
                dropwhile(
                    lambda e: not _correct_phase(e), reversed(game.event_log)
                ),
            )
            if _check_event(e)
        )

        return count >= SpaceRaceDefs.REQUIRED_OPS

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        opponent_name = next(p for p in game.players if p != player_name)
        opponent_inf = game.players[opponent_name].influence
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Gaining {SpaceRaceDefs.REWARD_GDP} GDP and reducing "
                    f"{opponent_name}'s influence to 0."
                ),
                gdp_delta={player_name: SpaceRaceDefs.REWARD_GDP},
                influence_delta={opponent_name: -opponent_inf},
            )
        ]


class CounterIntelligenceDefs:
    TARGET_OP: ClassVar[str] = "proxy-subversion"
    REWARD_GDP: ClassVar[int] = 5
    ENEMY_GDP: ClassVar[int] = -5


class CounterIntelligenceMandate(InstantMandate):
    card_kind: ClassVar[str] = "counter_intelligence"
    title: ClassVar[str] = "Counter-Intelligence"
    description: ClassVar[str] = (
        f"If opponent conducts {CounterIntelligenceDefs.TARGET_OP}, "
        "reverse the GDP effect."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        opponent_name = next(p for p in game.players if p != player_name)

        if game.current_phase not in (
            GamePhase.ROUND_EVENTS,
            GamePhase.OPERATIONS,
        ):
            return False

        # If the phase just advanced to ROUND_EVENTS, the round was incremented.
        target_round = game.current_round
        if game.current_phase == GamePhase.ROUND_EVENTS:
            target_round -= 1

        def _check_event(e: GameEvent) -> bool:
            return (
                isinstance(e, OperationConductedEvent)
                and e.current_phase == GamePhase.OPERATIONS
                and e.done_by_player(opponent_name)
                and e.operation == CounterIntelligenceDefs.TARGET_OP
            )

        for event in reversed(game.event_log):
            if event.current_round == target_round and _check_event(event):
                return True

            # We can safely break if we go past the target round, but if
            # event.current_round > target_round we just continue back.
            if event.current_round and event.current_round < target_round:
                break

        return False

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        opponent_name = next(p for p in game.players if p != player_name)
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Reversing {CounterIntelligenceDefs.TARGET_OP}."
                ),
                gdp_delta={
                    player_name: CounterIntelligenceDefs.REWARD_GDP,
                    opponent_name: CounterIntelligenceDefs.ENEMY_GDP,
                },
            )
        ]


class CoolerHeadsDefs:
    CLOCK_EFFECT: ClassVar[int] = -4
    INF_EFFECT: ClassVar[int] = 5


class CoolerHeadsMandate(InstantMandate):
    card_kind: ClassVar[str] = "cooler_heads"
    title: ClassVar[str] = "Cooler Heads"
    description: ClassVar[str] = (
        f"If the doomsday clock is at maximum, reduce clock by "
        f"{abs(CoolerHeadsDefs.CLOCK_EFFECT)} and gain "
        f"{CoolerHeadsDefs.INF_EFFECT} influence. This prevents a crisis."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        return game.doomsday_clock >= game.rules.max_clock_state

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Crisis averted! Clock {CoolerHeadsDefs.CLOCK_EFFECT}, "
                    f"Influence +{CoolerHeadsDefs.INF_EFFECT}."
                ),
                clock_delta=CoolerHeadsDefs.CLOCK_EFFECT,
                influence_delta={player_name: CoolerHeadsDefs.INF_EFFECT},
            )
        ]


def create_mandate_deck(rng: random.Random) -> Deck[BaseMandate]:
    cards: list[BaseMandate] = [
        SleepingGiantMandate(),
        AccelerationistMandate(),
        PacifistUtopiaMandate(),
        WarProfiteerMandate(),
        PopularJingoismMandate(),
        SpaceRaceMandate(),
        CounterIntelligenceMandate(),
        CoolerHeadsMandate(),
        MilitaryIndustrialComplexMandate(),
    ]
    return Deck[BaseMandate].create(cards, rng)


class MilitaryIndustrialComplexDefs:
    PLAYER_OP: ClassVar[str] = "conventional-offensive"
    OPPONENT_OP: ClassVar[str] = "stand-down"
    REWARD_GDP: ClassVar[int] = 10
    REWARD_INF: ClassVar[int] = 2


class MilitaryIndustrialComplexMandate(InstantMandate):
    card_kind: ClassVar[str] = "military_industrial_complex"
    title: ClassVar[str] = "Military-Industrial Complex"
    description: ClassVar[str] = (
        f"If you conduct {MilitaryIndustrialComplexDefs.PLAYER_OP} while your "
        f"opponent conducts {MilitaryIndustrialComplexDefs.OPPONENT_OP} in the "
        f"same round, gain {MilitaryIndustrialComplexDefs.REWARD_GDP} GDP and "
        f"{MilitaryIndustrialComplexDefs.REWARD_INF} influence."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        opponent_name = next(p for p in game.players if p != player_name)

        if game.current_phase != GamePhase.ROUND_EVENTS:
            return False

        # When the phase advances from OPERATIONS to ROUND_EVENTS, the
        # round number is incremented.
        target_round = game.current_round - 1

        def _correct_phase(e: GameEvent) -> bool:
            return (
                e.current_phase == GamePhase.OPERATIONS
                and e.current_round == target_round
            )

        events_in_phase = list(
            takewhile(
                _correct_phase,
                dropwhile(
                    lambda e: not _correct_phase(e), reversed(game.event_log)
                ),
            )
        )

        player_conducted_op = any(
            isinstance(e, OperationConductedEvent)
            and e.done_by_player(player_name)
            and e.operation == MilitaryIndustrialComplexDefs.PLAYER_OP
            for e in events_in_phase
        )

        opponent_conducted_op = any(
            isinstance(e, OperationConductedEvent)
            and e.done_by_player(opponent_name)
            and e.operation == MilitaryIndustrialComplexDefs.OPPONENT_OP
            for e in events_in_phase
        )

        return player_conducted_op and opponent_conducted_op

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Profiting from asymmetric conflict: "
                    f"+{MilitaryIndustrialComplexDefs.REWARD_GDP} GDP, "
                    f"+{MilitaryIndustrialComplexDefs.REWARD_INF} Influence."
                ),
                gdp_delta={
                    player_name: MilitaryIndustrialComplexDefs.REWARD_GDP
                },
                influence_delta={
                    player_name: MilitaryIndustrialComplexDefs.REWARD_INF
                },
            )
        ]
