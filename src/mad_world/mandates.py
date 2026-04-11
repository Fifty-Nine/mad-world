"""Mandate cards and mechanic definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from mad_world.cards import BaseCard
from mad_world.decks import Deck
from mad_world.enums import GamePhase
from mad_world.events import PlayerActor, SystemEvent

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


class SleepingGiantMandate(EndgameMandate):
    card_kind: ClassVar[str] = "sleeping_giant"
    title: ClassVar[str] = "Sleeping Giant"
    description: ClassVar[str] = (
        "Endgame - If exactly 15-19 GDP behind opponent, gain 20 GDP."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        player_gdp = game.players[player_name].gdp
        opponent_name = next(p for p in game.players if p != player_name)
        opponent_gdp = game.players[opponent_name].gdp

        diff = opponent_gdp - player_gdp
        return 15 <= diff <= 19

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    "Gaining 20 GDP."
                ),
                gdp_delta={player_name: 20},
            )
        ]


class AccelerationistMandate(EndgameMandate):
    card_kind: ClassVar[str] = "accelerationist"
    title: ClassVar[str] = "Accelerationist"
    description: ClassVar[str] = (
        "Endgame - If at the end of the game, the clock is at (max - 1), "
        "gain 15 GDP."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        return game.doomsday_clock == (game.rules.max_clock_state - 1)

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    "Gaining 15 GDP."
                ),
                gdp_delta={player_name: 15},
            )
        ]


class PacifistUtopiaMandate(EndgameMandate):
    card_kind: ClassVar[str] = "pacifist_utopia"
    title: ClassVar[str] = "Pacifist Utopia"
    description: ClassVar[str] = (
        "Endgame - If you have < (20% * max) escalation debt, "
        "multiply your GDP by 1.5."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        debt = game.escalation_track.count(PlayerActor(name=player_name))
        return debt < (0.2 * game.rules.max_clock_state)

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        player_gdp = game.players[player_name].gdp
        bonus = int(player_gdp * 0.5)
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Gaining {bonus} GDP."
                ),
                gdp_delta={player_name: bonus},
            )
        ]


class WarProfiteerMandate(EndgameMandate):
    card_kind: ClassVar[str] = "war_profiteer"
    title: ClassVar[str] = "War Profiteer"
    description: ClassVar[str] = (
        "Endgame - If both players have >60 GDP and the doomsday clock is "
        "near maximum (>= 25/30 proportion), steal 10 GDP from opponent."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        opponent_name = next(p for p in game.players if p != player_name)
        if (
            game.players[player_name].gdp <= 60
            or game.players[opponent_name].gdp <= 60
        ):
            return False

        ratio = game.doomsday_clock / float(game.rules.max_clock_state)
        # 25/30 = 0.8333...
        return ratio >= (25.0 / 30.0)

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        opponent_name = next(p for p in game.players if p != player_name)
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Stealing 10 GDP from {opponent_name}."
                ),
                gdp_delta={player_name: 10, opponent_name: -10},
            )
        ]


class PopularJingoismMandate(InstantMandate):
    card_kind: ClassVar[str] = "popular_jingoism"
    title: ClassVar[str] = "Popular Jingoism"
    description: ClassVar[str] = (
        "Place a bid of 5 or more during round 5, 6 or 7. Gain 10 GDP."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        if game.current_round not in (5, 6, 7):
            return False

        if game.current_phase != GamePhase.OPERATIONS_MESSAGING:
            return False

        def _check_event(e: GameEvent) -> bool:
            if (
                e.current_phase != GamePhase.BIDDING
                or player_name not in e.description
                or "bid " not in e.description
            ):
                return False
            try:
                words = e.description.split()
                idx = words.index("bid")
                bid_val = int(words[idx + 1])
            except (ValueError, IndexError):
                return False
            else:
                return bid_val >= 5

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
                    "Gaining 10 GDP."
                ),
                gdp_delta={player_name: 10},
            )
        ]


class SpaceRaceMandate(InstantMandate):
    card_kind: ClassVar[str] = "space_race"
    title: ClassVar[str] = "Space Race"
    description: ClassVar[str] = (
        "Conduct 5 domestic-investment operations in a single round. "
        "Gain 5 GDP and set opponent's influence to zero."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        if game.current_phase != GamePhase.ROUND_EVENTS:
            return False

        # When the phase advances from OPERATIONS to ROUND_EVENTS, the
        # round number is incremented.
        target_round = game.current_round - 1

        def _check_event(e: GameEvent) -> bool:
            return (
                e.current_phase == GamePhase.OPERATIONS
                and player_name in e.description
                and "domestic-investment" in e.description
            )

        count = sum(
            1
            for e in reversed(game.event_log)
            if e.current_round == target_round and _check_event(e)
        )

        return count >= 5

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        opponent_name = next(p for p in game.players if p != player_name)
        opponent_inf = game.players[opponent_name].influence
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    f"Gaining 5 GDP and reducing {opponent_name}'s influence "
                    "to 0."
                ),
                gdp_delta={player_name: 5},
                influence_delta={opponent_name: -opponent_inf},
            )
        ]


class CounterIntelligenceMandate(InstantMandate):
    card_kind: ClassVar[str] = "counter_intelligence"
    title: ClassVar[str] = "Counter-Intelligence"
    description: ClassVar[str] = (
        "If opponent conducts proxy-subversion, reverse the GDP effect."
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
                e.current_phase == GamePhase.OPERATIONS
                and opponent_name in e.description
                and "proxy-subversion" in e.description
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
                    "Reversing proxy-subversion."
                ),
                gdp_delta={player_name: 5, opponent_name: -5},
            )
        ]


class CoolerHeadsMandate(InstantMandate):
    card_kind: ClassVar[str] = "cooler_heads"
    title: ClassVar[str] = "Cooler Heads"
    description: ClassVar[str] = (
        "If the doomsday clock is at maximum, reduce clock by 4 and "
        "gain 5 influence. This prevents a crisis."
    )

    def is_met(self, game: GameState, player_name: str) -> bool:
        return game.doomsday_clock >= game.rules.max_clock_state

    def reward(self, game: GameState, player_name: str) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"{player_name} fulfilled '{self.title}' mandate! "
                    "Crisis averted! Clock -4, Influence +5."
                ),
                clock_delta=-4,
                influence_delta={player_name: 5},
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
    ]
    return Deck[BaseMandate].create(cards, rng)
