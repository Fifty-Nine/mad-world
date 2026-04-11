"""Round event cards and event deck management."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from mad_world.cards import BaseCard
from mad_world.decks import Deck
from mad_world.effects import (
    ArmsControlEffect,
    NoDomesticInvestmentEffect,
    NoZeroBidsEffect,
    UNPeacekeepingEffect,
)
from mad_world.events import GameEvent, SystemEvent

if TYPE_CHECKING:
    import random

    from mad_world.core import GameState


class BaseEventCard(BaseCard):
    """Base class for all event deck cards."""

    title: str = Field(description="The title of the event.")
    description: str = Field(
        description="The narrative description of the event.",
    )

    @abstractmethod
    def run(self, game: GameState) -> list[GameEvent]:
        """Execute the immediate effects of the event card."""


class ClockUpEvent(BaseEventCard):
    card_kind: ClassVar[str] = "clock_up"

    title: str = "Tensions Rise"
    description: str = (
        "Global tensions rise inexplicably, moving the world closer to "
        "midnight."
    )
    amount: int = 1

    def run(self, game: GameState) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - The doomsday clock "
                    f"increases by {self.amount}."
                ),
                clock_delta=self.amount,
            )
        ]


class ClockDownEvent(BaseEventCard):
    card_kind: ClassVar[str] = "clock_down"

    title: str = "Moment of Peace"
    description: str = (
        "A brief moment of international cooperation lowers global tensions."
    )
    amount: int = 1

    def run(self, game: GameState) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - The doomsday clock "
                    f"decreases by {self.amount}."
                ),
                clock_delta=-self.amount,
            )
        ]


class InfluenceEvent(BaseEventCard):
    card_kind: ClassVar[str] = "diplo_breakthrough"

    title: str = "Diplomatic Breakthrough"
    description: str = "A diplomatic breakthrough grants influence to a player."
    inf_bonus: int = 3
    player_idx: int

    def run(self, game: GameState) -> list[GameEvent]:
        player = game.player_names()[self.player_idx]
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - {player} "
                    f"gains {self.inf_bonus} influence."
                ),
                influence_delta={player: self.inf_bonus},
            )
        ]


class GDPEvent(BaseEventCard):
    card_kind: ClassVar[str] = "economic_boom"
    title: str = "Economic Boom"
    description: str = "A player experiences an economic boom."
    gdp_bonus: int = 4
    player_idx: int

    def run(self, game: GameState) -> list[GameEvent]:
        player = game.player_names()[self.player_idx]
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - {player} gains {self.gdp_bonus} "
                    "GDP."
                ),
                gdp_delta={player: self.gdp_bonus},
            )
        ]


class InfluenceBothEvent(BaseEventCard):
    card_kind: ClassVar[str] = "influence_both"

    title: str = "Global Summit"
    description: str = "A global summit grants influence to both players."
    inf_bonus: int = 2

    def run(self, game: GameState) -> list[GameEvent]:
        p1, p2 = game.player_names()
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - Both players gain "
                    f"{self.inf_bonus} influence."
                ),
                influence_delta={p1: self.inf_bonus, p2: self.inf_bonus},
            )
        ]


class BanZeroBidsEvent(BaseEventCard):
    card_kind: ClassVar[str] = "ban_zero_bids"

    title: str = "Breakdown in Communications"
    description: str = "De-escalation becomes impossible as talks break down."

    def run(self, game: GameState) -> list[GameEvent]:
        effect = NoZeroBidsEffect(duration=2)
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - A new ongoing effect "
                    f"'{effect.title}' has been applied."
                ),
                new_effects=[effect],
            )
        ]


class BanDomesticInvestmentEvent(BaseEventCard):
    card_kind: ClassVar[str] = "ban_domestic_investment"

    title: str = "Global Market Crash"
    description: str = "A sudden market crash halts domestic investments."

    def run(self, game: GameState) -> list[GameEvent]:
        effect = NoDomesticInvestmentEffect(duration=2)
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - A new ongoing effect "
                    f"'{effect.title}' has been applied."
                ),
                new_effects=[effect],
            )
        ]


class ArmsControlTreatyEvent(BaseEventCard):
    card_kind: ClassVar[str] = "arms_control_treaty"

    title: str = "Arms Control Treaty"
    description: str = (
        "International pressure forces a temporary limit on escalation."
    )

    def run(self, game: GameState) -> list[GameEvent]:
        effect = ArmsControlEffect(duration=2)
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - A new ongoing effect "
                    f"'{effect.title}' has been applied."
                ),
                new_effects=[effect],
            )
        ]


class UNPeacekeepingEvent(BaseEventCard):
    card_kind: ClassVar[str] = "un_peacekeeping_event"

    title: str = "UN Peacekeeping Mission"
    description: str = (
        "The UN has successfully deployed peacekeeping forces to global "
        "hotspots, making armed conflicts impossible."
    )

    def run(self, game: GameState) -> list[GameEvent]:
        effect = UNPeacekeepingEffect(duration=2)
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - A new ongoing effect "
                    f"'{effect.title}' has been applied."
                ),
                new_effects=[effect],
            )
        ]


def create_event_deck(rng: random.Random) -> Deck[BaseEventCard]:
    cards: list[BaseEventCard] = []

    for _ in range(3):
        cards.extend(
            [
                ClockUpEvent(),
                ClockDownEvent(),
                InfluenceEvent(player_idx=0),
                InfluenceEvent(player_idx=1),
                GDPEvent(player_idx=0),
                GDPEvent(player_idx=1),
                InfluenceBothEvent(),
                BanZeroBidsEvent(),
                BanDomesticInvestmentEvent(),
                ArmsControlTreatyEvent(),
                UNPeacekeepingEvent(),
            ]
        )

    return Deck[BaseEventCard].create(cards, rng)
