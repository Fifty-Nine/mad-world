"""Round event cards and event deck management."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from mad_world.cards import BaseCard
from mad_world.decks import Deck
from mad_world.effects import NoDomesticInvestmentEffect, NoZeroBidsEffect
from mad_world.events import EffectEvent, GameEvent, SystemEvent

if TYPE_CHECKING:
    import random

    from mad_world.core import GameState


class BaseEventCard(BaseCard):
    """Base class for all event deck cards."""

    title: str = Field(description="The title of the event.")
    description: str = Field(
        description="The narrative description of the event.",
    )

    def run(self, game: GameState) -> list[GameEvent]:
        """Execute the immediate effects of the event card."""
        raise NotImplementedError


class ClockUpEvent(BaseEventCard):
    card_kind: ClassVar[str] = "clock_up"

    title: str = "Tensions Rise"
    description: str = (
        "Global tensions rise inexplicably, moving the world closer to "
        "midnight."
    )

    def run(self, game: GameState) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - The doomsday clock increases by 1."
                ),
                clock_delta=1,
            )
        ]


class ClockDownEvent(BaseEventCard):
    card_kind: ClassVar[str] = "clock_down"

    title: str = "Moment of Peace"
    description: str = (
        "A brief moment of international cooperation lowers global tensions."
    )

    def run(self, game: GameState) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - The doomsday clock decreases by 1."
                ),
                clock_delta=-1,
            )
        ]


class InfluenceP1Event(BaseEventCard):
    card_kind: ClassVar[str] = "influence_p1"

    title: str = "Diplomatic Breakthrough"
    description: str = "A diplomatic breakthrough grants influence to Player 1."

    def run(self, game: GameState) -> list[GameEvent]:
        p1 = game.player_names()[0]
        return [
            SystemEvent(
                description=f"Event: {self.title} - {p1} gains 3 influence.",
                influence_delta={p1: 3},
            )
        ]


class InfluenceP2Event(BaseEventCard):
    card_kind: ClassVar[str] = "influence_p2"

    title: str = "Diplomatic Coup"
    description: str = "A diplomatic coup grants influence to Player 2."

    def run(self, game: GameState) -> list[GameEvent]:
        p2 = game.player_names()[1]
        return [
            SystemEvent(
                description=f"Event: {self.title} - {p2} gains 3 influence.",
                influence_delta={p2: 3},
            )
        ]


class GDPP1Event(BaseEventCard):
    card_kind: ClassVar[str] = "gdp_p1"

    title: str = "Economic Boom"
    description: str = "An economic boom grants GDP to Player 1."

    def run(self, game: GameState) -> list[GameEvent]:
        p1 = game.player_names()[0]
        return [
            SystemEvent(
                description=f"Event: {self.title} - {p1} gains 10 GDP.",
                gdp_delta={p1: 10},
            )
        ]


class GDPP2Event(BaseEventCard):
    card_kind: ClassVar[str] = "gdp_p2"

    title: str = "Industrial Surge"
    description: str = "An industrial surge grants GDP to Player 2."

    def run(self, game: GameState) -> list[GameEvent]:
        p2 = game.player_names()[1]
        return [
            SystemEvent(
                description=f"Event: {self.title} - {p2} gains 10 GDP.",
                gdp_delta={p2: 10},
            )
        ]


class InfluenceBothEvent(BaseEventCard):
    card_kind: ClassVar[str] = "influence_both"

    title: str = "Global Summit"
    description: str = "A global summit grants influence to both players."

    def run(self, game: GameState) -> list[GameEvent]:
        p1, p2 = game.player_names()
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title} - Both players gain 2 influence."
                ),
                influence_delta={p1: 2, p2: 2},
            )
        ]


class BanZeroBidsEvent(BaseEventCard):
    card_kind: ClassVar[str] = "ban_zero_bids"

    title: str = "Breakdown in Communications"
    description: str = "De-escalation becomes impossible as talks break down."

    def run(self, game: GameState) -> list[GameEvent]:
        effect = NoZeroBidsEffect(duration=2)
        return [
            EffectEvent(
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
            EffectEvent(
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
                InfluenceP1Event(),
                InfluenceP2Event(),
                GDPP1Event(),
                GDPP2Event(),
                InfluenceBothEvent(),
                BanZeroBidsEvent(),
                BanDomesticInvestmentEvent(),
            ]
        )

    return Deck[BaseEventCard].create(cards, rng)
