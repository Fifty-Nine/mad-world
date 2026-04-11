"""Round event cards and event deck management."""

from __future__ import annotations

from abc import abstractmethod
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

from pydantic import Field, model_validator

from mad_world.cards import BaseCard
from mad_world.decks import Deck
from mad_world.effects import (
    ArmsControlEffect,
    BaseEffect,
    NoDomesticInvestmentEffect,
    NoZeroBidsEffect,
)
from mad_world.events import GameEvent, SystemEvent
from mad_world.util import gain_or_lose, increase_or_decrease, risen_or_fallen

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
    def mechanics(self, game: GameState) -> str: ...

    @abstractmethod
    def run(self, game: GameState) -> list[GameEvent]:
        """Execute the immediate effects of the event card."""

    def create_event(self, game: GameState, **kwargs: Any) -> list[GameEvent]:
        return [
            SystemEvent(
                description=(
                    f"Event: {self.title}\n"
                    f"  {self.description}\n"
                    f"  {self.mechanics(game)}\n"
                ),
                **kwargs,
            )
        ]


class BasePlayerEffectCard(BaseEventCard):
    """Base class for event cards that apply an effect to one player."""

    player_idx: int
    amount: int

    @abstractmethod
    def effect_key(self) -> str:
        """Get the key used for the effect (e.g.
        influence_delta for influence)"""
        ...

    @abstractmethod
    def effect_units(self) -> str:
        """Get the units suffix of the effect (e.g. ' Inf' for influence)."""
        ...

    def player_name(self, game: GameState) -> str:
        """Get the name of the affected player."""
        return game.player_names()[self.player_idx]

    @override
    def mechanics(self, game: GameState) -> str:
        return (
            f"{self.player_name(game)} {gain_or_lose(self.amount)}s "
            f"{abs(self.amount)}{self.effect_units()}"
        )

    @override
    def run(self, game: GameState) -> list[GameEvent]:
        effects = {self.effect_key(): {self.player_name(game): self.amount}}
        return self.create_event(game, **effects)


class BaseOngoingEffectEvent(BaseEventCard):
    duration: int = Field(description="The duration of the applied effect.")

    @abstractmethod
    def effect_type(self) -> type[BaseEffect]:
        """Get the effect applied by this card."""
        ...

    @override
    def mechanics(self, _game: GameState) -> str:
        return (
            f"A new ongoing effect has been applied: {self.effect_type().title}"
        )

    @override
    def run(self, game: GameState) -> list[GameEvent]:
        return self.create_event(
            game, new_effects=[self.effect_type()(duration=self.duration)]
        )


class ClockChangeEvent(BaseEventCard):
    card_kind: ClassVar[str] = "clock_change"

    title: str = "Tides of Tension"
    description: str = ""
    amount: int = 1

    @model_validator(mode="after")
    def update_description(self) -> Self:
        self.description = (
            f"Global tensions have {risen_or_fallen(self.amount)} inexplicably."
        )
        return self

    @override
    def mechanics(self, _game: GameState) -> str:
        return (
            f"The doomsday clock {increase_or_decrease(self.amount)}s "
            f"by {abs(self.amount)}"
        )

    def run(self, game: GameState) -> list[GameEvent]:
        return self.create_event(game, clock_delta=self.amount)


class InfluenceChangeEvent(BasePlayerEffectCard):
    card_kind: ClassVar[str] = "diplo_event"

    title: str = ""
    description: str = ""
    amount: int = 3

    @model_validator(mode="after")
    def update_descriptions(self) -> Self:
        kind = "breakthrough" if self.amount >= 0 else "blunder"
        self.title = f"Diplomatic {kind.capitalize()}"
        self.description = "A player experiences a diplomatic {kind}"
        return self

    @override
    def effect_units(self) -> str:
        return " Inf"

    @override
    def effect_key(self) -> str:
        return "influence_delta"


class GDPEvent(BasePlayerEffectCard):
    card_kind: ClassVar[str] = "economic_event"
    title: str = ""
    description: str = ""
    amount: int = 4

    @model_validator(mode="after")
    def update_descriptions(self) -> Self:
        kind = "boom" if self.amount >= 0 else "recession"
        self.title = f"Economic {kind.capitalize()}"
        self.description = f"A player experiences an economic {kind}"
        return self

    @override
    def effect_units(self) -> str:
        return " GDP"

    @override
    def effect_key(self) -> str:
        return "gdp_delta"


class InfluenceBothEvent(BaseEventCard):
    card_kind: ClassVar[str] = "influence_both"

    title: str = "Global Summit"
    description: str = "A global summit grants influence to both players."
    amount: int = 2

    @override
    def mechanics(self, _game: GameState) -> str:
        return (
            f"Both players {gain_or_lose(self.amount)} {abs(self.amount)} Inf."
        )

    def run(self, game: GameState) -> list[GameEvent]:
        p1, p2 = game.player_names()
        return self.create_event(
            game, influence_delta={p1: self.amount, p2: self.amount}
        )


class BanZeroBidsEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "ban_zero_bids"

    title: str = "Breakdown in Communications"
    description: str = "De-escalation becomes impossible as talks break down."
    duration: int = 2

    @override
    def effect_type(self) -> type[NoZeroBidsEffect]:
        return NoZeroBidsEffect


class BanDomesticInvestmentEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "ban_domestic_investment"

    title: str = "Global Market Crash"
    description: str = "A sudden market crash halts domestic investments."
    duration: int = 2

    @override
    def effect_type(self) -> type[NoDomesticInvestmentEffect]:
        return NoDomesticInvestmentEffect


class ArmsControlTreatyEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "arms_control_treaty"

    title: str = "Arms Control Treaty"
    description: str = (
        "International pressure forces a temporary limit on escalation."
    )
    duration: int = 2

    @override
    def effect_type(self) -> type[ArmsControlEffect]:
        return ArmsControlEffect


default_frequencies: tuple[tuple[BaseEventCard, int], ...] = (
    (ClockChangeEvent(amount=1), 3),
    (ClockChangeEvent(amount=-1), 3),
    (ClockChangeEvent(amount=2), 1),
    (ClockChangeEvent(amount=-2), 1),
    (InfluenceChangeEvent(player_idx=0), 3),
    (InfluenceChangeEvent(player_idx=1), 3),
    (GDPEvent(player_idx=0, amount=4), 3),
    (GDPEvent(player_idx=1, amount=4), 3),
    (GDPEvent(player_idx=0, amount=-4), 2),
    (GDPEvent(player_idx=1, amount=-4), 2),
    (InfluenceBothEvent(), 3),
    (BanZeroBidsEvent(), 2),
    (BanDomesticInvestmentEvent(), 3),
    (ArmsControlTreatyEvent(), 3),
)


def create_event_deck(rng: random.Random) -> Deck[BaseEventCard]:
    return Deck[BaseEventCard].create(
        (
            chain.from_iterable(
                (card.model_copy() for _ in range(amount))
                for card, amount in default_frequencies
            )
        ),
        rng,
    )
