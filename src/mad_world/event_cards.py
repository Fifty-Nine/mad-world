"""Round event cards and event deck management."""

from __future__ import annotations

from abc import abstractmethod
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, override

from pydantic import ConfigDict, Field

from mad_world.cards import BaseCard
from mad_world.decks import Deck
from mad_world.effects import (
    ArmsControlEffect,
    BaseEffect,
    GlobalSanctionsEffect,
    HawkishResurgenceEffect,
    NoDomesticInvestmentEffect,
    NoZeroBidsEffect,
    ProxyWarEscalationEffect,
    SupplyChainShockEffect,
    TechnologicalBreakthroughEffect,
    UNPeacekeepingEffect,
)
from mad_world.events import GameEvent, SystemEvent
from mad_world.util import gain_or_lose, increase_or_decrease, risen_or_fallen

if TYPE_CHECKING:
    import random

    from mad_world.core import GameState


class BaseEventCard(BaseCard):
    """Base class for all event deck cards."""

    model_config = ConfigDict(frozen=True)

    @property
    @abstractmethod
    def title(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

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
        return game.player_names[self.player_idx]

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
    def mechanics(self, game: GameState) -> str:
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

    amount: int = 1

    @property
    @override
    def title(self) -> str:
        return "Tides of Tension"

    @property
    @override
    def description(self) -> str:
        return (
            f"Global tensions have {risen_or_fallen(self.amount)} inexplicably."
        )

    @override
    def mechanics(self, game: GameState) -> str:
        return (
            f"The doomsday clock {increase_or_decrease(self.amount)}s "
            f"by {abs(self.amount)}"
        )

    def run(self, game: GameState) -> list[GameEvent]:
        return self.create_event(game, clock_delta=self.amount)


class InfluenceChangeEvent(BasePlayerEffectCard):
    card_kind: ClassVar[str] = "diplo_event"

    amount: int = 3

    @property
    @override
    def title(self) -> str:
        kind = "breakthrough" if self.amount >= 0 else "blunder"
        return f"Diplomatic {kind.capitalize()}"

    @property
    @override
    def description(self) -> str:
        kind = "breakthrough" if self.amount >= 0 else "blunder"
        return f"A player experiences a diplomatic {kind}"

    @override
    def effect_units(self) -> str:
        return " Inf"

    @override
    def effect_key(self) -> str:
        return "influence_delta"


class GDPEvent(BasePlayerEffectCard):
    card_kind: ClassVar[str] = "economic_event"
    amount: int = 4

    @property
    @override
    def title(self) -> str:
        kind = "boom" if self.amount >= 0 else "recession"
        return f"Economic {kind.capitalize()}"

    @property
    @override
    def description(self) -> str:
        kind = "boom" if self.amount >= 0 else "recession"
        return f"A player experiences an economic {kind}"

    @override
    def effect_units(self) -> str:
        return " GDP"

    @override
    def effect_key(self) -> str:
        return "gdp_delta"


class InfluenceBothEvent(BaseEventCard):
    card_kind: ClassVar[str] = "influence_both"

    amount: int = 2

    @property
    @override
    def title(self) -> str:
        return "Global Summit"

    @property
    @override
    def description(self) -> str:
        return "A global summit grants influence to both players."

    @override
    def mechanics(self, game: GameState) -> str:
        return (
            f"Both players {gain_or_lose(self.amount)} {abs(self.amount)} Inf."
        )

    def run(self, game: GameState) -> list[GameEvent]:
        p1, p2 = game.player_names
        return self.create_event(
            game, influence_delta={p1: self.amount, p2: self.amount}
        )


class BanZeroBidsEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "ban_zero_bids"

    duration: int = 2

    @property
    @override
    def title(self) -> str:
        return "Breakdown in Communications"

    @property
    @override
    def description(self) -> str:
        return "De-escalation becomes impossible as talks break down."

    @override
    def effect_type(self) -> type[NoZeroBidsEffect]:
        return NoZeroBidsEffect


class HawkishResurgenceEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "hawkish_resurgence_event"

    duration: int = 2

    @property
    @override
    def title(self) -> str:
        return "Hawkish Resurgence"

    @property
    @override
    def description(self) -> str:
        return "Hardliners take control, making de-escalation impossible."

    @override
    def effect_type(self) -> type[HawkishResurgenceEffect]:
        return HawkishResurgenceEffect


class BanDomesticInvestmentEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "ban_domestic_investment"

    duration: int = 2

    @property
    @override
    def title(self) -> str:
        return "Global Market Crash"

    @property
    @override
    def description(self) -> str:
        return "A sudden market crash halts domestic investments."

    @override
    def effect_type(self) -> type[NoDomesticInvestmentEffect]:
        return NoDomesticInvestmentEffect


class GlobalSanctionsEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "global_sanctions_event"

    duration: int = 2

    @property
    @override
    def title(self) -> str:
        return "Global Sanctions"

    @property
    @override
    def description(self) -> str:
        return (
            "International condemnation temporarily halts aggressive "
            "resource grabs."
        )

    @override
    def effect_type(self) -> type[GlobalSanctionsEffect]:
        return GlobalSanctionsEffect


class ArmsControlTreatyEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "arms_control_treaty"

    duration: int = 2

    @property
    @override
    def title(self) -> str:
        return "Arms Control Treaty"

    @property
    @override
    def description(self) -> str:
        return "International pressure forces a temporary limit on escalation."

    @override
    def effect_type(self) -> type[ArmsControlEffect]:
        return ArmsControlEffect


class SupplyChainShockEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "supply_chain_shock_event"

    @property
    @override
    def title(self) -> str:
        return "Supply Chain Shock"

    @property
    @override
    def description(self) -> str:
        return "Global logistics networks are severely disrupted."

    duration: int = 2

    @override
    def effect_type(self) -> type[SupplyChainShockEffect]:
        return SupplyChainShockEffect


class ProxyWarEscalationEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "proxy_war_escalation_event"

    duration: int = 2

    @property
    @override
    def title(self) -> str:
        return "Proxy War Escalation"

    @property
    @override
    def description(self) -> str:
        return (
            "Foreign powers drastically increase their involvement in proxy "
            "conflicts."
        )

    @override
    def effect_type(self) -> type[ProxyWarEscalationEffect]:
        return ProxyWarEscalationEffect


class UNPeacekeepingEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "un_peacekeeping_event"

    duration: int = 2

    @property
    @override
    def title(self) -> str:
        return "UN Peacekeeping Mission"

    @property
    @override
    def description(self) -> str:
        return (
            "The UN has successfully deployed peacekeeping forces to global "
            "hotspots, making armed conflicts impossible."
        )

    @override
    def effect_type(self) -> type[UNPeacekeepingEffect]:
        return UNPeacekeepingEffect


class TechnologicalBreakthroughEvent(BaseOngoingEffectEvent):
    card_kind: ClassVar[str] = "technological_breakthrough_event"

    duration: int = 2

    @property
    @override
    def title(self) -> str:
        return "Technological Breakthrough"

    @property
    @override
    def description(self) -> str:
        return (
            "A sudden surge in technological innovation spurs economic growth "
            "and national pride."
        )

    @override
    def effect_type(self) -> type[TechnologicalBreakthroughEffect]:
        return TechnologicalBreakthroughEffect


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
    (HawkishResurgenceEvent(), 3),
    (BanDomesticInvestmentEvent(), 3),
    (GlobalSanctionsEvent(), 3),
    (ArmsControlTreatyEvent(), 3),
    (SupplyChainShockEvent(), 3),
    (ProxyWarEscalationEvent(), 3),
    (UNPeacekeepingEvent(), 3),
    (TechnologicalBreakthroughEvent(), 3),
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
