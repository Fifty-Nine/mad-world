"""Ongoing game effects and mandates."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar, override

from pydantic import Field

from mad_world.cards import BaseCard
from mad_world.events import GameEvent, SystemActor

if TYPE_CHECKING:
    from mad_world.core import GameState
    from mad_world.rules import OperationDefinition


class BaseEffect(BaseCard, ABC):
    """Base class for all ongoing effects in the game.

    Class attributes:
        title (str): The title of the effect.
        description (str): A narrative description of the effect.
        mechanics (str): A description of the effect's impact to the game's
                         mechanics.
    """

    title: ClassVar[str]
    description: ClassVar[str]
    mechanics: ClassVar[str]
    expiration_round: int | None = Field(
        default=None,
        description=(
            "The round in which this effect was triggered. "
            "This is filled in by the game loop."
        ),
    )

    def modify_operations(
        self, ops: dict[str, OperationDefinition]
    ) -> dict[str, OperationDefinition]:
        return ops

    def modify_bids(self, bids: list[int]) -> list[int]:
        return bids

    def is_expired(self, game: GameState) -> bool:
        return (
            self.expiration_round is not None
            and game.current_round > self.expiration_round
        )

    def on_expire(self, game: GameState) -> list[GameEvent]:
        return [
            GameEvent(
                actor=SystemActor(),
                description=f"Ongoing effect '{self.title}' has expired.",
            )
        ]


class NoZeroBidsEffect(BaseEffect):
    card_kind: ClassVar[str] = "no_zero_bids"

    title: ClassVar[str] = "Tense Atmosphere"
    description: ClassVar[str] = (
        "De-escalation is politically impossible right now."
    )
    mechanics: ClassVar[str] = (
        "During the bidding phase, zero bids are forbidden while the "
        "effect is ongoing."
    )

    @override
    def modify_bids(self, bids: list[int]) -> list[int]:
        return [b for b in bids if b != 0]


class NoDomesticInvestmentEffect(BaseEffect):
    card_kind: ClassVar[str] = "no_domestic_investment"

    title: ClassVar[str] = "Supply Chain Collapse"
    description: ClassVar[str] = (
        "Domestic investment operations are temporarily disabled."
    )
    mechanics: ClassVar[str] = (
        "During the operations phase, 'domestic-investment' operations are "
        "forbidden while the effect is ongoing."
    )

    @override
    def modify_operations(
        self, ops: dict[str, OperationDefinition]
    ) -> dict[str, OperationDefinition]:
        # Return a copy without 'domestic-investment'
        return {k: v for k, v in ops.items() if k != "domestic-investment"}
