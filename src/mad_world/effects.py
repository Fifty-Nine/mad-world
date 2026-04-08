"""Ongoing game effects and mandates."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from mad_world.cards import BaseCard
from mad_world.events import GameEvent, SystemActor

if TYPE_CHECKING:
    from mad_world.core import GameState
    from mad_world.rules import OperationDefinition


class BaseEffect(BaseCard):
    """Base class for all ongoing effects in the game."""

    title: str = Field(description="The title of the effect.")
    description: str = Field(
        description="The narrative description of the effect."
    )
    expiration_round: int | None = Field(
        default=None,
        description="The round at which this effect expires, if any.",
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

    title: str = "Tense Atmosphere"
    description: str = "De-escalation is politically impossible right now."

    def modify_bids(self, bids: list[int]) -> list[int]:
        return [b for b in bids if b != 0]


class NoDomesticInvestmentEffect(BaseEffect):
    card_kind: ClassVar[str] = "no_domestic_investment"

    title: str = "Supply Chain Collapse"
    description: str = (
        "Domestic investment operations are temporarily disabled."
    )

    def modify_operations(
        self, ops: dict[str, OperationDefinition]
    ) -> dict[str, OperationDefinition]:
        # Return a copy without 'domestic-investment'
        return {k: v for k, v in ops.items() if k != "domestic-investment"}
