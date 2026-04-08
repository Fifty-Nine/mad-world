"""Ongoing effects infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from mad_world.cards import BaseCard

if TYPE_CHECKING:
    from mad_world.rules import OperationDefinition


class BaseOngoingEffect(BaseCard):
    """Base class for all ongoing effects in the game."""

    title: str = Field(description="The title of the ongoing effect.")
    description: str = Field(
        description="The narrative description of the ongoing effect.",
    )
    expiration_round: int | None = Field(
        default=None,
        description=(
            "The round number after which this effect expires. "
            "If None, the effect lasts indefinitely."
        ),
    )

    def is_expired(self, current_round: int) -> bool:
        """Return True if the effect should be removed."""
        if self.expiration_round is None:
            return False
        return current_round > self.expiration_round

    def modify_allowed_bids(self, bids: list[int]) -> list[int]:
        """Hook to modify the allowed bids."""
        return bids

    def modify_allowed_operations(
        self, ops: dict[str, OperationDefinition]
    ) -> dict[str, OperationDefinition]:
        """Hook to modify the allowed operations."""
        return ops


class NoZeroBidsEffect(BaseOngoingEffect):
    """An effect that disallows bidding 0."""

    card_kind: ClassVar[str] = "effect_no_zero_bids"
    title: str = "Tense Negotiations"
    description: str = "Players may not bid 0 (stand down) this round."

    def modify_allowed_bids(self, bids: list[int]) -> list[int]:
        return [b for b in bids if b != 0]


class NoDomesticInvestmentEffect(BaseOngoingEffect):
    """An effect that disallows domestic investment."""

    card_kind: ClassVar[str] = "effect_no_domestic_investment"
    title: str = "Global Distraction"
    description: str = (
        "Domestic investment is disallowed while the world watches abroad."
    )

    def modify_allowed_operations(
        self, ops: dict[str, OperationDefinition]
    ) -> dict[str, OperationDefinition]:
        return {k: v for k, v in ops.items() if k != "domestic-investment"}
