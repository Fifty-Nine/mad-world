"""Ongoing game effects and mandates."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar, override

from pydantic import Field

from mad_world.cards import BaseCard
from mad_world.events import SystemEvent

if TYPE_CHECKING:
    from mad_world.core import GameState
    from mad_world.events import GameEvent
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
    duration: int | None = Field(description="The duration of the effect.")
    start_round: int | None = Field(
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
        assert bids, "Effect {self.title} removed all possible bids!"
        return bids

    def on_expire(self, game: GameState) -> list[GameEvent]:
        return [
            SystemEvent(
                description=f"Ongoing effect '{self.title}' has expired.",
            )
        ]

    @property
    def end_round(self) -> int | None:
        assert self.start_round is not None
        return (
            (self.start_round + self.duration)
            if self.duration is not None
            else None
        )

    def is_expired(self, game: GameState) -> bool:
        assert self.start_round is not None
        return (
            self.end_round is not None and game.current_round >= self.end_round
        )


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
        return super().modify_bids([b for b in bids if b != 0])


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


class ArmsControlEffect(BaseEffect):
    card_kind: ClassVar[str] = "arms_control"

    title: ClassVar[str] = "Arms Control Treaty"
    description: ClassVar[str] = (
        "A temporary treaty restricts extreme political maneuvers."
    )
    mechanics: ClassVar[str] = (
        "During the bidding phase, bids are capped at a maximum of 3 while "
        "the effect is ongoing."
    )

    @override
    def modify_bids(self, bids: list[int]) -> list[int]:
        return super().modify_bids([b for b in bids if b <= 3])


class UNPeacekeepingEffect(BaseEffect):
    card_kind: ClassVar[str] = "un_peacekeeping"

    title: ClassVar[str] = "UN Peacekeeping Mission"
    description: ClassVar[str] = (
        "UN Peacekeeping forces have been deployed, making armed conflicts "
        "impossible."
    )
    mechanics: ClassVar[str] = (
        "During the operations phase, 'proxy-subversion' and "
        "'conventional-offensive' operations are forbidden while the effect is "
        "ongoing."
    )

    def is_forbidden(self, op_name: str) -> bool:
        return op_name in ("proxy-subversion", "conventional-offensive")

    @override
    def modify_operations(
        self, ops: dict[str, OperationDefinition]
    ) -> dict[str, OperationDefinition]:
        return {op: v for op, v in ops.items() if not self.is_forbidden(op)}
