"""Events and event-related functionality."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mad_world.enums import GamePhase


class ActorKind(Enum):
    SYSTEM = 1
    PLAYER = 2


class SystemActor(BaseModel):
    actor_kind: Literal[ActorKind.SYSTEM] = Field(default=ActorKind.SYSTEM)


class PlayerActor(BaseModel):
    actor_kind: Literal[ActorKind.PLAYER] = Field(default=ActorKind.PLAYER)
    name: str


class GameEvent(BaseModel):
    """Represents a discrete state change in the game."""

    actor: Annotated[
        SystemActor | PlayerActor,
        Field(discriminator="actor_kind"),
    ]
    description: str = Field(description="A brief description of the event.")
    clock_delta: int = Field(
        default=0,
        description="The change in the doomsday clock.",
    )
    gdp_delta: dict[str, int] = Field(
        default_factory=dict,
        description="The change in GDP for each player.",
    )
    influence_delta: dict[str, int] = Field(
        default_factory=dict,
        description="The change in influence for each player.",
    )
    secret: bool = Field(
        default=False,
        description=(
            "True if this event should be hidden from players during gameplay."
        ),
    )
    current_round: int | None = Field(
        default=None,
        description=("The round in which this event occurred."),
    )
    current_phase: GamePhase | None = Field(
        default=None,
        description=("The phase in which this event occurred."),
    )
