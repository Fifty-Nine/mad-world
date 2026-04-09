"""Events and event-related functionality."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mad_world.cards import BaseCard
from mad_world.enums import GamePhase


class ActorKind(Enum):
    SYSTEM = 1
    PLAYER = 2


class SystemActor(BaseModel):
    actor_kind: Literal[ActorKind.SYSTEM] = Field(default=ActorKind.SYSTEM)

    def is_system(self) -> bool:
        return True

    def player(self) -> str | None:
        return None


class PlayerActor(BaseModel):
    actor_kind: Literal[ActorKind.PLAYER] = Field(default=ActorKind.PLAYER)
    name: str

    def is_system(self) -> bool:
        return False

    def player(self) -> str | None:
        return self.name


AnyActor = SystemActor | PlayerActor | None


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
    world_ending: bool = Field(
        default=False, description="True if this event ends the world."
    )
    # TODO This should go in an EffectEvent or similar subclass so that
    # we can correctly type this field as BaseEffect. This will be possible
    # after #51 lands.
    new_effects: list[BaseCard] = Field(
        default_factory=list,
        description="Ongoing effects applied by this event.",
    )
