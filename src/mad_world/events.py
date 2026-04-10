"""Events and event-related functionality."""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mad_world.effects import BaseEffect  # noqa: TC004
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


class EventKind(StrEnum):
    SYSTEM = "system"
    STATE = "state"
    ACTION = "action"
    MESSAGE = "message"
    EFFECT = "effect"


class BaseGameEvent(BaseModel):
    """Represents a discrete state change in the game."""

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


class EffectEvent(BaseGameEvent):
    event_kind: Literal[EventKind.EFFECT] = Field(default=EventKind.EFFECT)
    actor: SystemActor = Field(default_factory=SystemActor)
    new_effects: list[BaseEffect] = Field(
        default_factory=list,
        description="Ongoing effects applied by this event.",
    )


class SystemEvent(BaseGameEvent):
    event_kind: Literal[EventKind.SYSTEM] = Field(default=EventKind.SYSTEM)
    actor: SystemActor = Field(default_factory=SystemActor)


class StateEvent(BaseGameEvent):
    event_kind: Literal[EventKind.STATE] = Field(default=EventKind.STATE)
    actor: SystemActor = Field(default_factory=SystemActor)


class ActionEvent(BaseGameEvent):
    event_kind: Literal[EventKind.ACTION] = Field(default=EventKind.ACTION)
    actor: PlayerActor


class MessageEvent(BaseGameEvent):
    event_kind: Literal[EventKind.MESSAGE] = Field(default=EventKind.MESSAGE)
    actor: PlayerActor


GameEvent = Annotated[
    SystemEvent | StateEvent | ActionEvent | MessageEvent | EffectEvent,
    Field(discriminator="event_kind"),
]
