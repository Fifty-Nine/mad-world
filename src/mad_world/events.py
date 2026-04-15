"""Events and event-related functionality."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, override

from pydantic import BaseModel, ConfigDict, Field

from mad_world.enums import GamePhase

if TYPE_CHECKING:
    from mad_world.effects import BaseEffect  # noqa: TC004


class ActorKind(IntEnum):
    SYSTEM = 1
    PLAYER = 2


class SystemActor(BaseModel):
    model_config = ConfigDict(frozen=True)

    actor_kind: Literal[ActorKind.SYSTEM] = Field(default=ActorKind.SYSTEM)

    def is_system(self) -> bool:
        return True

    def player(self) -> str | None:
        return None


class PlayerActor(BaseModel):
    model_config = ConfigDict(frozen=True)

    actor_kind: Literal[ActorKind.PLAYER] = Field(default=ActorKind.PLAYER)
    name: str

    def is_system(self) -> bool:
        return False

    def player(self) -> str | None:
        return self.name


AnyActor = SystemActor | PlayerActor
OptActor = AnyActor | None


class EventKind(StrEnum):
    SYSTEM = "system"
    STATE = "state"
    ACTION = "action"
    MESSAGE = "message"
    BIDDING = "bidding"
    OPERATION_CONDUCTED = "operation_conducted"
    CRISIS_RESOLUTION = "crisis_resolution"
    MANDATE_FULFILLED = "mandate_fulfilled"
    CHANNEL_OPENED = "channel_opened"
    CHANNEL_REJECTED = "channel_rejected"


class BaseGameEvent(BaseModel):
    """Represents a discrete state change in the game."""

    model_config = ConfigDict(frozen=True)

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
    world_ending: bool = Field(
        default=False, description="True if this event ends the world."
    )
    new_effects: list[BaseEffect] = Field(
        default_factory=list,
        description="Ongoing effects applied by this event.",
    )
    shift_blame: tuple[AnyActor, int] | None = Field(
        default=None,
        description=(
            "Shift the given number of blame cubes from the event "
            "actor to another."
        ),
    )
    channels_opened: dict[str, int] = Field(
        default_factory=dict,
        description="The change to per-player channel quotas.",
    )

    def done_by_player(self, name: str) -> bool:
        return False


class SystemEvent(BaseGameEvent):
    event_kind: Literal[EventKind.SYSTEM] = Field(default=EventKind.SYSTEM)
    actor: SystemActor = Field(default_factory=SystemActor)


class StateEvent(BaseGameEvent):
    event_kind: Literal[EventKind.STATE] = Field(default=EventKind.STATE)
    actor: SystemActor = Field(default_factory=SystemActor)


class BaseActionEvent(BaseGameEvent):
    """Intermediate base class for events initiated by a player actor."""

    actor: PlayerActor

    @override
    def done_by_player(self, name: str) -> bool:
        return self.actor.name == name


class ActionEvent(BaseActionEvent):
    event_kind: Literal[EventKind.ACTION] = Field(default=EventKind.ACTION)


class BiddingEvent(BaseActionEvent):
    event_kind: Literal[EventKind.BIDDING] = Field(default=EventKind.BIDDING)
    bid: int


class OperationConductedEvent(BaseActionEvent):
    event_kind: Literal[EventKind.OPERATION_CONDUCTED] = Field(
        default=EventKind.OPERATION_CONDUCTED
    )
    operation: str


class CrisisResolutionEvent(BaseActionEvent):
    event_kind: Literal[EventKind.CRISIS_RESOLUTION] = Field(
        default=EventKind.CRISIS_RESOLUTION
    )


class MandateFulfilledEvent(BaseActionEvent):
    event_kind: Literal[EventKind.MANDATE_FULFILLED] = Field(
        default=EventKind.MANDATE_FULFILLED
    )
    mandate_title: str


class MessageEvent(BaseGameEvent):
    event_kind: Literal[EventKind.MESSAGE] = Field(default=EventKind.MESSAGE)
    actor: PlayerActor
    message: str | None = Field(description="The body of the message.")
    channel_message: bool = Field(
        description="True if this was sent as part of an open channel."
    )

    @override
    def done_by_player(self, name: str) -> bool:
        return self.actor.name == name


class ChannelOpenedEvent(BaseGameEvent):
    event_kind: Literal[EventKind.CHANNEL_OPENED] = Field(
        default=EventKind.CHANNEL_OPENED
    )
    actor: SystemActor = Field(default_factory=SystemActor)
    initiator: PlayerActor | None = Field(
        description=(
            "The player that requested the channel, "
            "or None if both players requested it."
        )
    )


class ChannelRejectedEvent(BaseActionEvent):
    event_kind: Literal[EventKind.CHANNEL_REJECTED] = Field(
        default=EventKind.CHANNEL_REJECTED
    )
    initiator: PlayerActor = Field(
        description="The player that requested the channel."
    )


GameEvent = Annotated[
    SystemEvent
    | StateEvent
    | ActionEvent
    | MessageEvent
    | BiddingEvent
    | OperationConductedEvent
    | CrisisResolutionEvent
    | MandateFulfilledEvent
    | ChannelOpenedEvent
    | ChannelRejectedEvent,
    Field(discriminator="event_kind"),
]


class LoggedEvent[T: BaseGameEvent](BaseModel):
    """A wrapper for a game event, recording when it occurred."""

    model_config = ConfigDict(frozen=True)

    round: int
    phase: GamePhase
    event: T
