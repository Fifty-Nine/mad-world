"""Events and event-related functionality."""

from __future__ import annotations

from enum import Enum, StrEnum
from itertools import dropwhile, takewhile
from typing import TYPE_CHECKING, Annotated, Literal, cast, override

from pydantic import BaseModel, Field

from mad_world.enums import GamePhase

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from mad_world.effects import BaseEffect  # noqa: TC004


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


class MessageEvent(BaseGameEvent):
    event_kind: Literal[EventKind.MESSAGE] = Field(default=EventKind.MESSAGE)
    actor: PlayerActor

    @override
    def done_by_player(self, name: str) -> bool:
        return self.actor.name == name


GameEvent = Annotated[
    SystemEvent
    | StateEvent
    | ActionEvent
    | MessageEvent
    | BiddingEvent
    | OperationConductedEvent
    | CrisisResolutionEvent,
    Field(discriminator="event_kind"),
]


class EventLogQuery[T: BaseGameEvent]:
    """A composable query builder for filtering streams of GameEvents."""

    _base_events: Sequence[T]
    _is_reversed: bool
    _predicates: list[Callable[[T], bool]]
    _round_bounds: tuple[int | None, int | None] | None
    _recent_phase_only: bool

    def __init__(
        self,
        events: Sequence[T] | EventLogQuery[T],
        *,
        is_reversed: bool = False,
        predicates: list[Callable[[T], bool]] | None = None,
        round_bounds: tuple[int | None, int | None] | None = None,
        recent_phase_only: bool = False,
    ) -> None:
        if isinstance(events, EventLogQuery):
            self._base_events = events._base_events  # noqa: SLF001
            self._is_reversed = events._is_reversed ^ is_reversed  # noqa: SLF001
            self._predicates = events._predicates.copy() + (predicates or [])  # noqa: SLF001

            curr_start, curr_end = events._round_bounds or (None, None)  # noqa: SLF001
            new_start, new_end = round_bounds or (None, None)

            final_start = curr_start
            if new_start is not None:
                final_start = (
                    max(curr_start, new_start)
                    if curr_start is not None
                    else new_start
                )

            final_end = curr_end
            if new_end is not None:
                final_end = (
                    min(curr_end, new_end) if curr_end is not None else new_end
                )

            self._round_bounds = (final_start, final_end)
            self._recent_phase_only = (
                events._recent_phase_only or recent_phase_only  # noqa: SLF001
            )
        else:
            self._base_events = events
            self._is_reversed = is_reversed
            self._predicates = predicates or []
            self._round_bounds = round_bounds
            self._recent_phase_only = recent_phase_only

    def _apply_recent_phase(self, it: Iterable[T]) -> Iterable[T]:
        if not self._base_events:
            return iter([])

        last_event = self._base_events[-1]
        t_round = last_event.current_round
        t_phase = last_event.current_phase

        def _matches(e: T) -> bool:
            return bool(
                e.current_round == t_round and e.current_phase == t_phase
            )

        if self._is_reversed:
            return takewhile(_matches, it)

        count = 0
        for e in reversed(self._base_events):
            if _matches(e):
                count += 1
            else:
                break
        return self._base_events[-count:] if count > 0 else []

    def _apply_round_bounds(self, it: Iterable[T]) -> Iterable[T]:
        start_round, end_round = self._round_bounds or (None, None)

        def _get_round(e: T) -> int:
            assert e.current_round is not None
            return e.current_round

        if self._is_reversed:
            if end_round is not None:
                it = dropwhile(
                    lambda e: _get_round(e) > end_round,
                    it,
                )
            if start_round is not None:
                it = takewhile(
                    lambda e: _get_round(e) >= start_round,
                    it,
                )
        else:
            if start_round is not None:
                it = dropwhile(
                    lambda e: _get_round(e) < start_round,
                    it,
                )
            if end_round is not None:
                it = takewhile(
                    lambda e: _get_round(e) <= end_round,
                    it,
                )

        if start_round is not None:
            it = (e for e in it if _get_round(e) >= start_round)
        if end_round is not None:
            it = (e for e in it if _get_round(e) <= end_round)

        return it

    def __iter__(self) -> Iterator[T]:

        it: Iterable[T]
        if self._is_reversed:
            it = reversed(self._base_events)
        else:
            it = iter(self._base_events)

        if self._recent_phase_only:
            it = self._apply_recent_phase(it)

        if self._round_bounds:
            it = self._apply_round_bounds(it)
        for pred in self._predicates:
            it = (e for e in it if pred(e))

        return iter(it)

    def to_list(self) -> list[T]:
        return list(self)

    def first(self) -> T | None:
        try:
            return next(iter(self))
        except StopIteration:
            return None

    def reverse(self) -> EventLogQuery[T]:
        return EventLogQuery(self, is_reversed=True)

    def of_type[U: GameEvent](self, event_type: type[U]) -> EventLogQuery[U]:
        def _pred(e: BaseGameEvent) -> bool:
            return isinstance(e, event_type)

        q = EventLogQuery(
            self, predicates=cast("list[Callable[[T], bool]]", [_pred])
        )

        return cast("EventLogQuery[U]", q)

    def filter(self, predicate: Callable[[T], bool]) -> EventLogQuery[T]:
        return EventLogQuery(self, predicates=[predicate])

    def in_phase(self, phase: GamePhase) -> EventLogQuery[T]:
        return self.filter(lambda e: e.current_phase == phase)

    def in_rounds(self, start_round: int, end_round: int) -> EventLogQuery[T]:
        return EventLogQuery(self, round_bounds=(start_round, end_round))

    def in_round(self, round_num: int) -> EventLogQuery[T]:
        return self.in_rounds(round_num, round_num)

    def in_recent_phase(self) -> EventLogQuery[T]:
        return EventLogQuery(self, recent_phase_only=True)
