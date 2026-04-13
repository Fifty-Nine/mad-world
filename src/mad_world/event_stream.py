from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from mad_world.events import GameEvent, LoggedEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from mad_world.enums import GamePhase


class EventStream:
    """A pure functional stream wrapper for game events.
    Note that this interface is assumed to be a single-shot
    iterator, so you need to either save your query result to
    a list (acceptable if only querying a small number of events)
    or ensure you only traverse the items once."""

    def __init__(self, stream: Iterable[LoggedEvent]) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[LoggedEvent]:
        return iter(self._stream)

    def filter(self, predicate: Callable[[LoggedEvent], bool]) -> EventStream:
        """Wraps the current stream in a new generator expression."""
        return EventStream(e for e in self._stream if predicate(e))

    def of_type[U: GameEvent](self, event_type: type[U]) -> Iterable[U]:
        """Filter events by their inner subclass type and unwrap them.
        Returns a raw Iterable (generator) of the unwrapped inner events,
        so it must be the terminal operation in the chain."""
        return (
            e.event for e in self._stream if isinstance(e.event, event_type)
        )

    def take_while(
        self, predicate: Callable[[LoggedEvent], bool]
    ) -> EventStream:
        """Yield events as long as the predicate is true, then stop."""
        return EventStream(itertools.takewhile(predicate, self._stream))

    def drop_while(
        self, predicate: Callable[[LoggedEvent], bool]
    ) -> EventStream:
        """Drop events as long as the predicate is true, then yield the rest."""
        return EventStream(itertools.dropwhile(predicate, self._stream))

    def in_round(self, round_num: int) -> EventStream:
        """Filter events strictly by round."""
        return self.filter(lambda e: e.round == round_num)

    def in_phase(self, phase: GamePhase) -> EventStream:
        """Filter events strictly by phase."""
        return self.filter(lambda e: e.phase == phase)

    def take_latest_phase_block(self) -> EventStream:
        """Yield events matching the round and phase of the first event,
        stopping when either changes. Assumes the stream is reversed.
        """

        def _generator() -> Iterator[LoggedEvent]:
            iterator = iter(self._stream)
            first_event = next(iterator, None)
            if first_event is None:
                return

            yield first_event
            yield from itertools.takewhile(
                lambda e: (
                    e.round == first_event.round
                    and e.phase == first_event.phase
                ),
                iterator,
            )

        return EventStream(_generator())

    def between_rounds(self, start: int, end: int) -> EventStream:
        """Yield events that occurred between 'start' and 'end' rounds,
        inclusive. This works whether the stream is reversed or not."""

        def _generator() -> Iterator[LoggedEvent]:
            iterator = iter(self._stream)
            first_event = next(iterator, None)
            if first_event is None:
                return

            forward = first_event.round <= end
            inner = itertools.chain([first_event], iterator)

            take = (
                (lambda e: e.round <= end)
                if forward
                else (lambda e: e.round >= start)
            )
            drop = (
                (lambda e: e.round < start)
                if forward
                else (lambda e: e.round > end)
            )

            yield from itertools.takewhile(
                take, itertools.dropwhile(drop, inner)
            )

        return EventStream(_generator())

    def __bool__(self) -> bool:
        """Evaluate truthiness by checking if the stream yields any elements.
        Because checking this consumes the first element of any underlying
        generators, we must wrap the stream with itertools.chain to restore it
        so it can be safely iterated again later.
        """
        iterator = iter(self._stream)
        try:
            first = next(iterator)
        except StopIteration:
            return False
        else:
            self._stream = itertools.chain([first], iterator)
            return True
