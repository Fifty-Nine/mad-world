from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from mad_world.events import GameEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from mad_world.enums import GamePhase


class EventStream[T: GameEvent]:
    """A pure functional stream wrapper for game events."""

    def __init__(self, stream: Iterable[T]) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[T]:
        return iter(self._stream)

    def filter(self, predicate: Callable[[T], bool]) -> EventStream[T]:
        """Wraps the current stream in a new generator expression."""
        return EventStream(e for e in self._stream if predicate(e))

    def of_type[U: GameEvent](self, event_type: type[U]) -> EventStream[U]:
        """Filter events strictly by their subclass type."""
        return EventStream(e for e in self._stream if isinstance(e, event_type))

    def take_while(self, predicate: Callable[[T], bool]) -> EventStream[T]:
        """Yield events as long as the predicate is true, then stop."""
        return EventStream(itertools.takewhile(predicate, self._stream))

    def drop_while(self, predicate: Callable[[T], bool]) -> EventStream[T]:
        """Drop events as long as the predicate is true, then yield the rest."""
        return EventStream(itertools.dropwhile(predicate, self._stream))

    def in_round(self, round_num: int) -> EventStream[T]:
        """Filter events strictly by round."""
        return self.filter(lambda e: e.round == round_num)

    def in_phase(self, phase: GamePhase) -> EventStream[T]:
        """Filter events strictly by phase."""
        return self.filter(lambda e: e.phase == phase)

    def take_latest_phase_block(self) -> EventStream[T]:
        """Yield events matching the round and phase of the first event,
        stopping when either changes. Assumes the stream is reversed.
        """

        def _generator() -> Iterator[T]:
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
