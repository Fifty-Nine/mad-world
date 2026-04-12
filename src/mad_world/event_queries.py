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

        def _generator() -> Iterator[T]:
            for e in self._stream:
                if predicate(e):
                    yield e
                else:
                    break

        return EventStream(_generator())

    def drop_while(self, predicate: Callable[[T], bool]) -> EventStream[T]:
        """Drop events as long as the predicate is true, then yield the rest."""

        def _generator() -> Iterator[T]:
            iterator = iter(self._stream)
            for e in iterator:
                if not predicate(e):
                    yield e
                    break
            for e in iterator:
                yield e

        return EventStream(_generator())

    def in_round(self, round_num: int) -> EventStream[T]:
        """Filter events strictly by round."""
        return self.filter(lambda e: e.current_round == round_num)

    def in_phase(self, phase: GamePhase) -> EventStream[T]:
        """Filter events strictly by phase."""
        return self.filter(lambda e: e.current_phase == phase)

    def take_latest_phase_block(self) -> EventStream[T]:
        """Yield events matching the round and phase of the first event,
        stopping when either changes. Assumes the stream is reversed.
        """

        def _generator() -> Iterator[T]:
            iterator = iter(self._stream)
            try:
                first_event = next(iterator)
            except StopIteration:
                return

            yield first_event
            target_round = first_event.current_round
            target_phase = first_event.current_phase

            for e in iterator:
                if (
                    e.current_round == target_round
                    and e.current_phase == target_phase
                ):
                    yield e
                else:
                    break

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
