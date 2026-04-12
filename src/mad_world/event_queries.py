from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, cast

from mad_world.events import GameEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from mad_world.enums import GamePhase


class EventLogQuery[T: GameEvent]:
    """A composable, lazily-evaluated query builder for game events."""

    def __init__(
        self,
        base_events: Sequence[T],
        *,
        is_reversed: bool = False,
        predicates: list[Callable[[T], bool]] | None = None,
        round_bounds: tuple[int | None, int | None] | None = None,
        recent_phase_only: bool = False,
    ) -> None:
        self._base_events = base_events
        self._is_reversed = is_reversed
        self._predicates: list[Callable[[T], bool]] = predicates or []
        self._round_bounds = round_bounds
        self._recent_phase_only = recent_phase_only

    def _copy(
        self,
        *,
        is_reversed: bool | None = None,
        predicates: list[Callable[[T], bool]] | None = None,
        round_bounds: tuple[int | None, int | None] | None = None,
        recent_phase_only: bool | None = None,
    ) -> EventLogQuery[T]:
        return EventLogQuery(
            base_events=self._base_events,
            is_reversed=self._is_reversed
            if is_reversed is None
            else is_reversed,
            predicates=self._predicates + (predicates or []),
            round_bounds=self._round_bounds
            if round_bounds is None
            else round_bounds,
            recent_phase_only=self._recent_phase_only
            if recent_phase_only is None
            else recent_phase_only,
        )

    def reverse(self) -> EventLogQuery[T]:
        """Reverse the direction of iteration."""
        return self._copy(is_reversed=not self._is_reversed)

    def filter(self, predicate: Callable[[T], bool]) -> EventLogQuery[T]:
        """Filter events based on a predicate."""
        return self._copy(predicates=[predicate])

    def of_type[U: GameEvent](self, event_type: type[U]) -> EventLogQuery[U]:
        """Filter events strictly by their subclass type."""
        new_query = self._copy(predicates=[lambda e: isinstance(e, event_type)])
        # Use a cast since we know the new query will only contain U
        return cast("EventLogQuery[U]", new_query)

    def in_round(self, round_num: int) -> EventLogQuery[T]:
        """Limit iteration to exactly this round."""
        return self.in_rounds(round_num, round_num)

    def in_rounds(
        self,
        start_round: int | None,
        end_round: int | None,
    ) -> EventLogQuery[T]:
        """Limit iteration to a range of rounds (inclusive)."""
        new_start, new_end = start_round, end_round

        if self._round_bounds:
            old_start, old_end = self._round_bounds
            if old_start is not None:
                new_start = (
                    max(new_start, old_start)
                    if new_start is not None
                    else old_start
                )
            if old_end is not None:
                new_end = (
                    min(new_end, old_end) if new_end is not None else old_end
                )

        return self._copy(round_bounds=(new_start, new_end))

    def in_phase(self, phase: GamePhase) -> EventLogQuery[T]:
        """Filter events strictly by phase."""
        return self._copy(predicates=[lambda e: e.current_phase == phase])

    def in_recent_phase(self) -> EventLogQuery[T]:
        """Limit iteration to only the most recent continuous block
        of the current round and phase.
        """
        return self._copy(recent_phase_only=True)

    def _create_iterator(self) -> Iterable[T]:
        if self._recent_phase_only:
            return self._iter_recent_phase()

        it: Iterable[T] = (
            reversed(self._base_events)
            if self._is_reversed
            else iter(self._base_events)
        )

        if self._round_bounds:
            it = self._apply_round_bounds(it)

        return it

    def _apply_round_bounds(self, it: Iterable[T]) -> Iterable[T]:
        assert self._round_bounds is not None
        start_r, end_r = self._round_bounds

        if not self._is_reversed:
            if start_r is not None:
                it = itertools.dropwhile(
                    lambda e: (
                        e.current_round is not None
                        and e.current_round < start_r
                    ),
                    it,
                )
            if end_r is not None:
                it = itertools.takewhile(
                    lambda e: (
                        e.current_round is None or e.current_round <= end_r
                    ),
                    it,
                )
        else:
            if end_r is not None:
                it = itertools.dropwhile(
                    lambda e: (
                        e.current_round is not None and e.current_round > end_r
                    ),
                    it,
                )
            if start_r is not None:
                it = itertools.takewhile(
                    lambda e: (
                        e.current_round is None or e.current_round >= start_r
                    ),
                    it,
                )

        if start_r is not None:
            it = (
                e
                for e in it
                if e.current_round is not None and e.current_round >= start_r
            )

        if end_r is not None:
            it = (
                e
                for e in it
                if e.current_round is not None and e.current_round <= end_r
            )

        return it

    def _iter_recent_phase(self) -> Iterable[T]:
        last_event = self._base_events[-1]
        assert last_event.current_round is not None
        assert last_event.current_phase is not None
        target_round = last_event.current_round
        target_phase = last_event.current_phase

        def _is_target(e: T) -> bool:
            return (
                e.current_round == target_round
                and e.current_phase == target_phase
            )

        if self._is_reversed:
            return itertools.takewhile(_is_target, reversed(self._base_events))

        count = 0
        for e in reversed(self._base_events):
            if _is_target(e):
                count += 1
            else:
                break
        return (
            self._base_events[i]
            for i in range(
                len(self._base_events) - count, len(self._base_events)
            )
        )

    def __iter__(self) -> Iterator[T]:
        if not self._base_events:
            return iter([])

        it = self._create_iterator()

        for predicate in self._predicates:
            it = (e for e in it if predicate(e))

        return iter(it)

    def __bool__(self) -> bool:
        return any(True for _ in self)
