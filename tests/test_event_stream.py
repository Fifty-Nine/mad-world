from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import overload

import pytest

from mad_world.enums import GamePhase
from mad_world.event_stream import EmptyStreamError, EventStream
from mad_world.events import (
    BiddingEvent,
    GameEvent,
    LoggedEvent,
    OperationConductedEvent,
    PlayerActor,
    SystemActor,
    SystemEvent,
)


def make_system_event(
    round_num: int, phase: GamePhase, desc: str = "sys"
) -> LoggedEvent[GameEvent]:
    return LoggedEvent(
        round=round_num,
        phase=phase,
        event=SystemEvent(
            description=desc,
            actor=SystemActor(),
        ),
    )


def make_bidding_event(
    round_num: int, phase: GamePhase, bid: int
) -> LoggedEvent[GameEvent]:
    return LoggedEvent(
        round=round_num,
        phase=phase,
        event=BiddingEvent(
            description=f"bid {bid}",
            actor=PlayerActor(name="Alice"),
            bid=bid,
        ),
    )


@pytest.fixture
def sample_events() -> Sequence[LoggedEvent[GameEvent]]:
    return [
        make_system_event(1, GamePhase.OPENING, "1"),
        make_system_event(1, GamePhase.ROUND_EVENTS, "2"),
        make_bidding_event(1, GamePhase.BIDDING, 5),
        make_system_event(2, GamePhase.ROUND_EVENTS, "3"),
        make_bidding_event(2, GamePhase.BIDDING, 10),
        make_bidding_event(2, GamePhase.BIDDING, 15),
        make_system_event(3, GamePhase.ROUND_EVENTS, "4"),
    ]


@pytest.fixture
def extended_events(
    sample_events: Sequence[LoggedEvent[GameEvent]],
) -> Sequence[LoggedEvent[GameEvent]]:
    return list(
        itertools.chain.from_iterable(
            [
                sample_events,
                (
                    x.model_copy(update={"round": x.round + 3})
                    for x in sample_events
                ),
            ]
        )
    )


def test_empty_events() -> None:
    query = EventStream[GameEvent]([])
    assert not query


def test_basic_iteration(
    sample_events: Sequence[LoggedEvent[GameEvent]],
) -> None:
    query = EventStream(sample_events)
    assert list(query) == list(sample_events)


def test_filter_predicate(
    sample_events: Sequence[LoggedEvent[GameEvent]],
) -> None:
    query = EventStream(sample_events).filter(
        lambda e: "bid" in e.event.description
    )
    result = list(query)
    assert len(result) == 3
    assert all(isinstance(e.event, BiddingEvent) for e in result)


def test_of_type(sample_events: Sequence[LoggedEvent[GameEvent]]) -> None:
    query = EventStream(sample_events).of_type(BiddingEvent)
    result = list(query.unwrap())
    assert len(result) == 3
    assert all(isinstance(e, BiddingEvent) for e in result)

    # Type hinting check (mostly for mypy)
    first_bid = result[0]
    assert first_bid.bid in (5, 10, 15)


def test_in_phase(sample_events: Sequence[LoggedEvent[GameEvent]]) -> None:
    query = EventStream(sample_events).in_phase(GamePhase.BIDDING)
    result = list(query)
    assert len(result) == 3
    assert all(e.phase == GamePhase.BIDDING for e in result)


def test_in_round(sample_events: Sequence[LoggedEvent[GameEvent]]) -> None:
    query = EventStream(sample_events).in_round(2)
    result = list(query)
    assert len(result) == 3
    assert all(e.round == 2 or e.round is None for e in result)
    # The null round is filtered out by the strict equality in in_round
    assert all(e.round == 2 for e in result)


def test_take_while(sample_events: Sequence[LoggedEvent[GameEvent]]) -> None:
    query = EventStream(sample_events).take_while(lambda e: e.round < 2)
    result = list(query)
    assert len(result) == 3  # 1, 1, 1
    assert all(e.round < 2 for e in result)


def test_drop_while(sample_events: Sequence[LoggedEvent[GameEvent]]) -> None:
    query = EventStream(sample_events).drop_while(lambda e: e.round < 2)
    result = list(query)
    assert len(result) == 4  # 2, 2, 2, 3
    assert all(e.round >= 2 for e in result)


def test_take_latest_phase_block(
    sample_events: Sequence[LoggedEvent[GameEvent]],
) -> None:
    # Modify sample to have a nice block at the end
    events = list(sample_events)
    events.append(make_system_event(3, GamePhase.ROUND_EVENTS, "5"))
    events.append(make_system_event(3, GamePhase.ROUND_EVENTS, "6"))

    # Need to pass reversed sequence
    query = EventStream(reversed(events)).take_latest_phase_block()
    result = list(query)
    assert len(result) == 3
    assert result[0].event.description == "6"
    assert result[1].event.description == "5"
    assert result[2].event.description == "4"


def test_lazy_evaluation() -> None:
    # A sequence that crashes if iterated
    class CrashingSequence(Sequence[LoggedEvent[GameEvent]]):
        def __init__(self) -> None:
            self.length = 5

        @overload
        def __getitem__(self, index: int) -> LoggedEvent[GameEvent]: ...
        @overload
        def __getitem__(
            self, index: slice
        ) -> Sequence[LoggedEvent[GameEvent]]: ...
        def __getitem__(
            self, index: int | slice
        ) -> LoggedEvent[GameEvent] | Sequence[LoggedEvent[GameEvent]]:
            raise ValueError("Iterated!")

        def __len__(self) -> int:
            return self.length

    seq = CrashingSequence()
    query = EventStream(seq)

    # These should not crash
    q2 = query.filter(lambda e: True).in_round(5).of_type(BiddingEvent)

    # This should crash
    with pytest.raises(ValueError, match="Iterated!"):
        list(q2)


def test_bool_operator(sample_events: Sequence[LoggedEvent[GameEvent]]) -> None:
    # Ensure checking bool does not consume elements
    query = EventStream(sample_events)
    assert query
    assert len(list(query)) == len(sample_events)
    query = EventStream(sample_events)
    assert query

    query = EventStream(reversed(sample_events))
    assert query

    query = query.in_round(1)
    assert query

    query = EventStream[GameEvent]([])
    assert not query

    query = query.take_latest_phase_block()
    assert not query


def test_trivial_predicate(
    extended_events: Sequence[LoggedEvent[GameEvent]],
) -> None:
    query = EventStream(list(extended_events)).filter(lambda _: False)
    assert not query

    query = EventStream(list(extended_events)).filter(lambda _: True)
    assert list(query) == list(extended_events)


def test_composability(
    extended_events: Sequence[LoggedEvent[GameEvent]],
) -> None:
    query = (
        EventStream(extended_events)
        .drop_while(lambda e: e.round < 2)
        .take_while(lambda e: e.round <= 4)
        .filter(lambda e: e.phase == GamePhase.BIDDING)
        .of_type(BiddingEvent)
    )
    result = list(query.unwrap())
    assert len(result) == 3
    assert all(isinstance(e, BiddingEvent) for e in result)


def test_take_latest_phase_block_empty() -> None:
    query = EventStream[GameEvent]([]).take_latest_phase_block()
    assert not query
    assert list(query) == []


def test_take_rounds(extended_events: Sequence[LoggedEvent[GameEvent]]) -> None:
    query = EventStream(extended_events).between_rounds(2, 4)
    assert query

    results = list(query)
    assert len(results) == 7
    assert results == extended_events[3:10]

    query = EventStream(reversed(extended_events)).between_rounds(2, 4)
    assert query
    results = list(query)
    assert len(results) == 7
    assert results == list(reversed(extended_events[3:10]))

    query = EventStream(extended_events).between_rounds(4, 2)
    assert not query
    assert list(query) == []

    query = (
        EventStream(extended_events).between_rounds(1, 2).between_rounds(3, 4)
    )
    assert not query
    assert list(query) == []

    query = EventStream(extended_events).between_rounds(1, 1)
    assert query
    assert list(query) == list(EventStream(extended_events).in_round(1))

    query = EventStream(reversed(extended_events)).between_rounds(1, 1)
    assert query
    assert list(query) == list(
        EventStream(reversed(extended_events)).in_round(1)
    )

    query = EventStream[GameEvent]([]).between_rounds(1, 3)
    assert not query
    assert list(query) == []


def test_head(sample_events: Sequence[LoggedEvent[GameEvent]]) -> None:
    query = EventStream(sample_events)
    # Successive calls to head() consume elements
    assert query.head() == sample_events[0]
    assert query.head() == sample_events[1]

    # Empty stream without default raises EmptyStreamError
    query = EventStream[GameEvent]([])
    with pytest.raises(EmptyStreamError):
        query.head()

    # Empty stream with default returns the default
    assert query.head(default="foobar") == "foobar"
    assert query.head(default=None) is None

    my_object = object()
    assert query.head(default=my_object) is my_object

    # Filtering down to empty stream also raises
    query = EventStream(sample_events).of_type(OperationConductedEvent)
    with pytest.raises(EmptyStreamError):
        query.head()


def test_slice(sample_events: Sequence[LoggedEvent[GameEvent]]) -> None:
    # Test slice with stop only
    query = EventStream(sample_events).slice(3)
    assert list(query) == list(sample_events[:3])

    # Test slice with start and stop
    query = EventStream(sample_events).slice(2, 5)
    assert list(query) == list(sample_events[2:5])

    # Test slice with start, stop, and step
    query = EventStream(sample_events).slice(1, 6, 2)
    assert list(query) == list(sample_events[1:6:2])

    # Test that slicing an empty stream returns an empty stream
    query = EventStream[GameEvent]([]).slice(5)
    assert list(query) == []
