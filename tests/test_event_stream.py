from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, overload

import pytest

from mad_world.enums import GamePhase
from mad_world.event_stream import EventStream
from mad_world.events import (
    BiddingEvent,
    GameEvent,
    PlayerActor,
    SystemActor,
    SystemEvent,
)


def make_system_event(
    round_num: int | None, phase: GamePhase | None, desc: str = "sys"
) -> SystemEvent:
    return SystemEvent(
        description=desc,
        current_round=round_num,
        current_phase=phase,
        actor=SystemActor(),
    )


def make_bidding_event(
    round_num: int | None, phase: GamePhase | None, bid: int
) -> BiddingEvent:
    return BiddingEvent(
        description=f"bid {bid}",
        current_round=round_num,
        current_phase=phase,
        actor=PlayerActor(name="Alice"),
        bid=bid,
    )


@pytest.fixture
def sample_events() -> Sequence[GameEvent]:
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
def extended_events(sample_events: Sequence[GameEvent]) -> Sequence[GameEvent]:
    return list(
        itertools.chain.from_iterable(
            [
                sample_events,
                (
                    x.model_copy(update={"current_round": x.round + 3})
                    for x in sample_events
                ),
            ]
        )
    )


def test_empty_events() -> None:
    query = EventStream[GameEvent]([])
    assert not query


def test_basic_iteration(sample_events: Sequence[GameEvent]) -> None:
    query = EventStream(sample_events)
    assert list(query) == list(sample_events)


def test_filter_predicate(sample_events: Sequence[GameEvent]) -> None:
    query = EventStream(sample_events).filter(lambda e: "bid" in e.description)
    result = list(query)
    assert len(result) == 3
    assert all(isinstance(e, BiddingEvent) for e in result)


def test_of_type(sample_events: Sequence[GameEvent]) -> None:
    query = EventStream(sample_events).of_type(BiddingEvent)
    result = list(query)
    assert len(result) == 3
    assert all(isinstance(e, BiddingEvent) for e in result)

    # Type hinting check (mostly for mypy)
    first_bid = result[0]
    assert first_bid.bid in (5, 10, 15)


def test_in_phase(sample_events: Sequence[GameEvent]) -> None:
    query = EventStream(sample_events).in_phase(GamePhase.BIDDING)
    result = list(query)
    assert len(result) == 3
    assert all(e.current_phase == GamePhase.BIDDING for e in result)


def test_in_round(sample_events: Sequence[GameEvent]) -> None:
    query = EventStream(sample_events).in_round(2)
    result = list(query)
    assert len(result) == 3
    assert all(e.round == 2 or e.round is None for e in result)
    # The null round is filtered out by the strict equality in in_round
    assert all(e.round == 2 for e in result)


def test_take_while(sample_events: Sequence[GameEvent]) -> None:
    query = EventStream(sample_events).take_while(lambda e: e.round < 2)
    result = list(query)
    assert len(result) == 3  # 1, 1, 1
    assert all(e.round < 2 for e in result)


def test_drop_while(sample_events: Sequence[GameEvent]) -> None:
    query = EventStream(sample_events).drop_while(lambda e: e.round < 2)
    result = list(query)
    assert len(result) == 4  # 2, 2, 2, 3
    assert all(e.round >= 2 for e in result)


def test_take_latest_phase_block(sample_events: Sequence[GameEvent]) -> None:
    # Modify sample to have a nice block at the end
    events = list(sample_events)
    events.append(make_system_event(3, GamePhase.ROUND_EVENTS, "5"))
    events.append(make_system_event(3, GamePhase.ROUND_EVENTS, "6"))

    # Need to pass reversed sequence
    query = EventStream(reversed(events)).take_latest_phase_block()
    result = list(query)
    assert len(result) == 3
    assert result[0].description == "6"
    assert result[1].description == "5"
    assert result[2].description == "4"


def test_lazy_evaluation() -> None:
    # A sequence that crashes if iterated
    class CrashingSequence(Sequence[GameEvent]):
        def __init__(self) -> None:
            self.length = 5

        @overload
        def __getitem__(self, index: int) -> GameEvent: ...
        @overload
        def __getitem__(self, index: slice) -> Sequence[GameEvent]: ...
        def __getitem__(self, index: Any) -> Any:
            raise ValueError("Iterated!")

        def __len__(self) -> int:
            return self.length

    seq = CrashingSequence()
    query = EventStream(seq)

    # These should not crash
    q2 = query.filter(lambda e: True).of_type(BiddingEvent).in_round(5)

    # This should crash
    with pytest.raises(ValueError, match="Iterated!"):
        list(q2)


def test_bool_operator(sample_events: Sequence[GameEvent]) -> None:
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

    query = EventStream([])
    assert not query

    query = query.take_latest_phase_block()
    assert not query


def test_trivial_predicate(extended_events: Sequence[GameEvent]) -> None:
    query = EventStream(list(extended_events)).filter(lambda _: False)
    assert not query

    query = EventStream(list(extended_events)).filter(lambda _: True)
    assert list(query) == list(extended_events)


def test_composability(extended_events: Sequence[GameEvent]) -> None:
    query = (
        EventStream(extended_events)
        .drop_while(lambda e: e.round < 2)
        .take_while(lambda e: e.round <= 4)
        .filter(lambda e: e.current_phase == GamePhase.BIDDING)
        .of_type(BiddingEvent)
    )
    result = list(query)
    assert len(result) == 3  # Two bids in round 2, one bid in round 4
    assert all(isinstance(e, BiddingEvent) for e in result)
    assert all(e.current_phase == GamePhase.BIDDING for e in result)
    assert all(e.round in (2, 4) for e in result)


def test_take_latest_phase_block_empty() -> None:
    query = EventStream[GameEvent]([]).take_latest_phase_block()
    assert not query
    assert list(query) == []
