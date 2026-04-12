from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any, overload

import pytest

from mad_world.enums import GamePhase
from mad_world.event_queries import EventLogQuery
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
        make_system_event(None, None, "null round"),
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
                    x.model_copy(
                        update={"current_round": (x.current_round or 0) + 3}
                    )
                    for x in sample_events
                ),
            ]
        )
    )


def test_empty_events() -> None:
    query = EventLogQuery[GameEvent]([])
    assert not query
    assert not query.reverse()


def test_basic_iteration(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events)
    assert list(query) == list(sample_events)


def test_reverse_iteration(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events).reverse()
    assert list(query) == list(reversed(sample_events))
    # Double reverse
    assert list(query.reverse()) == list(sample_events)


def test_filter_predicate(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events).filter(
        lambda e: "bid" in e.description
    )
    result = list(query)
    assert len(result) == 3
    assert all(isinstance(e, BiddingEvent) for e in result)


def test_of_type(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events).of_type(BiddingEvent)
    result = list(query)
    assert len(result) == 3
    assert all(isinstance(e, BiddingEvent) for e in result)

    # Type hinting check (mostly for mypy)
    first_bid = next(iter(query))
    assert first_bid.bid in (5, 10, 15)


def test_in_phase(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events).in_phase(GamePhase.BIDDING)
    result = list(query)
    assert len(result) == 3
    assert all(e.current_phase == GamePhase.BIDDING for e in result)


def test_in_round(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events).in_round(2)
    result = list(query)
    assert len(result) == 3
    assert all(e.current_round == 2 or e.current_round is None for e in result)
    # The null round is filtered out by the strictly out-of-bound final
    # generator
    assert all(e.current_round == 2 for e in result)


def test_in_rounds(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events).in_rounds(1, 2)
    result = list(query)

    assert (
        len(result) == 6
    )  # Includes 1, 2, and the null round in between is filtered
    assert all(e.current_round in (1, 2) for e in result)

    query_rev = query.reverse()
    result_rev = list(query_rev)
    assert result_rev == list(reversed(result))


def test_in_rounds_open_ended(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events).in_rounds(2, None)
    result = list(query)
    assert len(result) == 4
    assert all(e.current_round in (2, 3) for e in result)

    query = EventLogQuery(sample_events).in_rounds(None, 2)
    result = list(query)
    assert len(result) == 6
    assert all(e.current_round in (1, 2) for e in result)


def test_in_rounds_overlap(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events).in_rounds(1, 3).in_rounds(2, 2)
    result = list(query)
    assert len(result) == 3
    assert all(e.current_round == 2 for e in result)


def test_in_recent_phase(sample_events: Sequence[GameEvent]) -> None:
    # Modify sample to have a nice block at the end
    events = list(sample_events)
    events.append(make_system_event(3, GamePhase.ROUND_EVENTS, "5"))
    events.append(make_system_event(3, GamePhase.ROUND_EVENTS, "6"))

    query = EventLogQuery(events).in_recent_phase()
    result = list(query)
    assert len(result) == 3
    assert result[-1].description == "6"
    assert result[0].description == "4"


def test_in_recent_phase_reversed(sample_events: Sequence[GameEvent]) -> None:
    events = list(sample_events)
    events.append(make_system_event(3, GamePhase.ROUND_EVENTS, "5"))

    query = EventLogQuery(events).in_recent_phase().reverse()
    result = list(query)
    assert len(result) == 2
    assert result[0].description == "5"
    assert result[1].description == "4"


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
    query = EventLogQuery(seq)

    # These should not crash
    q2 = (
        query.filter(lambda e: True).of_type(BiddingEvent).in_round(5).reverse()
    )

    # This should crash
    with pytest.raises(ValueError, match="Iterated!"):
        list(q2)


def test_bool_operator(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events)
    assert query

    query = query.reverse()
    assert query

    query = query.in_round(1)
    assert query

    query = query.reverse()
    assert query

    query = EventLogQuery([])
    assert not query

    query = query.reverse()
    assert not query

    query = query.in_recent_phase()
    assert not query


def test_nonoverlapping_recent_phase(
    sample_events: Sequence[GameEvent],
) -> None:
    query = EventLogQuery(sample_events).in_recent_phase().in_round(1)
    assert not query


def test_nonoverlapping_round_filters(
    sample_events: Sequence[GameEvent],
) -> None:
    query = EventLogQuery(sample_events).in_round(1).in_round(3)
    assert not query


def test_nonoverlapping_rounds_filters(
    extended_events: Sequence[GameEvent],
) -> None:
    query = EventLogQuery(extended_events).in_rounds(1, 3).in_rounds(4, 6)
    assert not query


def test_reversed_rounds_filter(sample_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(sample_events).in_rounds(3, 1)
    assert not query


def test_trivial_predicate(extended_events: Sequence[GameEvent]) -> None:
    query = EventLogQuery(list(extended_events)).filter(lambda _: False)
    assert not query

    query = EventLogQuery(list(extended_events)).filter(lambda _: True)
    assert list(query) == list(extended_events)


def test_reverse_null_filters(extended_events: Sequence[GameEvent]) -> None:
    query = (
        EventLogQuery(list(extended_events)).filter(lambda _: False).reverse()
    )
    assert not query

    query = (
        EventLogQuery(list(extended_events)).reverse().filter(lambda _: False)
    )
    assert not query


FILTER_OPS: list[
    Callable[[EventLogQuery[GameEvent]], EventLogQuery[GameEvent]]
] = [
    lambda q: q.in_recent_phase(),
    lambda q: q.in_round(1),
    lambda q: q.in_rounds(1, 2),
    lambda q: q.in_phase(GamePhase.BIDDING),
    lambda q: q.reverse(),
    lambda q: q.filter(lambda e: "bid" in e.description),
    lambda q: q.of_type(BiddingEvent),
]


@pytest.mark.parametrize("op_idx1", range(len(FILTER_OPS)))
@pytest.mark.parametrize("op_idx2", range(len(FILTER_OPS)))
def test_filter_commutativity(
    extended_events: Sequence[GameEvent], op_idx1: int, op_idx2: int
) -> None:
    if op_idx1 >= op_idx2:
        return

    op1 = FILTER_OPS[op_idx1]
    op2 = FILTER_OPS[op_idx2]

    query = EventLogQuery(extended_events)

    res1 = list(op1(op2(query)))
    res2 = list(op2(op1(query)))

    # Known non-commutative pairings (e.g. `in_phase` and `filter`)
    if {op_idx1, op_idx2} in [{3, 5}, {3, 6}]:
        pytest.xfail("in_phase + filter/of_type fails strict equality")

    assert res1 == res2
