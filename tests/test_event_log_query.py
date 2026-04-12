from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mad_world.enums import GamePhase
from mad_world.events import (
    ActionEvent,
    BiddingEvent,
    CrisisResolutionEvent,
    EventKind,
    EventLogQuery,
    MessageEvent,
    OperationConductedEvent,
    PlayerActor,
    StateEvent,
    SystemActor,
    SystemEvent,
)

if TYPE_CHECKING:
    from mad_world.core import GameState


def test_event_log_query_reverse() -> None:
    events = [
        SystemEvent(description="1"),
        SystemEvent(description="2"),
        SystemEvent(description="3"),
    ]
    query = EventLogQuery(events)
    rev_query = query.reverse()
    assert [e.description for e in rev_query.to_list()] == ["3", "2", "1"]
    fwd_query = rev_query.reverse()
    assert [e.description for e in fwd_query.to_list()] == ["1", "2", "3"]


def test_event_log_query_of_type() -> None:
    events = [
        SystemEvent(description="sys1"),
        StateEvent(description="state1"),
        SystemEvent(description="sys2"),
    ]
    query = EventLogQuery(events)
    sys_query = query.of_type(SystemEvent)
    sys_events = sys_query.to_list()
    assert len(sys_events) == 2
    assert sys_events[0].description == "sys1"
    assert sys_events[1].description == "sys2"


def test_event_log_query_filter() -> None:
    events = [
        SystemEvent(description="sys1", current_round=1),
        SystemEvent(description="sys2", current_round=2),
    ]
    query = EventLogQuery(events)
    filtered = query.filter(lambda e: e.current_round == 1).to_list()
    assert len(filtered) == 1
    assert filtered[0].description == "sys1"


def test_event_log_query_in_phase() -> None:
    events = [
        SystemEvent(description="1", current_phase=GamePhase.BIDDING),
        SystemEvent(description="2", current_phase=GamePhase.OPERATIONS),
    ]
    bidding = EventLogQuery(events).in_phase(GamePhase.BIDDING).to_list()
    assert len(bidding) == 1
    assert bidding[0].description == "1"


def test_event_log_query_in_rounds() -> None:
    events = [
        SystemEvent(description="1", current_round=1),
        SystemEvent(description="2", current_round=1),
        SystemEvent(description="3", current_round=2),
        SystemEvent(description="4", current_round=3),
        SystemEvent(description="5", current_round=4),
    ]
    query = EventLogQuery(events)
    res_fwd = query.in_rounds(2, 3).to_list()
    assert len(res_fwd) == 2
    assert [e.description for e in res_fwd] == ["3", "4"]

    res_rev = query.reverse().in_rounds(2, 3).to_list()
    assert len(res_rev) == 2
    assert [e.description for e in res_rev] == ["4", "3"]

    res_round = query.in_round(2).to_list()
    assert len(res_round) == 1
    assert res_round[0].description == "3"


def test_event_log_query_in_recent_phase() -> None:
    events = [
        SystemEvent(
            description="1", current_round=1, current_phase=GamePhase.BIDDING
        ),
        SystemEvent(
            description="2", current_round=1, current_phase=GamePhase.OPERATIONS
        ),
        SystemEvent(
            description="3", current_round=2, current_phase=GamePhase.BIDDING
        ),
        SystemEvent(
            description="4", current_round=2, current_phase=GamePhase.BIDDING
        ),
    ]
    query = EventLogQuery(events)

    recent_fwd = query.in_recent_phase().to_list()
    assert len(recent_fwd) == 2
    assert [e.description for e in recent_fwd] == ["3", "4"]

    recent_rev = query.reverse().in_recent_phase().to_list()
    assert len(recent_rev) == 2
    assert [e.description for e in recent_rev] == ["4", "3"]


def test_event_log_query_in_recent_phase_empty() -> None:
    query: EventLogQuery[SystemEvent] = EventLogQuery(
        cast("list[SystemEvent]", [])
    )
    assert len(query.in_recent_phase().to_list()) == 0


def test_event_log_query_integration() -> None:
    events = [
        SystemEvent(
            description="1", current_round=1, current_phase=GamePhase.BIDDING
        ),
        SystemEvent(
            description="2", current_round=2, current_phase=GamePhase.BIDDING
        ),
        ActionEvent(
            actor=PlayerActor(name="Alpha"),
            description="3",
            current_round=2,
            current_phase=GamePhase.OPERATIONS,
        ),
        SystemEvent(
            description="4", current_round=2, current_phase=GamePhase.OPERATIONS
        ),
        SystemEvent(
            description="5", current_round=3, current_phase=GamePhase.BIDDING
        ),
        SystemEvent(
            description="6",
            current_round=3,
            current_phase=GamePhase.ROUND_EVENTS,
        ),
    ]
    query = EventLogQuery(events)

    res = (
        query.in_round(2)
        .in_phase(GamePhase.OPERATIONS)
        .of_type(ActionEvent)
        .to_list()
    )
    assert len(res) == 1
    assert res[0].description == "3"

    res2 = query.reverse().in_rounds(1, 2).of_type(SystemEvent).to_list()
    from mad_world.events import StateEvent  # noqa: PLC0415

    assert len(res2) == 3
    assert [e.description for e in res2] == ["4", "2", "1"]

    m = MessageEvent(actor=PlayerActor(name="x"), description="y")
    b = BiddingEvent(actor=PlayerActor(name="x"), description="y", bid=1)
    o = OperationConductedEvent(
        actor=PlayerActor(name="x"), description="y", operation="z"
    )
    c = CrisisResolutionEvent(actor=PlayerActor(name="x"), description="y")
    c = CrisisResolutionEvent(actor=PlayerActor(name="x"), description="y")
    s = StateEvent(description="y")

    assert m.event_kind == EventKind.MESSAGE
    assert b.event_kind == EventKind.BIDDING
    assert o.event_kind == EventKind.OPERATION_CONDUCTED
    assert c.event_kind == EventKind.CRISIS_RESOLUTION
    assert s.event_kind == EventKind.STATE
    from mad_world.events import GameEvent  # noqa: PLC0415

    assert GameEvent is not None
    sys = SystemEvent(description="sys")
    from mad_world.events import SystemActor  # noqa: PLC0415

    sys_act = SystemActor()
    from mad_world.events import ActorKind  # noqa: PLC0415

    assert sys_act.actor_kind == ActorKind.SYSTEM
    from mad_world.events import BaseGameEvent  # noqa: PLC0415

    base = BaseGameEvent(description="base")
    assert base.done_by_player("Alpha") is False
    action_ev = ActionEvent(
        actor=PlayerActor(name="Beta"), description="action"
    )
    assert action_ev.done_by_player("Alpha") is False

    msg = MessageEvent(
        actor=PlayerActor(name="Beta"),
        description="msg",
    )
    assert msg.done_by_player("Alpha") is False
    assert sys.event_kind == EventKind.SYSTEM


def test_game_state_query_events(basic_game: GameState) -> None:
    basic_game.event_log = [
        SystemEvent(
            description="1", current_round=1, current_phase=GamePhase.BIDDING
        ),
        SystemEvent(
            description="2", current_round=2, current_phase=GamePhase.OPERATIONS
        ),
    ]
    res = EventLogQuery(basic_game.event_log).in_round(2).to_list()
    assert len(res) == 1
    assert res[0].description == "2"


def test_events_actors_methods() -> None:
    s = SystemActor()
    assert s.is_system() is True
    assert s.player() is None

    p = PlayerActor(name="Alpha")
    assert p.is_system() is False
    assert p.player() == "Alpha"

    a = ActionEvent(actor=p, description="action")
    assert a.done_by_player("Alpha") is True
    assert a.done_by_player("Omega") is False


def test_event_log_query_first() -> None:
    events = [SystemEvent(description="1"), SystemEvent(description="2")]
    query = EventLogQuery(events)
    first_event = query.first()
    assert first_event is not None
    assert first_event.description == "1"

    empty_query: EventLogQuery[SystemEvent] = EventLogQuery(
        cast("list[SystemEvent]", [])
    )
    assert empty_query.first() is None


def test_event_log_query_iter() -> None:
    events = [SystemEvent(description="1")]
    query = EventLogQuery(events)
    for e in query:
        assert e.description == "1"
