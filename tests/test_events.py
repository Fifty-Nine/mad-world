"""Tests for GameEvent types and methods."""

from __future__ import annotations

import pytest

from mad_world.events import (
    ActionEvent,
    GameEvent,
    MessageEvent,
    PlayerActor,
    StateEvent,
    SystemEvent,
)


@pytest.mark.parametrize(
    ("query_player", "event", "expected"),
    [
        ("any", SystemEvent(description=""), False),
        ("player", SystemEvent(description="player"), False),
        ("any", StateEvent(description=""), False),
        ("player", StateEvent(description="player"), False),
        (
            "foo",
            ActionEvent(actor=PlayerActor(name="bar"), description="foo"),
            False,
        ),
        (
            "foo",
            ActionEvent(actor=PlayerActor(name="foo"), description="bar"),
            True,
        ),
        (
            "bar",
            MessageEvent(
                actor=PlayerActor(name="foo"),
                description=("foo sent a message to bar:\n  sup?"),
                message="sup?",
                channel_message=False,
            ),
            False,
        ),
        (
            "bar",
            MessageEvent(
                actor=PlayerActor(name="bar"),
                description=("bar sent a message to foo:\n  nmu?"),
                message="nmu?",
                channel_message=False,
            ),
            True,
        ),
    ],
)
def test_done_by_player(
    *, query_player: str, event: GameEvent, expected: bool
) -> None:
    assert event.done_by_player(query_player) == expected
