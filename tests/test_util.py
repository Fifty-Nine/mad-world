"""Tests for the utility module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_world.events import PlayerActor, SystemActor
from mad_world.util import (
    BadClampRangeError,
    aretry,
    clamp,
    cost_or_gain,
    defrag_escalation_track,
    escalation_bar,
    escalation_budget,
    extract_json_from_response,
    gain_or_lose,
    get_class_name,
    get_doomsday_bids,
    get_subclass_by_name,
    increase_or_decrease,
    pareto_optimal_bid,
    remove_ordering_prefix,
    reorder_schema_properties,
    wrap_text,
)

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT


def test_wrap_text_basic() -> None:
    text = (
        "This is a long line that should be wrapped because it exceeds the "
        "width of eighty characters."
    )
    wrapped = wrap_text(text, width=40)
    lines = wrapped.splitlines()
    assert len(lines) > 1
    for line in lines:
        assert len(line) <= 40


def test_wrap_text_preserve_newlines() -> None:
    text = "Line 1\nLine 2\n\nLine 4"
    wrapped = wrap_text(text)
    assert wrapped == "Line 1\nLine 2\n\nLine 4"


def test_wrap_text_indent() -> None:
    text = "Long line that needs wrapping"
    indent = "  "
    wrapped = wrap_text(text, indent=indent, width=15)
    lines = wrapped.splitlines()
    for line in lines:
        assert line.startswith(indent)
        assert len(line) <= 15


def test_wrap_text_empty_lines() -> None:
    text = "A\n\nB"
    wrapped = wrap_text(text, indent="  ")
    assert wrapped == "  A\n  \n  B"


def test_wrap_text_with_indentation() -> None:
    text = (
        "This is text\n"
        "  with its own embedded indentation:\n"
        "    here's an even more deeply nested line that exceeds the width "
        "limit and should be wrapped but preserve the nested indentation.\n"
        "    and here's another indented line.\n"
        "\n"
        "and finally here's an unindented line.\n"
    )
    wrapped = wrap_text(text, width=40)
    assert wrapped == (
        "This is text\n"
        "  with its own embedded indentation:\n"
        "    here's an even more deeply nested\n"
        "    line that exceeds the width limit\n"
        "    and should be wrapped but preserve\n"
        "    the nested indentation.\n"
        "    and here's another indented line.\n"
        "\n"
        "and finally here's an unindented line.\n"
    )


def test_wrap_empty() -> None:
    assert wrap_text("") == ""
    assert wrap_text("", indent="  ") == ""


@pytest.mark.parametrize(
    ("clock", "max_clock", "bid"),
    [
        (0, 25, 12),
        (24, 25, 0),
        (23, 25, 0),
        (16, 25, 4),
        (0, 30, 14),
        (29, 30, 0),
        (28, 30, 0),
        (27, 30, 1),
        (16, 30, 6),
    ],
)
def test_escalation_budget(clock: int, max_clock: int, bid: int) -> None:
    assert escalation_budget(clock, max_clock) == bid


@pytest.mark.parametrize(
    ("clock", "max_clock", "allowed_bids", "bid"),
    [
        (0, 30, [0, 10], 10),
        (0, 25, [0, 1, 2, 3], 3),
        (29, 30, [0, 1], 0),
        (25, 30, [0, 1, 3, 5, 10], 1),
        (20, 30, [0, 1, 3, 5, 10], 3),
        (0, 1, [], 0),
    ],
)
def test_pareto_optimal_bid(
    clock: int,
    max_clock: int,
    allowed_bids: list[int],
    bid: int,
) -> None:
    assert pareto_optimal_bid(clock, max_clock, allowed_bids) == bid


def test_get_class_name() -> None:
    assert get_class_name("crazy_ivan") == "CrazyIvan"
    assert get_class_name("crazy-ivan") == "CrazyIvan"
    assert get_class_name("crazyIvan") == "CrazyIvan"
    assert get_class_name("CrazyIvan") == "CrazyIvan"
    assert get_class_name("pacifist") == "Pacifist"


def test_get_subclass_by_name() -> None:
    class Base(ABC):
        def __init__(self, name: str, value: int = 0) -> None:
            self.name = name
            self.value = value

        @abstractmethod
        def do_something(self) -> None:
            pass

    class SubClass(Base):
        def do_something(self) -> None:
            pass

    class AnotherSub(Base):
        def do_something(self) -> None:
            pass

    class NotASub:
        pass

    module = __name__

    # Test various naming formats and instantiation
    # SubClass with positional arg
    obj = get_subclass_by_name(module, "SubClass", Base, "test-1")
    assert isinstance(obj, SubClass)
    assert obj.name == "test-1"
    assert obj.value == 0

    # sub_class with keyword arg
    obj = get_subclass_by_name(module, "sub_class", Base, "test-2", value=42)
    assert isinstance(obj, SubClass)
    assert obj.name == "test-2"
    assert obj.value == 42

    # sub-class
    obj = get_subclass_by_name(module, "sub-class", Base, "test-3")
    assert isinstance(obj, SubClass)

    # SUB_CLASS
    obj = get_subclass_by_name(module, "SUB_CLASS", Base, "test-4")
    assert isinstance(obj, SubClass)

    # another_sub
    obj = get_subclass_by_name(module, "another_sub", Base, "test-5")
    assert isinstance(obj, AnotherSub)

    # Should return None for:
    # 1. Classes that don't exist
    assert get_subclass_by_name(module, "Unknown", Base, "test") is None
    # 2. Classes that are not subclasses of the expected type
    # (NotASub is not in Base.__subclasses__())
    assert get_subclass_by_name(module, "not_a_sub", Base, "test") is None
    # 3. The base class itself (it's not in its own __subclasses__())
    assert get_subclass_by_name(module, "Base", Base, "test") is None
    # 4. Wrong module
    assert (
        get_subclass_by_name("wrong.module", "SubClass", Base, "test") is None
    )


@pytest.mark.parametrize(
    ("clock", "max_clock", "allowed_bids", "risky", "deadly"),
    [
        (0, 25, (0, 1, 3, 5, 8), [], []),
        (24, 25, (0, 1, 3, 5, 8), [(0, 3), (1, 1)], [3, 5, 8]),
        (20, 25, (0, 1, 3, 5, 8), [(0, 8), (1, 5), (3, 3), (5, 1)], [8]),
        (10, 25, (0, 1, 3, 5, 8), [(8, 8)], []),
        (0, 1, (0, 1, 3, 5, 8), [(0, 3), (1, 1)], [3, 5, 8]),
        (0, 1, (0,), [], []),
        (29, 30, (0, 1, 3, 5, 10), [(0, 3), (1, 1)], [3, 5, 10]),
    ],
)
def test_get_doomsday_bids(
    clock: int,
    max_clock: int,
    allowed_bids: list[int],
    risky: list[tuple[int, int]],
    deadly: list[int],
) -> None:
    assert all(b not in deadly for (b, _) in risky)
    assert get_doomsday_bids(clock, max_clock, -1, allowed_bids) == (
        risky,
        deadly,
    )


def test_reorder_schema_properties() -> None:
    schema = {
        "properties": {
            "foo": {"type": "string"},
            "action": {"type": "string"},
            "bar": {"type": "number"},
        },
        "required": ["foo", "action"],
        "$defs": {
            "sub": {
                "properties": {
                    "baz": {"type": "boolean"},
                    "action": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    }

    reordered = reorder_schema_properties(schema, "foo")

    # Check main properties
    assert "99_foo" in reordered["properties"]
    assert "01_bar" in reordered["properties"]
    assert "00_action" in reordered["properties"]
    assert "foo" not in reordered["properties"]
    assert "bar" not in reordered["properties"]
    assert "action" not in reordered["properties"]

    # Check required
    assert "99_foo" in reordered["required"]
    assert "00_action" in reordered["required"]
    assert "foo" not in reordered["required"]
    assert "action" not in reordered["required"]

    # Check $defs
    sub = reordered["$defs"]["sub"]
    assert "00_baz" in sub["properties"]
    assert "01_action" in sub["properties"]
    assert "01_action" in sub["required"]


def test_reorder_schema_properties_no_action() -> None:
    schema = {
        "properties": {
            "foo": {"type": "string"},
        },
        "required": ["foo", "action"],
    }
    reordered = reorder_schema_properties(schema, "action")
    assert "00_foo" in reordered["properties"]
    assert "99_action" not in reordered["properties"]
    assert "00_foo" in reordered["required"]
    assert "99_action" in reordered["required"]
    assert "foo" not in reordered["required"]
    assert "action" not in reordered["required"]


def test_reorder_schema_properties_empty() -> None:
    schema: dict[str, Any] = {}
    assert reorder_schema_properties(schema, "action") == {}


def test_remove_ordering_prefix_basic() -> None:
    assert remove_ordering_prefix("00_foo", is_key=True) == "foo"
    assert remove_ordering_prefix("foo", is_key=True) == "foo"
    assert remove_ordering_prefix("00_foo", is_key=False) == "00_foo"


def test_remove_ordering_prefix_recursive() -> None:
    obj = {
        "00_foo": "01_bar",
        "02_baz": ["03_qux", {"04_nested": "05_val"}],
        "simple": 123,
    }
    expected = {
        "foo": "01_bar",
        "baz": ["03_qux", {"nested": "05_val"}],
        "simple": 123,
    }
    assert remove_ordering_prefix(obj) == expected


def test_remove_ordering_prefix_other_types() -> None:
    assert remove_ordering_prefix(123) == 123
    assert remove_ordering_prefix(None) is None
    assert remove_ordering_prefix(3.14) == 3.14


@pytest.mark.parametrize(
    ("value", "inc_dec", "cost_gain", "gain_lose"),
    [
        (100, "increase", "gain", "gain"),
        (-100, "decrease", "cost", "lose"),
        (0, "increase", "gain", "gain"),
        (-1, "decrease", "cost", "lose"),
        (1, "increase", "gain", "gain"),
    ],
)
def test_increase_or_decrease(
    value: int, inc_dec: str, cost_gain: str, gain_lose: str
) -> None:
    assert increase_or_decrease(value) == inc_dec
    assert cost_or_gain(value) == cost_gain
    assert gain_or_lose(value) == gain_lose


@pytest.mark.parametrize(
    ("val", "min_val", "max_val", "expect"),
    [
        (5, 0, 10, 5),
        (-1, 0, 10, 0),
        (20, 0, 10, 10),
        (-10, -20, -5, -10),
        (-30, -20, -5, -20),
        (-1, -20, -5, -5),
        (0.0, -10.0, 10.0, 0.0),
        (-10.1, -10.0, 10.0, -10.0),
        (10.1, -10.0, 10.0, 10.0),
        ("foo", "aaa", "zzz", "foo"),
        ("ccc", "aaa", "bbb", "bbb"),
    ],
)
def test_clamp(
    val: SupportsRichComparisonT,
    min_val: SupportsRichComparisonT,
    max_val: SupportsRichComparisonT,
    expect: SupportsRichComparisonT,
) -> None:
    assert clamp(val, min_val, max_val) == expect


def test_clamp_range() -> None:
    with pytest.raises(BadClampRangeError):
        clamp(0, 10, 0)

    with pytest.raises(BadClampRangeError):
        clamp(0.0, 10.0, 0.0)

    with pytest.raises(BadClampRangeError):
        clamp("foo", "zzz", "aaa")


def test_escalation_bar() -> None:
    p1 = PlayerActor(name="Bar")
    p2 = PlayerActor(name="Foo")
    sys = SystemActor()
    track = [None, sys, p2, None, sys, p1]
    assert defrag_escalation_track(track) == [p1, p2, sys, sys, None, None]

    assert escalation_bar(track, defrag=True) == (
        "+------+\n|BFxx  |\n+------+\n"
    )

    assert escalation_bar(track, defrag=False) == (
        "+------+\n| xF xB|\n+------+\n"
    )


# Test data for extract_json_from_response
_TEST_JSON_EXTRACT_CASES: list[tuple[str, str]] = [
    # JSON in markdown block with language specifier
    (
        '```json\n{"action": "bid", "bid": 5}\n```',
        '{"action": "bid", "bid": 5}',
    ),
    # JSON in markdown block without language specifier
    (
        '```\n{"action": "bid", "bid": 5}\n```',
        '{"action": "bid", "bid": 5}',
    ),
    # Multiple code blocks - should get the last one
    (
        (
            '```json\n{"first": 1}\n```\n\nSome text\n\n'
            '```json\n{"second": 2}\n```'
        ),
        '{"second": 2}',
    ),
    # Code block with extra whitespace inside
    (
        '```json\n  {\n    "foo": "bar"\n  }\n```',
        '{\n    "foo": "bar"\n  }',
    ),
    # No code blocks - return original
    (
        "Just plain text without any code blocks",
        "Just plain text without any code blocks",
    ),
    # Empty code block
    (
        "```\nSome text",
        "```\nSome text",
    ),
    # Code block with trailing content after closing ```
    (
        '```json\n{"foo": "bar"}\n```\nThis should be ignored',
        '{"foo": "bar"}',
    ),
    # Nested backticks in text (edge case)
    (
        ('Some text `not a block` and then ```json\n{"real": "json"}\n```'),
        '{"real": "json"}',
    ),
    # Only closing backticks
    (
        "```",
        "```",
    ),
    # No closing backticks - should return original
    (
        '```json\n{"incomplete": true',
        '```json\n{"incomplete": true',
    ),
    # JSON with special characters
    (
        '```json\n{"message": "It\'s a test", "count": 42}\n```',
        '{"message": "It\'s a test", "count": 42}',
    ),
    # Multiple lines of whitespace between backticks
    (
        '```json\n\n\n{"foo": "bar"}\n\n\n```',
        '{"foo": "bar"}',
    ),
    # Raw JSON without markdown blocks
    (
        'Here is the result: {"action": "bid", "bid": 5}',
        '{"action": "bid", "bid": 5}',
    ),
    # Multiple objects, should get the last one
    (
        'First {"a": 1} then {"b": 2}',
        '{"b": 2}',
    ),
    # Nested JSON
    ('{"nested": {"inner_key": 1}}', '{"nested": {"inner_key": 1}}'),
    # Multiple nested JSON objects
    (
        'First {"a": {"x": 1}} then {"b": {"y": 2}}',
        '{"b": {"y": 2}}',
    ),
    # Invalid JSON followed by valid JSON
    (
        'Bad {a: 1} Good {"b": 2}',
        '{"b": 2}',
    ),
    # Valid JSON followed by invalid JSON
    (
        'Good {"a": 1} Bad {b: 2}',
        '{"a": 1}',
    ),
    # Valid JSON inside string literal of another valid JSON
    (
        '{"message": "payload is {\\"key\\": \\"value\\"}"}',
        '{"message": "payload is {\\"key\\": \\"value\\"}"}',
    ),
]


@pytest.mark.parametrize(
    ("response", "expected"),
    _TEST_JSON_EXTRACT_CASES,
)
def test_extract_json_from_response(
    response: str,
    expected: str,
) -> None:
    assert extract_json_from_response(response) == expected


@pytest.mark.asyncio
async def test_aretry() -> None:
    cb = AsyncMock(return_value=1)
    assert await aretry(func=cb, allowed_exceptions=[]) == 1
    cb.side_effect = ValueError("my-text")
    with pytest.raises(ValueError, match="my-text"):
        await aretry(func=cb, allowed_exceptions=[])

    assert await aretry(func=cb, allowed_exceptions=[ValueError]) is None

    cb.side_effect = [ValueError("once"), TypeError("twice"), 0]
    assert (
        await aretry(func=cb, allowed_exceptions=[ValueError, TypeError]) == 0
    )


@pytest.mark.asyncio
async def test_aretry_on_error() -> None:
    cb = AsyncMock(return_value=1)
    on_error = MagicMock()
    assert await aretry(func=cb, allowed_exceptions=[], on_error=on_error) == 1
    assert not on_error.called

    cb.side_effect = ValueError("my-text")
    with pytest.raises(ValueError, match="my-text"):
        await aretry(func=cb, allowed_exceptions=[], on_error=on_error)

    assert not on_error.called

    assert (
        await aretry(
            func=cb,
            allowed_exceptions=[ValueError],
            count=10,
            on_error=on_error,
        )
        is None
    )
    assert on_error.call_count == 10
    assert on_error.call_count == 10

    on_error.reset_mock()

    cb.side_effect = [ValueError("once"), TypeError("twice"), 0]
    assert (
        await aretry(
            func=cb,
            allowed_exceptions=[ValueError, TypeError],
            count=2,
            on_error=on_error,
        )
        is None
    )
    assert on_error.call_count == 2

    on_error.reset_mock()
    cb.side_effect = [ValueError("once"), TypeError("twice"), 0]
    assert (
        await aretry(
            func=cb,
            allowed_exceptions=[ValueError, TypeError],
            count=3,
            on_error=on_error,
        )
        == 0
    )
    assert on_error.call_count == 2
