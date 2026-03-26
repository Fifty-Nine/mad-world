"""Tests for the utility module."""

import pytest

from mad_world.util import escalation_budget, pareto_optimal_bid, wrap_text


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


@pytest.mark.parametrize(
    "clock,max_clock,bid",
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
    "clock,max_clock,allowed_bids,bid",
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
    clock: int, max_clock: int, allowed_bids: list[int], bid: int
) -> None:
    assert pareto_optimal_bid(clock, max_clock, allowed_bids) == bid
