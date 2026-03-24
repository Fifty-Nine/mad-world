"""Tests for the utility module."""

from mad_world.util import wrap_text


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
