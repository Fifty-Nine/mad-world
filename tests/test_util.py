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
    # "Long line that" -> 14 chars
    # "  Long line that" -> 16 chars? No, textwrap.wrap counts indent in width.
    # width=15, indent="  " -> 13 chars of content + 2 chars of indent.
    lines = wrapped.splitlines()
    for line in lines:
        assert line.startswith(indent)
        assert len(line) <= 15


def test_wrap_text_empty_lines() -> None:
    text = "A\n\nB"
    wrapped = wrap_text(text, indent="  ")
    # My current implementation: "  A\n  \n  B"
    # Wait, textwrap.wrap for empty string returns empty list.
    # My loop: if not line.strip(): wrapped_lines.append(indent)
    assert wrapped == "  A\n  \n  B"
