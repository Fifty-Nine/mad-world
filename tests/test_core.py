"""Tests for the core module."""

from mad_world.core import get_greeting


def test_get_greeting() -> None:
    """Test the greeting function."""
    assert get_greeting() == "Welcome to Mad World!"
