"""Tests for persona generation."""

from __future__ import annotations

from mad_world.personas import is_trivial_persona, random_persona


def test_random_persona() -> None:
    """Test generating a random persona."""
    persona = random_persona()
    assert is_trivial_persona(persona)


def test_is_trivial_persona() -> None:
    """Test checking if a persona is trivial."""
    assert is_trivial_persona("Erratic Appeaser")
    assert is_trivial_persona("Zealous Zealot")
    assert not is_trivial_persona("Erratic")
    assert not is_trivial_persona("Appeaser")
    assert not is_trivial_persona("Erratic Appeaser with a twist")
    assert not is_trivial_persona("erratic appeaser")
    assert not is_trivial_persona("Erratic appeaser")
    assert not is_trivial_persona("erratic Appeaser")
    assert is_trivial_persona("Erratic Appeaser\n")
