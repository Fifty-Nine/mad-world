"""Tests for card decks."""

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from mad_world.decks import Deck

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def stable_deck(
    seeded_rng: random.Random,
) -> Generator[tuple[Deck[int], random.Random], None, None]:
    def fixed_shuffle(values: list[int]) -> None:
        values.sort(reverse=True)

    result = Deck.create([1, 2, 3, 4, 5], seeded_rng)

    with patch.object(seeded_rng, "shuffle", side_effect=fixed_shuffle):
        result.shuffle_draw(seeded_rng, with_discard=True)
        yield result, seeded_rng


def test_deck_draw(stable_deck: tuple[Deck[int], random.Random]) -> None:
    deck, rng = stable_deck
    assert len(deck) == 5
    deck.shuffle_draw(rng, with_discard=True)

    assert [deck.draw(rng) for i in range(5)] == [1, 2, 3, 4, 5]
    assert deck.available_to_draw() == 0
    assert deck.cards_in_play() == 5


def test_draw_from_discard(
    stable_deck: tuple[Deck[int], random.Random],
) -> None:
    deck, rng = stable_deck
    deck.draw_pile = []
    deck.discard_pile = [2, 4, 6, 8]

    assert [deck.draw(rng) for i in range(4)] == [2, 4, 6, 8]


def test_deck_discard(stable_deck: tuple[Deck[int], random.Random]) -> None:
    deck, rng = stable_deck
    assert deck.draw(rng) == 1
    assert 1 in deck.in_play
    assert 1 not in deck.draw_pile
    assert 1 not in deck.discard_pile

    with pytest.raises(ValueError, match="x not in list"):
        deck.discard(7)

    deck.discard(1)
    assert 1 not in deck.in_play
    assert 1 not in deck.draw_pile
    assert 1 in deck.discard_pile

    deck.shuffle_draw(rng, with_discard=False)
    assert 1 not in deck.in_play
    assert 1 not in deck.draw_pile
    assert 1 in deck.discard_pile

    deck.shuffle_draw(rng, with_discard=True)
    assert 1 not in deck.in_play
    assert 1 in deck.draw_pile
    assert 1 not in deck.discard_pile

    assert deck.draw(rng) == 1


def test_draw_empty_deck(stable_deck: tuple[Deck[int], random.Random]) -> None:
    deck, rng = stable_deck
    for _ in range(len(deck)):
        deck.draw(rng)

    with pytest.raises(IndexError, match="pop from empty list"):
        deck.draw(rng)


def test_deck_round_trip() -> None:
    rng = random.Random(123)
    deck = Deck[str].create(
        initial_cards=[f"card_{n}" for n in range(1000)], rng=rng
    )
    deck.discard_pile = deck.draw_pile
    deck.draw_pile = []

    rng2 = copy.deepcopy(rng)

    loaded = Deck[str].model_validate(deck.model_dump())
    assert loaded.draw(rng) == deck.draw(rng2)
    assert id(loaded) != id(deck)
