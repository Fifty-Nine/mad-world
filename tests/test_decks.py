"""Tests for card decks."""

from __future__ import annotations

import copy
import random

import pytest

from mad_world.decks import Deck, DeckEmptyError


@pytest.fixture
def stable_deck(
    stable_rng: random.Random,
) -> Deck[int]:
    return Deck.create([1, 2, 3, 4, 5], stable_rng)


def test_deck_draw(stable_deck: Deck[int], stable_rng: random.Random) -> None:
    assert len(stable_deck) == 5
    stable_deck.shuffle_draw(stable_rng, with_discard=True)

    assert [stable_deck.draw(stable_rng) for i in range(5)] == [1, 2, 3, 4, 5]
    assert stable_deck.available_to_draw() == 0
    assert stable_deck.cards_in_play() == 5


def test_draw_from_discard(
    stable_deck: Deck[int],
    stable_rng: random.Random,
) -> None:
    stable_deck.draw_pile = []
    stable_deck.discard_pile = [2, 4, 6, 8]

    assert [stable_deck.draw(stable_rng) for i in range(4)] == [2, 4, 6, 8]


def test_deck_discard(
    stable_deck: Deck[int], stable_rng: random.Random
) -> None:
    assert stable_deck.draw(stable_rng) == 1
    assert 1 in stable_deck.in_play
    assert 1 not in stable_deck.draw_pile
    assert 1 not in stable_deck.discard_pile
    assert 1 not in stable_deck.trash_pile

    with pytest.raises(ValueError, match="x not in list"):
        stable_deck.discard(7)

    stable_deck.discard(1)
    assert 1 not in stable_deck.in_play
    assert 1 not in stable_deck.draw_pile
    assert 1 in stable_deck.discard_pile
    assert 1 not in stable_deck.trash_pile

    stable_deck.shuffle_draw(stable_rng, with_discard=False)
    assert 1 not in stable_deck.in_play
    assert 1 not in stable_deck.draw_pile
    assert 1 in stable_deck.discard_pile
    assert 1 not in stable_deck.trash_pile

    stable_deck.shuffle_draw(stable_rng, with_discard=True)
    assert 1 not in stable_deck.in_play
    assert 1 in stable_deck.draw_pile
    assert 1 not in stable_deck.discard_pile
    assert 1 not in stable_deck.trash_pile

    assert stable_deck.draw(stable_rng) == 1


def test_draw_empty_deck(
    stable_deck: Deck[int], stable_rng: random.Random
) -> None:
    for _ in range(len(stable_deck)):
        stable_deck.draw(stable_rng)

    with pytest.raises(DeckEmptyError):
        stable_deck.draw(stable_rng)


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


def test_deck_trash(stable_deck: Deck[int], stable_rng: random.Random) -> None:
    assert stable_deck.draw(stable_rng) == 1
    assert 1 in stable_deck.in_play
    assert 1 not in stable_deck.draw_pile
    assert 1 not in stable_deck.discard_pile
    assert 1 not in stable_deck.trash_pile

    with pytest.raises(ValueError, match="x not in list"):
        stable_deck.trash(7)

    stable_deck.trash(1)
    assert 1 not in stable_deck.in_play
    assert 1 not in stable_deck.draw_pile
    assert 1 not in stable_deck.discard_pile
    assert 1 in stable_deck.trash_pile

    stable_deck.shuffle_draw(stable_rng, with_discard=False)
    assert 1 not in stable_deck.in_play
    assert 1 not in stable_deck.draw_pile
    assert 1 not in stable_deck.discard_pile
    assert 1 in stable_deck.trash_pile

    stable_deck.shuffle_draw(stable_rng, with_discard=True)
    assert 1 not in stable_deck.in_play
    assert 1 not in stable_deck.draw_pile
    assert 1 not in stable_deck.discard_pile
    assert 1 in stable_deck.trash_pile

    assert stable_deck.draw(stable_rng) == 2
