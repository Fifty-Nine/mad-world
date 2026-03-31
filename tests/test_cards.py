"""Tests for Card serialization infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import pytest
from pydantic import Field, ValidationError

from mad_world.cards import BaseCard, CardNameCollisionError
from mad_world.decks import Deck

if TYPE_CHECKING:
    import random


class BaseTestFooCard(BaseCard):
    pass


class BaseTestBarCard(BaseCard):
    pass


class IntermediateBaseBarCard(BaseTestBarCard):
    pass


class ConcreteFoo1Card(BaseTestFooCard):
    card_kind: ClassVar[Literal["foo1"]] = "foo1"


class ConcreteFoo2Card(BaseTestFooCard):
    card_kind: ClassVar[Literal["foo2"]] = "foo2"


class ConcreteBar1Card(BaseTestBarCard):
    card_kind: ClassVar[Literal["bar1"]] = "bar1"


class ConcreteBar2Card(IntermediateBaseBarCard):
    card_kind: ClassVar[Literal["bar2"]] = "bar2"


def test_card_serialize() -> None:
    assert ConcreteFoo1Card().model_dump() == {"card_kind": "foo1"}
    assert ConcreteFoo2Card().model_dump() == {"card_kind": "foo2"}
    assert ConcreteBar1Card().model_dump() == {"card_kind": "bar1"}
    assert ConcreteBar2Card().model_dump() == {"card_kind": "bar2"}


def test_card_deserialize() -> None:
    assert (
        BaseTestFooCard.model_validate({"card_kind": "foo1"})
        == ConcreteFoo1Card()
    )
    assert (
        BaseTestFooCard.model_validate({"card_kind": "foo2"})
        == ConcreteFoo2Card()
    )
    assert (
        BaseTestBarCard.model_validate({"card_kind": "bar1"})
        == ConcreteBar1Card()
    )
    assert (
        BaseTestBarCard.model_validate({"card_kind": "bar2"})
        == ConcreteBar2Card()
    )


def test_heterogenous_card_list(stable_rng: random.Random) -> None:
    foo_cards = Deck[BaseTestFooCard].create(
        [ConcreteFoo1Card(), ConcreteFoo1Card(), ConcreteFoo2Card()],
        rng=stable_rng,
    )
    bar_cards = Deck[BaseTestBarCard].create(
        [ConcreteBar2Card(), ConcreteBar1Card(), ConcreteBar2Card()],
        rng=stable_rng,
    )

    all_cards: list[BaseCard] = []
    all_cards.extend(foo_cards.draw_pile.copy())
    all_cards.extend(bar_cards.draw_pile.copy())

    foobar_cards = Deck[BaseCard].create(all_cards, rng=stable_rng)

    assert foo_cards == Deck[BaseTestFooCard].model_validate(
        foo_cards.model_dump()
    )
    assert bar_cards == Deck[BaseTestBarCard].model_validate(
        bar_cards.model_dump()
    )

    assert foobar_cards == Deck[BaseCard].model_validate(
        foobar_cards.model_dump()
    )


def test_card_compare() -> None:
    assert ConcreteFoo1Card() < ConcreteFoo2Card()

    with pytest.raises(TypeError, match="not supported between instances"):
        assert ConcreteFoo1Card() < 0


def test_registry_collision() -> None:
    with pytest.raises(
        CardNameCollisionError, match='Card with kind "foo1" already exists'
    ):

        class NewFooCard(BaseTestFooCard):
            card_kind: ClassVar[Literal["foo1"]] = "foo1"


def test_bad_card_kind() -> None:
    with pytest.raises(ValidationError):
        BaseCard.model_validate({"card_kind": 1})


def test_bad_card_serialization_type() -> None:
    with pytest.raises(ValidationError):
        BaseCard.model_validate(1)


def test_bad_card_wrong_base() -> None:
    with pytest.raises(ValidationError):
        BaseTestFooCard.model_validate({"card_kind": "bar1"})


def test_bad_card_deserialize_wrong_concrete() -> None:
    with pytest.raises(ValidationError):
        ConcreteFoo1Card.model_validate({"card_kind": "foo2"})


def test_card_instance_state() -> None:
    class StatefulCard(BaseCard):
        card_kind: ClassVar[Literal["stateful"]] = "stateful"
        field: str = Field()

    card = StatefulCard(field="foo")

    assert card == StatefulCard.model_validate(card.model_dump())
    assert card == BaseCard.model_validate(card.model_dump())
