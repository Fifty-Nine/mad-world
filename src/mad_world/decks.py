"""Generic logic for working with card decks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, SerializeAsAny

if TYPE_CHECKING:
    import random
    from collections.abc import Iterable
    from typing import Self


class DeckEmptyError(RuntimeError):
    """Exception raised when the caller attempts to draw from a deck where
    all cards are currently in play."""

    def __init__(self) -> None:
        super().__init__(
            "Attempted to draw from a deck where all cards are already in play."
        )


type Pile[T] = list[SerializeAsAny[T]]


class Deck[T](BaseModel):
    draw_pile: Pile[T] = Field(
        description="The deck from which cards are drawn."
    )
    discard_pile: Pile[T] = Field(
        default_factory=list,
        description="The deck to which discarded cards are added.",
    )
    in_play: Pile[T] = Field(
        default_factory=list,
        description="The list of cards currently missing from the deck because "
        "they are in play.",
    )
    trash_pile: Pile[T] = Field(
        default_factory=list,
        description="The deck to which trashed cards are added.",
    )

    @classmethod
    def create(cls, initial_cards: Iterable[T], rng: random.Random) -> Self:
        result = cls(draw_pile=list(initial_cards))
        result.shuffle_draw(rng, with_discard=True)
        return result

    def shuffle_draw(self, rng: random.Random, *, with_discard: bool) -> None:
        if with_discard:
            self.draw_pile.extend(self.discard_pile)
            self.discard_pile.clear()

        rng.shuffle(self.draw_pile)

    def draw(self, rng: random.Random) -> T:
        if len(self.draw_pile) == 0:
            self.shuffle_draw(rng, with_discard=True)

        if len(self.draw_pile) == 0:
            raise DeckEmptyError

        result = self.draw_pile.pop()
        self.in_play.append(result)
        return result

    def discard(self, card: T) -> None:
        self.in_play.remove(card)
        self.discard_pile.append(card)

    def trash(self, card: T) -> None:
        self.in_play.remove(card)
        self.trash_pile.append(card)

    def available_to_draw(self) -> int:
        return len(self.draw_pile) + len(self.discard_pile)

    def cards_in_play(self) -> int:
        return len(self.in_play)

    def __len__(self) -> int:
        return self.available_to_draw() + self.cards_in_play()
