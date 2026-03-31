"""Generic logic for working with card decks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import random
    from typing import Self


class DeckEmptyError(RuntimeError):
    """Exception raised when the caller attempts to draw from a deck where
    all cards are currently in play."""

    def __init__(self) -> None:
        super().__init__(
            "Attempted to draw from a deck where all cards are already in play."
        )


class Deck[T](BaseModel):
    draw_pile: list[T] = Field(
        description="The deck from which cards are drawn."
    )
    discard_pile: list[T] = Field(
        default_factory=list,
        description="The deck to which discarded cards are added.",
    )
    in_play: list[T] = Field(
        default_factory=list,
        description="The list of cards currently missing from the deck because "
        "they are in play.",
    )

    @classmethod
    def create(cls, initial_cards: list[T], rng: random.Random) -> Self:
        result = cls(draw_pile=initial_cards.copy())
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

    def available_to_draw(self) -> int:
        return len(self.draw_pile) + len(self.discard_pile)

    def cards_in_play(self) -> int:
        return len(self.in_play)

    def __len__(self) -> int:
        return self.available_to_draw() + self.cards_in_play()
