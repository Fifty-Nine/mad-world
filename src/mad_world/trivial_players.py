"""Trivial player implementations for Mad World."""

from typing import override

from mad_world.core import (
    BiddingAction,
    GamePlayer,
    GameState,
    OperationsAction,
)


class CrazyIvan(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> str | None:
        return "I'm crazy Ivan. Prepare to die!"

    @override
    def bid(
        self, game: GameState, message_from_opponent: str | None
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent=None,
            bid=max(game.rules.allowed_bids),
            internal_monologue="No thoughts, head empty.",
        )

    @override
    def operations(
        self, game: GameState, message_to_opponent: str | None
    ) -> OperationsAction:
        return OperationsAction(
            message_to_opponent=None,
            operations=["first-strike"],
            internal_monologue="I'm crazy!",
        )
