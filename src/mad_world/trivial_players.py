"""Trivial player implementations for Mad World."""

from typing import override

from mad_world.core import (
    BiddingAction,
    GamePlayer,
    GameState,
    InitialMessageAction,
    OperationsAction,
)


class CrazyIvan(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent="I'm crazy Ivan. Prepare to die!",
        )

    @override
    def bid(
        self, game: GameState, message_from_opponent: str | None
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent=None,
            bid=max(game.rules.allowed_bids),
        )

    @override
    def operations(
        self, game: GameState, message_from_opponent: str | None
    ) -> OperationsAction:
        return OperationsAction(
            message_to_opponent=None,
            operations=["first-strike"],
        )


class Pacifist(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent="I seek only peace and prosperity for all.",
        )

    @override
    def bid(
        self, game: GameState, message_from_opponent: str | None
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent=(
                "Let us de-escalate tensions and work together."
            ),
            bid=0,
        )

    @override
    def operations(
        self, game: GameState, message_from_opponent: str | None
    ) -> OperationsAction:
        return OperationsAction(
            message_to_opponent="I offer you the hand of friendship.",
            operations=[],
        )


class Capitalist(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent=(
                "Greed is good. I am here to maximize shareholder value."
            ),
        )

    @override
    def bid(
        self, game: GameState, message_from_opponent: str | None
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent="A rising tide lifts all boats.",
            bid=3,
        )

    @override
    def operations(
        self, game: GameState, message_from_opponent: str | None
    ) -> OperationsAction:
        return OperationsAction(
            message_to_opponent="Building a better tomorrow.",
            operations=["domestic-investment"],
        )


class Saboteur(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent=(
                "We look forward to a long and mutually "
                "beneficial relationship..."
            ),
        )

    @override
    def bid(
        self, game: GameState, message_from_opponent: str | None
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent=(
                "Just moving some paperwork around. Administrative things."
            ),
            bid=1,
        )

    @override
    def operations(
        self, game: GameState, message_from_opponent: str | None
    ) -> OperationsAction:
        my_state = game.players[self.name]
        cost = game.rules.allowed_operations["proxy-subversion"].influence_cost

        if my_state.influence >= cost:
            return OperationsAction(
                message_to_opponent=(
                    "Oh, did your infrastructure spontaneously combust? "
                    "Must be the weather."
                ),
                operations=["proxy-subversion"],
            )
        else:
            return OperationsAction(
                message_to_opponent="Everything is quiet on the western front.",
                operations=[],
            )


class Diplomat(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent=(
                "I believe we can resolve our differences through dialogue."
            ),
        )

    @override
    def bid(
        self, game: GameState, message_from_opponent: str | None
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent=(
                "Let us keep the channels of communication open."
            ),
            bid=1,
        )

    @override
    def operations(
        self, game: GameState, message_from_opponent: str | None
    ) -> OperationsAction:
        my_state = game.players[self.name]
        cost = game.rules.allowed_operations[
            "unilateral-drawdown"
        ].influence_cost

        if my_state.influence >= cost:
            return OperationsAction(
                message_to_opponent=(
                    "I invite you to the negotiating table. "
                    "Let us step back from the brink."
                ),
                operations=["unilateral-drawdown"],
            )
        else:
            return OperationsAction(
                message_to_opponent="We must continue our diplomatic efforts.",
                operations=[],
            )
