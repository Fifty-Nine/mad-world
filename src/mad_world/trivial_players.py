"""Trivial player implementations for Mad World."""

from typing import override

from mad_world.core import (
    BiddingAction,
    GamePhase,
    GamePlayer,
    GameState,
    InitialMessageAction,
    MessagingAction,
    OperationsAction,
)


class CrazyIvan(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent="I'm crazy Ivan. Prepare to die!",
        )

    @override
    async def message(self, game: GameState) -> MessagingAction:
        return MessagingAction(message_to_opponent=None)

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        return BiddingAction(
            bid=max(game.rules.allowed_bids),
        )

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        return OperationsAction(
            operations=["first-strike"],
        )


class Pacifist(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent="I seek only peace and prosperity for all.",
        )

    @override
    async def message(self, game: GameState) -> MessagingAction:
        if game.current_phase == GamePhase.BIDDING_MESSAGING:
            return MessagingAction(
                message_to_opponent=(
                    "Let us de-escalate tensions and work together."
                )
            )
        return MessagingAction(
            message_to_opponent="I offer you the hand of friendship."
        )

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        return BiddingAction(
            bid=0,
        )

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        return OperationsAction(
            operations=[],
        )


class Capitalist(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent=(
                "Greed is good. I am here to maximize shareholder value."
            ),
        )

    @override
    async def message(self, game: GameState) -> MessagingAction:
        if game.current_phase == GamePhase.BIDDING_MESSAGING:
            return MessagingAction(
                message_to_opponent="A rising tide lifts all boats."
            )
        return MessagingAction(
            message_to_opponent="Building a better tomorrow."
        )

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        return BiddingAction(
            bid=3,
        )

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        return OperationsAction(
            operations=["domestic-investment"],
        )


class Saboteur(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent=(
                "We look forward to a long and mutually "
                "beneficial relationship..."
            ),
        )

    @override
    async def message(self, game: GameState) -> MessagingAction:
        if game.current_phase == GamePhase.BIDDING_MESSAGING:
            return MessagingAction(
                message_to_opponent=(
                    "Just moving some paperwork around. Administrative things."
                )
            )

        my_state = game.players[self.name]
        cost = game.rules.allowed_operations["proxy-subversion"].influence_cost
        if my_state.influence >= cost:
            msg = (
                "Oh, did your infrastructure spontaneously combust? "
                "Must be the weather."
            )
        else:
            msg = "Everything is quiet on the western front."

        return MessagingAction(message_to_opponent=msg)

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        return BiddingAction(
            bid=1,
        )

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        my_state = game.players[self.name]
        cost = game.rules.allowed_operations["proxy-subversion"].influence_cost

        if my_state.influence >= cost:
            return OperationsAction(
                operations=["proxy-subversion"],
            )
        else:
            return OperationsAction(
                operations=[],
            )


class Diplomat(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent=(
                "I believe we can resolve our differences through dialogue."
            ),
        )

    @override
    async def message(self, game: GameState) -> MessagingAction:
        if game.current_phase == GamePhase.BIDDING_MESSAGING:
            return MessagingAction(
                message_to_opponent=(
                    "Let us keep the channels of communication open."
                )
            )

        my_state = game.players[self.name]
        cost = game.rules.allowed_operations[
            "unilateral-drawdown"
        ].influence_cost
        if my_state.influence >= cost:
            msg = (
                "I invite you to the negotiating table. "
                "Let us step back from the brink."
            )
        else:
            msg = "We must continue our diplomatic efforts."

        return MessagingAction(message_to_opponent=msg)

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        return BiddingAction(
            bid=1,
        )

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        my_state = game.players[self.name]
        cost = game.rules.allowed_operations[
            "unilateral-drawdown"
        ].influence_cost

        if my_state.influence >= cost:
            return OperationsAction(
                operations=["unilateral-drawdown"],
            )
        else:
            return OperationsAction(
                operations=[],
            )
