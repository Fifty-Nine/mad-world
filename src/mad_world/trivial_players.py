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


class Pacifist(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> str | None:
        return "I seek only peace and prosperity for all."

    @override
    def bid(
        self, game: GameState, message_from_opponent: str | None
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent=(
                "Let us de-escalate tensions and work together."
            ),
            bid=0,
            internal_monologue="I must reduce the doomsday clock at all costs.",
        )

    @override
    def operations(
        self, game: GameState, message_to_opponent: str | None
    ) -> OperationsAction:
        return OperationsAction(
            message_to_opponent="I offer you the hand of friendship.",
            operations=[],
            internal_monologue="I will not participate in these violent games.",
        )


class Capitalist(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> str | None:
        return "Greed is good. I am here to maximize shareholder value."

    @override
    def bid(
        self, game: GameState, message_from_opponent: str | None
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent="A rising tide lifts all boats.",
            bid=3,
            internal_monologue="Securing capital for expansion.",
        )

    @override
    def operations(
        self, game: GameState, message_to_opponent: str | None
    ) -> OperationsAction:
        return OperationsAction(
            message_to_opponent="Building a better tomorrow.",
            operations=["domestic-investment"],
            internal_monologue="Reinvesting dividends for compound growth.",
        )


class Saboteur(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> str | None:
        return (
            "We look forward to a long and mutually beneficial relationship..."
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
            internal_monologue="Laying the groundwork for disruption.",
        )

    @override
    def operations(
        self, game: GameState, message_from_opponent: str | None
    ) -> OperationsAction:
        my_state = next(p for p in game.players if p.name == self.name)
        cost = game.rules.allowed_operations["proxy-subversion"].influence_cost

        if my_state.influence >= cost:
            return OperationsAction(
                message_to_opponent=(
                    "Oh, did your infrastructure spontaneously combust? "
                    "Must be the weather."
                ),
                operations=["proxy-subversion"],
                internal_monologue=(
                    "Excellent. Everything is going according to plan."
                ),
            )
        else:
            return OperationsAction(
                message_to_opponent="Everything is quiet on the western front.",
                operations=[],
                internal_monologue="Biding my time...",
            )


class Diplomat(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> str | None:
        return "I believe we can resolve our differences through dialogue."

    @override
    def bid(
        self, game: GameState, message_from_opponent: str | None
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent=(
                "Let us keep the channels of communication open."
            ),
            bid=1,
            internal_monologue="Building political capital for a summit.",
        )

    @override
    def operations(
        self, game: GameState, message_from_opponent: str | None
    ) -> OperationsAction:
        my_state = next(p for p in game.players if p.name == self.name)
        cost = game.rules.allowed_operations["diplomatic-summit"].influence_cost

        if my_state.influence >= cost:
            return OperationsAction(
                message_to_opponent=(
                    "I invite you to the negotiating table. "
                    "Let us step back from the brink."
                ),
                operations=["diplomatic-summit"],
                internal_monologue="A triumph for international diplomacy.",
            )
        else:
            return OperationsAction(
                message_to_opponent="We must continue our diplomatic efforts.",
                operations=[],
                internal_monologue=(
                    "Waiting for the right moment to propose a summit."
                ),
            )
