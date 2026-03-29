"""Trivial player implementations for Mad World."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from mad_world.actions import (
    BaseAction,
    BiddingAction,
    InitialMessageAction,
    MessagingAction,
    OperationsAction,
)
from mad_world.enums import GamePhase
from mad_world.players import GamePlayer
from mad_world.util import get_subclass_by_name

if TYPE_CHECKING:
    from mad_world.core import GameState
    from mad_world.crises import GenericCrisis


class TrivialPlayer(GamePlayer):
    def __init__(self, name: str, aggressive: bool) -> None:
        super().__init__(name)
        self.aggressive = aggressive

    @override
    async def crisis[T: BaseAction](
        self, game: GameState, crisis: GenericCrisis[T]
    ) -> T:
        return crisis.get_default_action(self.aggressive)


def get_trivial_player(kind: str, name: str) -> TrivialPlayer | None:
    """Finds a trivial player class by name."""
    return get_subclass_by_name(__name__, kind, TrivialPlayer, name)


class CrazyIvan(TrivialPlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name, aggressive=True)

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


class Pacifist(TrivialPlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name, aggressive=False)

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


class Capitalist(TrivialPlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name, aggressive=True)

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


class Saboteur(TrivialPlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name, aggressive=True)

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


class Diplomat(TrivialPlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name, aggressive=False)

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
