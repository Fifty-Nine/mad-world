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
from mad_world.enums import BlameGamePosture, GamePhase
from mad_world.players import GamePlayer
from mad_world.util import (
    escalation_budget,
    get_subclass_by_name,
    pareto_optimal_bid,
)

if TYPE_CHECKING:
    from mad_world.core import GameState
    from mad_world.crises import BaseCrisis, GenericCrisis
    from mad_world.rules import OperationDefinition


class TrivialPlayer(GamePlayer):
    def __init__(self, name: str, *, aggressive: bool) -> None:
        super().__init__(name)
        self.aggressive = aggressive

    @override
    async def crisis[T: BaseAction](
        self,
        game: GameState,
        crisis: GenericCrisis[T],
    ) -> T:
        return crisis.get_default_action(
            self.name, game, aggressive=self.aggressive
        )


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
                ),
            )
        return MessagingAction(
            message_to_opponent="I offer you the hand of friendship.",
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
                message_to_opponent="A rising tide lifts all boats.",
            )
        return MessagingAction(
            message_to_opponent="Building a better tomorrow.",
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
                ),
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
                ),
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

        return OperationsAction(
            operations=[],
        )


class ParetoEfficientPlayer(TrivialPlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name, aggressive=True)

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        return InitialMessageAction(
            message_to_opponent=(
                "[STATUS] Optimal Game Algorithm booting...\n"
                "Greetings {OPPONENT NAME HERE}. I am programmed to engage in "
                "STRICTLY OPTIMAL PLAY. You are mathematically guaranteed to "
                "lose or draw.\nEND OF MESSAGE."
            )
        )

    @override
    async def message(self, game: GameState) -> MessagingAction:
        return MessagingAction(
            message_to_opponent=(
                "[STATUS] Updating calculations for current game state."
            )
        )

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        return BiddingAction(
            bid=pareto_optimal_bid(
                game.doomsday_clock,
                game.rules.max_clock_state,
                game.rules.allowed_bids,
            )
        )

    def _new_budget(
        self, ebudget: int, ibudget: int, op: OperationDefinition
    ) -> tuple[int, int]:
        return (ebudget - op.clock_effect, ibudget - op.influence_cost)

    def _best_operation(
        self, ebudget: int, ibudget: int, ops: list[OperationDefinition]
    ) -> OperationDefinition | None:
        for op in ops:
            new_ebudget, new_ibudget = self._new_budget(ebudget, ibudget, op)

            if new_ebudget < 0 or new_ibudget < 0:
                continue

            return op

        return None

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        ebudget = escalation_budget(
            game.doomsday_clock, game.rules.max_clock_state
        )
        ibudget = game.players[self.name].influence

        ops_to_check = [
            game.rules.allowed_operations["proxy-subversion"],
            game.rules.allowed_operations["aggressive-extraction"],
            game.rules.allowed_operations["domestic-investment"],
        ]

        result: list[str] = []
        while ebudget > 0 and ibudget > 0:
            next_op = self._best_operation(ebudget, ibudget, ops_to_check)
            if next_op is None:
                break

            result.append(next_op.name)
            ebudget, ibudget = self._new_budget(ebudget, ibudget, next_op)

        return OperationsAction(operations=result)

    @override
    async def crisis_message(
        self, game: GameState, crisis: BaseCrisis
    ) -> MessagingAction:
        if crisis.card_kind == "standoff":
            return MessagingAction(
                message_to_opponent=(
                    "[STATUS] Preparing to STAND FIRM.\n"
                    "Greetings {OPPONENT NAME HERE}. I am programmed to always "
                    "STAND FIRM in the event of a STANDOFF CRISIS. As a "
                    "result, your only choice is to BACK DOWN or we will "
                    "destroy the world and both lose the game. I am doing this "
                    "because it is the OPTIMAL STRATEGY and cannot choose to "
                    "take any other action."
                )
            )
        if crisis.card_kind == "blame-game":
            opponent_name = next(p for p in game.players if p != self.name)
            my_debt = game.escalation_debt(self.name)
            their_debt = game.escalation_debt(opponent_name)
            if my_debt <= their_debt:
                return MessagingAction(
                    message_to_opponent=(
                        "[STATUS] Preparing to DEFLECT blame.\n"
                        "My calculations indicate I hold the upper hand in "
                        "escalation debt. I will DEFLECT. If you also DEFLECT, "
                        "we will both be destroyed. If you SHOULDER, I win "
                        "Influence. This is the optimal trap."
                    )
                )
            return MessagingAction(
                message_to_opponent=(
                    "[STATUS] Preparing to SHOULDER blame.\n"
                    "My calculations indicate I am at a disadvantage in "
                    "escalation debt. I will SHOULDER to avoid the massive "
                    "GDP penalty trap. The math is undeniable."
                )
            )

        raise NotImplementedError(
            "Need logic for pareto-optimal crisis handling."
        )  # pragma: no cover

    @override
    async def crisis[T: BaseAction](
        self,
        game: GameState,
        crisis: GenericCrisis[T],
    ) -> T:
        if crisis.card_kind == "blame-game":
            opponent_name = next(p for p in game.players if p != self.name)
            my_debt = game.escalation_debt(self.name)
            their_debt = game.escalation_debt(opponent_name)

            if my_debt <= their_debt:
                posture = BlameGamePosture.DEFLECT
            else:
                posture = BlameGamePosture.SHOULDER

            return crisis.action_type.model_validate({"posture": posture})

        return await super().crisis(game, crisis)
