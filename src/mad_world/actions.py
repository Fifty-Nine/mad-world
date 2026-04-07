from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mad_world.core import GameState


class InvalidActionError(Exception):
    """Raised during Action validation when an action
    is not allowed under the current game state or
    rules.
    """


class InvalidBiddingActionError(InvalidActionError):
    def __init__(self, bid: int, allowed_bids: list[int]) -> None:
        super().__init__(
            f"INVALID BID: Your bid of {bid} is not allowed. "
            f"Allowed bids are {allowed_bids}."
        )


class InsufficientInfluenceError(InvalidActionError):
    def __init__(
        self, *, available: int, cost: int, operation: str | None = None
    ) -> None:
        text = (
            (
                f"'{operation}' costs {cost} Inf, but you only "
                f"have {available} Inf."
            )
            if operation is not None
            else (
                f"The submitted operations require a total of {cost} Inf, "
                f"but you only have {available} Inf."
            )
        )
        super().__init__(f"INSUFFICIENT INFLUENCE: {text}")


class InsufficientGDPError(InvalidActionError):
    def __init__(self, *, available: int, cost: int) -> None:
        super().__init__(
            f"INSUFFICIENT GDP: The submitted bid requires {cost} GDP, but "
            "you only have {available}."
        )


class InvalidGDPAmountError(InvalidActionError):
    def __init__(self) -> None:
        super().__init__(
            "INVALID GDP AMOUNT: GDP investment amount cannot be negative."
        )


class InvalidOperationError(InvalidActionError):
    def __init__(self, *, operation: str, allowed: list[str]) -> None:
        super().__init__(
            f"INVALID OPERATION: '{operation}' is not a valid "
            "operation. Allowed operations are: {allowed}"
        )


class BaseAction(BaseModel):
    def validate_semantics(self, game: GameState, player_name: str) -> None:
        pass


class MessagingAction(BaseAction):
    message_to_opponent: str | None = Field(
        default=None,
        description="A message that will be passed to your opponent. You can "
        "use this to conduct diplomacy, respond to inquiries, "
        "issue threats, etc.",
    )


class InitialMessageAction(MessagingAction):
    pass


class BiddingAction(BaseAction):
    """Indicates a player's actions during the bidding phase."""

    bid: int = Field(
        description="Your influence bid for this phase. This value will be "
        "added directly to your current influence. This bid also increases "
        "the doomsday clock by the same amount. The bid must be one of the "
        "values allowed by the rules (see the 'allowed_bids' field in the "
        "rules) or you will automatically bid the maximum possible amount. "
        "A bid of 0 is de-escalatory and reduces the doomsday clock by 1.",
    )

    def validate_semantics(self, game: GameState, player_name: str) -> None:
        if self.bid not in game.rules.allowed_bids:
            raise InvalidBiddingActionError(self.bid, game.rules.allowed_bids)


class OperationsAction(BaseAction):
    operations: list[str] = Field(
        description="The set of operations to conduct this turn. Each string "
        "must be a valid operation allowed by the rules. You must "
        "have sufficient influence to conduct the operation.",
    )

    def validate_semantics(self, game: GameState, player_name: str) -> None:
        for op_name in self.operations:
            game.validate_operation(op_name, player_name)

        total_cost = sum(
            game.rules.allowed_operations[op].influence_cost
            for op in self.operations
        )
        player_state = game.players[player_name]
        if total_cost > player_state.influence:
            raise InsufficientInfluenceError(
                cost=total_cost, available=player_state.influence
            )
