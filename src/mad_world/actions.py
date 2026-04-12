from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mad_world.enums import OpenChannelPreference

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
            f"you only have {available}."
        )


class InvalidGDPAmountError(InvalidActionError):
    def __init__(self) -> None:
        super().__init__(
            "INVALID GDP AMOUNT: GDP investment amount cannot be negative."
        )


class InvalidInfluenceAmountError(InvalidActionError):
    def __init__(self) -> None:
        super().__init__(
            "INVALID INFLUENCE AMOUNT: Influence investment amount cannot be "
            "negative."
        )


class InvalidOperationError(InvalidActionError):
    def __init__(self, *, operation: str, allowed: list[str]) -> None:
        super().__init__(
            f"INVALID OPERATION: '{operation}' is not a valid "
            "operation. Allowed operations are: {allowed}"
        )


class InvalidChannelRequestError(InvalidActionError):
    def __init__(self, *, limit: int) -> None:
        super().__init__(
            f"INVALID CHANNEL PREFERENCE: You have already successfully "
            f"requested {limit} channels this game and cannot "
            "request any more."
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
    channel_preference: OpenChannelPreference = Field(
        default=OpenChannelPreference.ACCEPT,
        description="Your preference for opening a direct back-and-forth "
        "communication channel with your opponent this phase. "
        "REQUEST will attempt to open a channel. ACCEPT will open a channel "
        "if your opponent requested one. REJECT will never open a channel.",
    )

    def validate_semantics(self, game: GameState, player_name: str) -> None:
        limit = game.rules.max_channels_per_game
        if (
            self.channel_preference == OpenChannelPreference.REQUEST
            and game.players[player_name].channels_opened >= limit
        ):
            raise InvalidChannelRequestError(limit=limit)

    def requests_channel(self) -> bool:
        return self.channel_preference == OpenChannelPreference.REQUEST

    def accepts_channel(self) -> bool:
        return self.channel_preference in (
            OpenChannelPreference.REQUEST,
            OpenChannelPreference.ACCEPT,
        )


class ChatAction(BaseAction):
    message: str = Field(
        max_length=256,
        description="Your message to the opponent.",
    )
    end_channel: bool = Field(
        default=False,
        description="If true, the channel will be closed immediately after "
        "this message is sent.",
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
        if self.bid not in game.allowed_bids:
            raise InvalidBiddingActionError(self.bid, game.allowed_bids)


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
            game.allowed_operations[op].influence_cost for op in self.operations
        )
        player_state = game.players[player_name]
        if total_cost > player_state.influence:
            raise InsufficientInfluenceError(
                cost=total_cost, available=player_state.influence
            )
