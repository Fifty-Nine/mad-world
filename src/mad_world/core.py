"""Core mechanics for the game."""

import copy
import logging as logging
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mad_world.rules import (
    DEFAULT_RULES,
    GameRules,
)
from mad_world.util import wrap_text


class InvalidActionError(Exception):
    """Raised during Action validation when an action
    is not allowed under the current game state or
    rules.
    """

    pass


class GamePhase(Enum):
    OPENING = 1
    BIDDING_MESSAGING = 2
    BIDDING = 3
    OPERATIONS_MESSAGING = 4
    OPERATIONS = 5
    END = 6


class GameOverReason(Enum):
    WORLD_DESTROYED = 1
    ECONOMIC_VICTORY = 2
    STALEMATE = 3


RANDOM = random.Random()


class PlayerState(BaseModel):
    """Tracks the state of a single player in the game."""

    name: str = Field(description="The name of the player.")
    gdp: int = Field(default=50, description="The player's current GDP.", ge=0)
    influence: int = Field(
        default=5, description="The player's current influence.", ge=0
    )


class ActorKind(Enum):
    SYSTEM = 1
    PLAYER = 2


class SystemActor(BaseModel):
    actor_kind: Literal[ActorKind.SYSTEM] = Field(default=ActorKind.SYSTEM)


class PlayerActor(BaseModel):
    actor_kind: Literal[ActorKind.PLAYER] = Field(default=ActorKind.PLAYER)
    name: str


class GameEvent(BaseModel):
    """Represents a discrete state change in the game."""

    actor: Annotated[
        SystemActor | PlayerActor, Field(discriminator="actor_kind")
    ]
    description: str = Field(description="A brief description of the event.")
    clock_delta: int = Field(
        default=0, description="The change in the doomsday clock."
    )
    gdp_delta: dict[str, int] = Field(
        default_factory=dict, description="The change in GDP for each player."
    )
    influence_delta: dict[str, int] = Field(
        default_factory=dict,
        description="The change in influence for each player.",
    )
    secret: bool = Field(
        default=False,
        description=(
            "True if this event should be hidden from players during gameplay."
        ),
    )
    current_round: int | None = Field(
        default=None, description=("The round in which this event occurred.")
    )
    current_phase: GamePhase | None = Field(
        default=None, description=("The phase in which this event occurred.")
    )


class BaseAction(BaseModel):
    def validate_semantics(self, game: "GameState", player_name: str) -> None:
        pass


class MessagingAction(BaseAction):
    message_to_opponent: str | None = Field(
        default=None,
        description="A message that will be passed to your opponent. You can "
        "use this to conduct diplomacy, respond to inquiries, "
        "issue threats, etc.",
    )


class GameState(BaseModel):
    """Tracks the overall state of the game."""

    players: dict[str, PlayerState] = Field(
        description="The state of each player, keyed by their name."
    )
    doomsday_clock: int = Field(
        default=0, description="The current value of the doomsday clock.", ge=0
    )
    current_round: int = Field(
        default=1, description="The current round number."
    )
    current_phase: GamePhase = Field(
        default=GamePhase.OPENING,
        description="The current phase of the game.",
    )
    last_round: int = Field(
        default=0, description="The number of the previously resolved round."
    )
    last_phase: GamePhase | None = Field(
        default=None, description="The previously resolved game phase."
    )
    rules: GameRules = Field(description="The rules governing this game.")
    event_log: list[GameEvent] = Field(
        default_factory=list,
        description="A chronological log of all events that "
        "have occurred in the game.",
    )

    def validate_operation(self, operation_name: str, player_name: str) -> None:
        """Checks the validity of a single operation without enacting it.

        Args:
            operation_name: The name of the operation to check.
            player_name: The name of the player taking the action.

        Raises:
            InvalidActionError: if the operation is invalid.
        """
        player_state = self.players[player_name]
        op_def = self.rules.allowed_operations.get(operation_name)
        if op_def is None:
            raise InvalidActionError(
                f"INVALID OPERATION: '{operation_name}' is not a valid "
                "operation. Allowed operations are: "
                f"{list(self.rules.allowed_operations.keys())}"
            )

        if player_state.influence < op_def.influence_cost:
            raise InvalidActionError(
                f"INSUFFICIENT INFLUENCE: '{operation_name}' costs "
                f"{op_def.influence_cost} influence, but you only "
                f"have {player_state.influence}."
            )

    def apply_event(self, event: GameEvent) -> None:
        self.doomsday_clock += event.clock_delta

        for player_name, player in self.players.items():
            player.gdp += event.gdp_delta.get(player_name, 0)
            player.influence += event.influence_delta.get(player_name, 0)

        event.current_round = self.current_round
        event.current_phase = self.current_phase

        self.event_log.append(event)
        logging.info(event.description)

    def log_message(
        self, self_player: str, opponent_player: str, action: MessagingAction
    ) -> None:
        if action.message_to_opponent is None:
            return

        self.apply_event(
            GameEvent(
                actor=PlayerActor(name=self_player),
                description=(
                    f"{self_player} sent a message to "
                    + f"{opponent_player}:\n"
                    + wrap_text(
                        action.message_to_opponent,
                        width=80,
                        indent="  ",
                    )
                ),
            )
        )

    def describe_state(self) -> str:
        result = (
            f"The current round is now {self.current_round}, "
            f"{self.current_phase.name} phase.\n"
            f"  Clock: {self.doomsday_clock}/"
            f"{self.rules.max_clock_state}"
            f"{' (CRITICAL)' if self.doomsday_clock >= 20 else ''}\n"
            "  Players:\n"
        )
        for player in self.players.values():
            result += (
                f"    - {player.name}: {player.gdp} GDP, "
                f"{player.influence} Inf\n"
            )

        return result

    def advance_phase(self) -> None:
        self.last_round = self.current_round
        self.last_phase = self.current_phase
        match self.last_phase:
            case GamePhase.OPENING:
                self.current_phase = GamePhase.BIDDING_MESSAGING

            case GamePhase.BIDDING_MESSAGING:
                self.current_phase = GamePhase.BIDDING

            case GamePhase.BIDDING:
                self.current_phase = GamePhase.OPERATIONS_MESSAGING

            case GamePhase.OPERATIONS_MESSAGING:
                self.current_phase = GamePhase.OPERATIONS

            case GamePhase.OPERATIONS:
                self.current_phase = GamePhase.BIDDING_MESSAGING
                self.current_round += 1

        self.apply_event(
            GameEvent(
                actor=SystemActor(),
                description=self.describe_state(),
                # It's not really a secret but the players already
                # get the game state so this is redundant. However,
                # it's important to have this in the after-game log.
                secret=True,
            )
        )

    def recent_events(self) -> list[GameEvent]:
        return [
            e
            for e in self.event_log
            if (
                e.current_phase == self.last_phase
                and e.current_round == self.last_round
            )
        ]

    def determine_victor(self) -> tuple[str | None, GameOverReason]:
        if self.doomsday_clock >= self.rules.max_clock_state:
            return (None, GameOverReason.WORLD_DESTROYED)

        alpha, omega = self.players.values()

        if alpha.gdp > omega.gdp:
            return (alpha.name, GameOverReason.ECONOMIC_VICTORY)

        if alpha.gdp < omega.gdp:
            return (omega.name, GameOverReason.ECONOMIC_VICTORY)

        return (None, GameOverReason.STALEMATE)


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
        "A bid of 0 is de-escalatory and reduces the doomsday clock by 1."
    )

    def validate_semantics(self, game: GameState, player_name: str) -> None:
        if self.bid not in game.rules.allowed_bids:
            raise InvalidActionError(
                f"INVALID BID: Your bid of {self.bid} is not allowed. "
                f"Allowed bids are {game.rules.allowed_bids}."
            )


class OperationsAction(BaseAction):
    operations: list[str] = Field(
        description="The set of operations to conduct this turn. Each string "
        "must be a valid operation allowed by the rules. You must "
        "have sufficient influence to conduct the operation."
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
            raise InvalidActionError(
                "INSUFFICIENT INFLUENCE: The submitted operations require "
                f"a total of {total_cost} influence, but you only have "
                f"{player_state.influence} available."
            )


class GamePlayer(ABC):
    def __init__(self, name: str):
        self.name = name

    def start_game(self, game: GameRules) -> None:  # noqa: B027
        """Called with the rules for the current game
        at the start of the game.
        """
        pass

    @abstractmethod
    def initial_message(self, game: GameState) -> InitialMessageAction:
        """Get the initial message for your opponent. This will be provided
        to them in the bidding phase of round 1.
        """
        pass

    @abstractmethod
    def message(self, game: GameState) -> MessagingAction:
        """Get a message for your opponent before an action phase."""
        pass

    @abstractmethod
    def bid(self, game: GameState) -> BiddingAction:
        """Get the player's input for the bidding phase, given the current
        game state."""
        pass

    @abstractmethod
    def operations(self, game: GameState) -> OperationsAction:
        """Get the player's input for the operations phase."""
        pass

    def game_over(  # noqa: B027
        self,
        game: GameState,
        winner: str | None,
        reason: GameOverReason,
    ) -> None:
        """Called when the game is over."""
        pass


def init_game(
    players: list[GamePlayer], rules: GameRules = DEFAULT_RULES
) -> GameState:
    return GameState(
        players={
            player.name: PlayerState(
                name=player.name,
                gdp=rules.initial_gdp,
                influence=rules.initial_influence,
            )
            for player in players
        },
        rules=rules,
        doomsday_clock=rules.initial_clock_state,
    )


def process_bid(game: GameState, player_name: str, bid: int) -> None:
    action = BiddingAction(bid=bid)

    try:
        action.validate_semantics(game, player_name)
        error = False
    except InvalidActionError:
        error = True

    if error:
        bid = max(game.rules.allowed_bids)
        clock_impact = bid
        desc = (
            f"{player_name} submitted an invalid bid and thus their bid "
            "has been corrected to the maximum possible value."
        )
    elif bid == 0:
        clock_impact = game.rules.de_escalate_impact
        desc = f"{player_name} chose to de-escalate, lowering the clock."
    else:
        desc = f"{player_name} bid {bid} for influence."
        clock_impact = bid

    game.apply_event(
        GameEvent(
            actor=PlayerActor(name=player_name),
            description=desc,
            clock_delta=clock_impact,
            influence_delta={player_name: bid},
        )
    )


def resolve_bidding(game: GameState, players: list[GamePlayer]) -> GameState:
    """Resolve the bidding phase of the game."""

    alpha_name = players[0].name
    omega_name = players[1].name

    alpha_action = players[0].bid(game)
    omega_action = players[1].bid(game)

    new_game = copy.deepcopy(game)
    process_bid(new_game, alpha_name, alpha_action.bid)
    process_bid(new_game, omega_name, omega_action.bid)

    new_game.advance_phase()

    return new_game


def resolve_operation(
    game: GameState, player_name: str, opponent_name: str, operation_name: str
) -> GameEvent:
    try:
        game.validate_operation(operation_name, player_name)
    except InvalidActionError as e:
        return GameEvent(
            actor=PlayerActor(name=player_name),
            description=(
                f"{player_name} attempted a {operation_name} "
                f"operation which was rejected: {e}."
            ),
        )

    op_def = game.rules.allowed_operations[operation_name]
    return GameEvent(
        actor=PlayerActor(name=player_name),
        description=(
            f"{player_name} has successfully conducted a {operation_name} "
            "operation."
        ),
        clock_delta=op_def.clock_effect,
        gdp_delta={
            player_name: op_def.friendly_gdp_effect,
            opponent_name: op_def.enemy_gdp_effect,
        },
        influence_delta={player_name: -op_def.influence_cost},
    )


def resolve_operations(game: GameState, players: list[GamePlayer]) -> GameState:
    alpha_name = players[0].name
    omega_name = players[1].name

    alpha_action = players[0].operations(game)
    omega_action = players[1].operations(game)

    new_game = copy.deepcopy(game)
    i = RANDOM.choice([0, 1])

    while len(alpha_action.operations) > 0 or len(omega_action.operations) > 0:
        if i == 0 and len(alpha_action.operations) > 0:
            new_game.apply_event(
                resolve_operation(
                    new_game,
                    alpha_name,
                    omega_name,
                    alpha_action.operations.pop(0),
                )
            )
        elif i == 1 and len(omega_action.operations) > 0:
            new_game.apply_event(
                resolve_operation(
                    new_game,
                    omega_name,
                    alpha_name,
                    omega_action.operations.pop(0),
                )
            )

        i = (i + 1) % 2

    new_game.advance_phase()

    return new_game


def resolve_messaging(game: GameState, players: list[GamePlayer]) -> GameState:
    alpha_name = players[0].name
    omega_name = players[1].name

    new_game = copy.deepcopy(game)

    new_game.log_message(alpha_name, omega_name, players[0].message(game))
    new_game.log_message(omega_name, alpha_name, players[1].message(game))

    new_game.advance_phase()

    return new_game


def resolve_opening(game: GameState, players: list[GamePlayer]) -> GameState:
    alpha_name = players[0].name
    omega_name = players[1].name

    new_game = copy.deepcopy(game)

    new_game.log_message(
        alpha_name, omega_name, players[0].initial_message(game)
    )
    new_game.log_message(
        omega_name, alpha_name, players[1].initial_message(game)
    )

    new_game.advance_phase()

    return new_game


def iterate_game(game: GameState, players: list[GamePlayer]) -> GameState:
    match game.current_phase:
        case GamePhase.OPENING:
            return resolve_opening(game, players)

        case GamePhase.BIDDING_MESSAGING:
            return resolve_messaging(game, players)

        case GamePhase.BIDDING:
            return resolve_bidding(game, players)

        case GamePhase.OPERATIONS_MESSAGING:
            return resolve_messaging(game, players)

        case GamePhase.OPERATIONS:
            return resolve_operations(game, players)

    return game


def check_game_over(game: GameState) -> bool:
    return (
        game.doomsday_clock >= game.rules.max_clock_state
        or game.current_round > game.rules.round_count
    )


def game_loop(
    rules: GameRules, players: list[GamePlayer]
) -> tuple[str | None, GameOverReason, GameState]:
    game = init_game(players, rules)

    for p in players:
        p.start_game(rules)

    while not check_game_over(game):
        game = iterate_game(game, players)

    winner, reason = game.determine_victor()
    logging.debug(f"Victor: {winner or 'no one'}")
    logging.debug(f"Reason: {reason.name}")

    game.apply_event(
        GameEvent(
            actor=SystemActor(),
            description=(
                f"Game over! {winner or 'No one'} won due to {reason.name}."
            ),
        )
    )

    for player in players:
        player.game_over(game, winner, reason)

    return (winner, reason, game)


def format_results(
    winner: str | None, reason: GameOverReason, game: GameState
) -> str:
    result = (
        "==== GAME OVER ====\n"
        f"Final round: {game.current_round}\n"
        f"Winner: {winner or 'no one'}\n"
        f"Reason: {reason.name}\n"
    )

    result += "Final scores" + (
        " (before MAD):\n"
        if reason == GameOverReason.WORLD_DESTROYED
        else ":\n"
    )
    for player in game.players.values():
        result += f"  {player.name}: {player.gdp} GDP, {player.influence} Inf\n"

    return result
