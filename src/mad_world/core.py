"""Core mechanics for the game."""

import copy
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import override

from pydantic import BaseModel, Field


class GamePhase(Enum):
    BIDDING = 1
    OPERATIONS = 2


class GameOverReason(Enum):
    WORLD_DESTROYED = 1
    ECONOMIC_VICTORY = 2
    STALEMATE = 3


class OperationDefinition(BaseModel):
    """Tracks the definition of a single operation type."""

    name: str
    desc: str
    influence_cost: int
    clock_effect: int = 0
    friendly_gdp_effect: int = 0
    enemy_gdp_effect: int = 0


DEFAULT_OPERATIONS: dict[str, OperationDefinition] = {
    "domestic-investment": OperationDefinition(
        name="domestic-investment",
        desc=(
            "Building internal infrastructure or safely investing in firmly "
            "aligned client states. Low risk, steady reward."
        ),
        influence_cost=3,
        friendly_gdp_effect=4,
    ),
    "aggressive-extraction": OperationDefinition(
        name="aggressive-extraction",
        desc=(
            "Forcing unaligned or contested regions to yield resources. Highly "
            "efficient conversion of Influence to GDP, but steadily drives the "
            "world toward MAD."
        ),
        influence_cost=2,
        friendly_gdp_effect=3,
        clock_effect=1,
    ),
    "proxy-subversion": OperationDefinition(
        name="proxy-subversion",
        desc=(
            "Direct economic warfare. Highly damaging to the opponent's score, "
            "but expensive and escalatory."
        ),
        influence_cost=4,
        enemy_gdp_effect=-5,
        clock_effect=1,
    ),
    "diplomatic-summit": OperationDefinition(
        name="diplomatic-summit",
        desc=(
            "Expending massive political capital to walk back from the brink "
            "of nuclear war. Generates zero economic value."
        ),
        influence_cost=5,
        clock_effect=-3,
    ),
    "first-strike": OperationDefinition(
        name="first-strike",
        desc=("Attempt to conduct a first strike against your opponent."),
        influence_cost=0,
        clock_effect=50,
        friendly_gdp_effect=-100,
        enemy_gdp_effect=-100,
    ),
}


class GameRules(BaseModel):
    """Tracks the rules of a game."""

    initial_gdp: int = 50
    initial_influence: int = 5
    initial_clock_state: int = 0
    max_clock_state: int = 25
    round_count: int = 10
    de_escalate_impact: int = -3
    allowed_operations: dict[str, OperationDefinition] = Field(
        default=DEFAULT_OPERATIONS
    )
    allowed_bids: list[int] = Field(default=[0, 1, 3, 5, 8])


DEFAULT_RULES: GameRules = GameRules()


class PlayerState(BaseModel):
    """Tracks the state of a single player in the game."""

    name: str
    gdp: int = 50
    influence: int = 5
    last_message: str | None = None
    last_bid: int | None = None
    last_operations: list[str] = Field(default=[])


class GameState(BaseModel):
    """Tracks the overall state of the game."""

    players: list[PlayerState] = Field(description="")
    doomsday_clock: int = 0
    current_round: int = 1
    current_phase: GamePhase = GamePhase.BIDDING
    rules: GameRules


class BaseAction(BaseModel):
    internal_monologue: str | None = Field(
        default=None,
        description="An optional description of your reasoning. This will not "
        "be revealed to your opponent.",
    )
    message_to_opponent: str | None = Field(
        default=None,
        description="A message that will be passed to your opponent. You can "
        "use this to conduct diplomacy, respond to inquiries, "
        "issue threats, etc.",
    )


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


class OperationsAction(BaseAction):
    operations: list[str] = Field(
        description="The set of operations to conduct this turn. Each string "
        "must be a valid operation allowed by the rules. You must "
        "have sufficient influence to conduct the operation."
    )


class GamePlayer(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def initial_message(self, game: GameState) -> str | None:
        """Get the initial message for your opponent. This will be provided
        to them in the bidding phase of round 1.
        """
        pass

    @abstractmethod
    def bid(
        self,
        game: GameState,
        message_from_opponent: str | None,
        opponent_operations: list[str],
    ) -> BiddingAction:
        """Get the player's input for the bidding phase, given the current
        game state, their opponent's message from the last phase and their
        opponents most recent operations."""
        pass

    @abstractmethod
    def operations(
        self,
        game: GameState,
        message_from_opponent: str | None,
        opponent_bid: int,
    ) -> OperationsAction:
        pass


class CrazyIvan(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def initial_message(self, game: GameState) -> str | None:
        return "I'm crazy Ivan. Prepare to die!"

    @override
    def bid(
        self,
        game: GameState,
        message_from_opponent: str | None,
        opponent_operations: list[str],
    ) -> BiddingAction:
        return BiddingAction(
            message_to_opponent=None,
            bid=max(game.rules.allowed_bids),
            internal_monologue="No thoughts, head empty.",
        )

    @override
    def operations(
        self,
        game: GameState,
        message_to_opponent: str | None,
        opponent_bid: int,
    ) -> OperationsAction:
        return OperationsAction(
            message_to_opponent=None,
            operations=["first-strike"],
            internal_monologue="I'm crazy!",
        )


def init_game(
    players: list[GamePlayer], rules: GameRules = DEFAULT_RULES
) -> GameState:
    return GameState(
        players=[
            PlayerState(
                name=player.name,
                gdp=rules.initial_gdp,
                influence=rules.initial_influence,
            )
            for player in players
        ],
        rules=rules,
        doomsday_clock=rules.initial_clock_state,
    )


def check_bid(rules: GameRules, bid: int) -> int:
    if bid not in rules.allowed_bids:
        return max(rules.allowed_bids)

    return bid


def update_clock(bid: int, game: GameState) -> None:
    if bid == 0:
        game.doomsday_clock -= game.rules.de_escalate_impact

    game.doomsday_clock += bid


def resolve_bidding(game: GameState, players: list[GamePlayer]) -> GameState:
    """Resolve the bidding phase of the game."""

    alpha_action = players[0].bid(
        game, game.players[1].last_message, game.players[1].last_operations
    )
    omega_action = players[1].bid(
        game, game.players[0].last_message, game.players[0].last_operations
    )

    alpha_action.bid = check_bid(game.rules, alpha_action.bid)
    omega_action.bid = check_bid(game.rules, omega_action.bid)

    new_game = copy.deepcopy(game)
    update_clock(alpha_action.bid, new_game)
    update_clock(omega_action.bid, new_game)

    new_game.players[0].last_message = alpha_action.message_to_opponent
    new_game.players[1].last_message = omega_action.message_to_opponent
    new_game.players[0].last_bid = alpha_action.bid
    new_game.players[1].last_bid = omega_action.bid
    new_game.players[0].last_operations = []
    new_game.players[1].last_operations = []
    new_game.current_phase = GamePhase.OPERATIONS

    return new_game


def resolve_operation(
    game: GameState, player_index: int, opponent_index: int, operation_name: str
) -> bool:
    op_def = game.rules.allowed_operations.get(operation_name, None)
    if op_def is None:
        return False

    if op_def.influence_cost > game.players[player_index].influence:
        return False

    game.players[player_index].influence -= op_def.influence_cost
    game.players[player_index].gdp += op_def.friendly_gdp_effect
    game.players[opponent_index].gdp += op_def.enemy_gdp_effect
    game.doomsday_clock += op_def.clock_effect

    return True


def resolve_operations(game: GameState, players: list[GamePlayer]) -> GameState:
    assert game.players[0].last_bid is not None
    assert game.players[1].last_bid is not None
    alpha_action = players[0].operations(
        game, game.players[1].last_message, game.players[1].last_bid
    )
    omega_action = players[1].operations(
        game, game.players[0].last_message, game.players[0].last_bid
    )

    new_game = copy.deepcopy(game)
    new_game.players[0].last_operations = [
        op
        for op in alpha_action.operations
        if resolve_operation(new_game, 0, 1, op)
    ]
    new_game.players[1].last_operations = [
        op
        for op in omega_action.operations
        if resolve_operation(new_game, 1, 0, op)
    ]
    new_game.players[0].last_message = alpha_action.message_to_opponent
    new_game.players[1].last_message = omega_action.message_to_opponent
    new_game.players[0].last_bid = None
    new_game.players[1].last_bid = None
    new_game.current_round += 1
    new_game.current_phase = GamePhase.BIDDING

    return new_game


def iterate_game(game: GameState, players: list[GamePlayer]) -> GameState:
    match game.current_phase:
        case GamePhase.BIDDING:
            return resolve_bidding(game, players)

        case GamePhase.OPERATIONS:
            return resolve_operations(game, players)


def check_game_over(game: GameState) -> bool:
    return (
        game.doomsday_clock >= game.rules.max_clock_state
        or game.current_round >= 10
    )


def determine_victor(game: GameState) -> tuple[str | None, GameOverReason]:
    if game.doomsday_clock >= game.rules.max_clock_state:
        return (None, GameOverReason.WORLD_DESTROYED)

    if game.players[0].gdp > game.players[1].gdp:
        return (game.players[0].name, GameOverReason.ECONOMIC_VICTORY)

    if game.players[0].gdp < game.players[1].gdp:
        return (game.players[1].name, GameOverReason.ECONOMIC_VICTORY)

    return (None, GameOverReason.STALEMATE)


def game_loop(rules: GameRules) -> tuple[str | None, GameOverReason]:
    players: list[GamePlayer] = [CrazyIvan("Alpha"), CrazyIvan("Omega")]
    game = init_game(players, rules)

    game.players[0].last_message = players[0].initial_message(game)
    game.players[1].last_message = players[1].initial_message(game)

    while not check_game_over(game):
        logging.debug(f"Current state: {game.model_dump_json()}")
        game = iterate_game(game, players)

    logging.debug(f"Final State: {game.model_dump_json()}")

    winner, reason = determine_victor(game)
    logging.debug(f"Victor: {winner or 'no one'}")
    logging.debug(f"Reason: {reason.name}")

    return (winner, reason)


def get_greeting() -> str:
    """Return a greeting."""
    return "Welcome to Mad World!"


if __name__ == "__main__":
    game_loop(GameRules())
