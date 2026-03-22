"""Core mechanics for the game."""

import copy
import logging
import pprint
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Literal

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

    name: str = Field(description="The name of the operation.")
    description: str = Field(
        description="A brief description of the operation."
    )
    influence_cost: int = Field(
        description="The influence cost of the operation."
    )
    clock_effect: int = Field(
        description="The clock impact of the operation.", default=0
    )
    friendly_gdp_effect: int = Field(
        description="The GDP impact on the acting player.", default=0
    )
    enemy_gdp_effect: int = Field(
        description="The GDP impact on the opposing player.", default=0
    )


DEFAULT_OPERATIONS: dict[str, OperationDefinition] = {
    "domestic-investment": OperationDefinition(
        name="domestic-investment",
        description=(
            "Building internal infrastructure or safely investing in firmly "
            "aligned client states. Low risk, steady reward."
        ),
        influence_cost=3,
        friendly_gdp_effect=4,
    ),
    "aggressive-extraction": OperationDefinition(
        name="aggressive-extraction",
        description=(
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
        description=(
            "Direct economic warfare. Highly damaging to the opponent's score, "
            "but expensive and escalatory."
        ),
        influence_cost=4,
        enemy_gdp_effect=-5,
        clock_effect=1,
    ),
    "diplomatic-summit": OperationDefinition(
        name="diplomatic-summit",
        description=(
            "Expending massive political capital to walk back from the brink "
            "of nuclear war. Generates zero economic value."
        ),
        influence_cost=5,
        clock_effect=-3,
    ),
    "first-strike": OperationDefinition(
        name="first-strike",
        description=(
            "Attempt to conduct a first strike against your opponent."
        ),
        influence_cost=0,
        clock_effect=50,
        friendly_gdp_effect=-100,
        enemy_gdp_effect=-100,
    ),
}


class GameRules(BaseModel):
    """Tracks the rules of a game."""

    initial_gdp: int = Field(default=50, description="Initial GDP value.")
    initial_influence: int = Field(
        default=5, description="Initial influence value."
    )
    initial_clock_state: int = Field(
        default=0, description="Initial doomsday clock value."
    )
    max_clock_state: int = Field(
        default=25, description="Maximum doomsday clock value."
    )
    round_count: int = Field(
        default=10, description="Maximum number of rounds."
    )
    de_escalate_impact: int = Field(
        default=-1, description="The clock impact of a de-escalatory bid."
    )
    allowed_operations: dict[str, OperationDefinition] = Field(
        default=DEFAULT_OPERATIONS,
        description="The set of operations allowed in the game.",
    )
    allowed_bids: list[int] = Field(
        default=[0, 1, 3, 5, 8],
        description="The set of bids allowed in the game.",
    )


DEFAULT_RULES: GameRules = GameRules()
RANDOM = random.Random()


class PlayerState(BaseModel):
    """Tracks the state of a single player in the game."""

    name: str = Field(description="The name of the player.")
    gdp: int = Field(default=50, description="The player's current GDP.", ge=0)
    influence: int = Field(
        default=5, description="The player's current influence.", ge=0
    )
    last_message: str | None = Field(
        default=None,
        description=("The last message sent by this player to their opponent."),
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
        default=GamePhase.BIDDING,
        description="The current phase of the game.",
    )
    rules: GameRules = Field(description="The rules governing this game.")
    event_log: list[GameEvent] = Field(
        default_factory=list,
        description="A chronological log of all events that "
        "have occurred in the game.",
    )

    def apply_event(self, event: GameEvent) -> None:
        self.doomsday_clock += event.clock_delta

        for player_name, player in self.players.items():
            player.gdp += event.gdp_delta.get(player_name, 0)
            player.influence += event.influence_delta.get(player_name, 0)

        self.event_log.append(event)

    def send_message(
        self, from_player: str, to_player: str, message: str | None
    ) -> None:
        self.players[from_player].last_message = message
        if message is None:
            return

        self.apply_event(
            GameEvent(
                actor=PlayerActor(name=from_player),
                description=(
                    f"{from_player} sent a message to {to_player}: {message}"
                ),
            )
        )


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
    ) -> OperationsAction:
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
    if bid not in game.rules.allowed_bids:
        bid = max(game.rules.allowed_bids)
        clock_impact = bid
        desc = (
            f"{player_name} submitted an invalid bid and thus their bid "
            "has been corrected to the maximum possible value."
        )
    if bid == 0:
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

    alpha_action = players[0].bid(game, game.players[omega_name].last_message)
    omega_action = players[1].bid(game, game.players[alpha_name].last_message)

    new_game = copy.deepcopy(game)
    process_bid(new_game, alpha_name, alpha_action.bid)
    process_bid(new_game, omega_name, omega_action.bid)

    new_game.send_message(
        alpha_name, omega_name, alpha_action.message_to_opponent
    )
    new_game.send_message(
        omega_name, alpha_name, omega_action.message_to_opponent
    )

    new_game.current_phase = GamePhase.OPERATIONS

    return new_game


def resolve_operation(
    game: GameState, player_name: str, opponent_name: str, operation_name: str
) -> GameEvent:
    op_def = game.rules.allowed_operations.get(operation_name, None)
    if op_def is None:
        return GameEvent(
            actor=PlayerActor(name=player_name),
            description=(
                f"{player_name} attempted an operation not allowed by the "
                f'current rules ("{operation_name}") and as a result the '
                "action is null and void."
            ),
        )

    if op_def.influence_cost > game.players[player_name].influence:
        return GameEvent(
            actor=PlayerActor(name=player_name),
            description=(
                f"{player_name} attempted to perform an operation "
                f'("{operation_name}") but lacked sufficient influence to do '
                "so. As a result the action is null and void."
            ),
        )

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

    alpha_action = players[0].operations(
        game, game.players[omega_name].last_message
    )
    omega_action = players[1].operations(
        game, game.players[alpha_name].last_message
    )

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

    new_game.send_message(
        alpha_name, omega_name, alpha_action.message_to_opponent
    )
    new_game.send_message(
        omega_name, alpha_name, omega_action.message_to_opponent
    )

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
        or game.current_round > game.rules.round_count
    )


def determine_victor(game: GameState) -> tuple[str | None, GameOverReason]:
    if game.doomsday_clock >= game.rules.max_clock_state:
        return (None, GameOverReason.WORLD_DESTROYED)

    alpha, omega = game.players.values()

    if alpha.gdp > omega.gdp:
        return (alpha.name, GameOverReason.ECONOMIC_VICTORY)

    if alpha.gdp < omega.gdp:
        return (omega.name, GameOverReason.ECONOMIC_VICTORY)

    return (None, GameOverReason.STALEMATE)


def game_loop(
    rules: GameRules, players: list[GamePlayer]
) -> tuple[str | None, GameOverReason, list[GameEvent]]:
    game = init_game(players, rules)

    alpha_name = players[0].name
    omega_name = players[1].name

    game.send_message(alpha_name, omega_name, players[0].initial_message(game))
    game.send_message(omega_name, alpha_name, players[1].initial_message(game))

    while not check_game_over(game):
        logging.debug(f"Current state: {game.model_dump_json()}")
        game = iterate_game(game, players)

    logging.debug(f"Final State: {game.model_dump_json()}")

    winner, reason = determine_victor(game)
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

    return (winner, reason, game.event_log)


if __name__ == "__main__":
    from mad_world.trivial_players import CrazyIvan

    logging.basicConfig(level=logging.DEBUG)
    pprint.pprint(
        game_loop(GameRules(), [CrazyIvan("Alpha"), CrazyIvan("Omega")])
    )
