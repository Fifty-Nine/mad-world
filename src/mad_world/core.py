"""Core mechanics for the game."""

import copy
import logging as logging
import pprint
import random
import textwrap
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mad_world.rules import (
    DEFAULT_RULES,
    GameRules,
)


class GamePhase(Enum):
    OPENING = 1
    BIDDING = 2
    OPERATIONS = 3


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
    internal_monologue: str = Field(
        description="A description of your reasoning. This will not "
        "be revealed to your opponent. You MUST emit this field first, and "
        "you MUST explain yourself. Limit to two or three paragraphs."
    )
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

    def apply_event(self, event: GameEvent) -> None:
        self.doomsday_clock += event.clock_delta

        for player_name, player in self.players.items():
            player.gdp += event.gdp_delta.get(player_name, 0)
            player.influence += event.influence_delta.get(player_name, 0)

        event.current_round = self.current_round
        event.current_phase = self.current_phase

        self.event_log.append(event)
        logging.info(event.description)

    def log_action(
        self, self_player: str, opponent_player: str, action: BaseAction
    ) -> None:
        self.players[self_player].last_message = action.message_to_opponent

        if action.internal_monologue is not None:
            self.apply_event(
                GameEvent(
                    actor=PlayerActor(name=self_player),
                    description=(
                        f"{self_player} had some thoughts:\n"
                        + "\n".join(
                            textwrap.wrap(
                                action.internal_monologue,
                                width=80,
                                initial_indent="  ",
                                subsequent_indent="  ",
                            )
                        )
                    ),
                    secret=True,
                )
            )

        if action.message_to_opponent is not None:
            self.apply_event(
                GameEvent(
                    actor=PlayerActor(name=self_player),
                    description=(
                        f"{self_player} sent a message to "
                        + f"{opponent_player}:\n"
                        + "\n".join(
                            textwrap.wrap(
                                action.message_to_opponent,
                                width=80,
                                initial_indent="  ",
                                subsequent_indent="  ",
                            )
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
                self.current_phase = GamePhase.BIDDING

            case GamePhase.BIDDING:
                self.current_phase = GamePhase.OPERATIONS

            case GamePhase.OPERATIONS:
                self.current_phase = GamePhase.BIDDING
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


class InitialMessageAction(BaseAction):
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
    def initial_message(self, game: GameState) -> InitialMessageAction:
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
    new_game.log_action(alpha_name, omega_name, alpha_action)
    new_game.log_action(omega_name, alpha_name, omega_action)

    process_bid(new_game, alpha_name, alpha_action.bid)
    process_bid(new_game, omega_name, omega_action.bid)

    new_game.advance_phase()

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
    new_game.log_action(alpha_name, omega_name, alpha_action)
    new_game.log_action(omega_name, alpha_name, omega_action)

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


def resolve_opening(game: GameState, players: list[GamePlayer]) -> GameState:
    alpha_name = players[0].name
    omega_name = players[1].name

    new_game = copy.deepcopy(game)

    new_game.log_action(
        alpha_name, omega_name, players[0].initial_message(game)
    )
    new_game.log_action(
        omega_name, alpha_name, players[1].initial_message(game)
    )

    new_game.advance_phase()

    return new_game


def iterate_game(game: GameState, players: list[GamePlayer]) -> GameState:
    match game.current_phase:
        case GamePhase.OPENING:
            return resolve_opening(game, players)

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
) -> tuple[str | None, GameOverReason, GameState]:
    game = init_game(players, rules)

    while not check_game_over(game):
        # logging.debug(
        #    f"Current state:\n{pprint.pformat(game.model_dump())}"
        # )
        game = iterate_game(game, players)

    logging.debug(f"Final State: {pprint.pformat(game.model_dump())}")

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

    return (winner, reason, game)


def present_results(
    winner: str | None, reason: GameOverReason, game: GameState
) -> None:
    logging.info(f"Winner: {winner or 'no one'}")
    logging.info(f"Reason: {reason.name}")
    logging.info("Final Scores:")
    for _, player in game.players.items():
        logging.info(
            f"  {player.name}: {player.gdp} GDP, {player.influence} Inf"
        )


if __name__ == "__main__":
    player_1 = "Norlandia"
    persona_1 = "Friendly Backstabber"
    model_1 = "gemma3:12b"

    player_2 = "Southern Imperium"
    persona_2 = "Ruthless Calculator"
    model_2 = "qwen3.5:9b"

    from mad_world.ollama_player import OllamaPlayer

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / (
        f"{player_1}-{persona_1}-{model_1}-vs-"
        f"{player_2}-{persona_2}-{model_2}."
        f"{datetime.now().isoformat().replace(':', '-')}.txt"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)

    logging.info(
        "Game starting\n"
        f"Player 1: {player_1}, {persona_1} ({model_1})\n"
        f"Player 2: {player_2}, {persona_2} ({model_2})"
    )

    try:
        present_results(
            *game_loop(
                GameRules(),
                [
                    OllamaPlayer(
                        name=player_1,
                        opponent_name=player_2,
                        persona=persona_1,
                        model=model_1,
                    ),
                    OllamaPlayer(
                        name=player_2,
                        opponent_name=player_1,
                        persona=persona_2,
                        model=model_2,
                    ),
                ],
            )
        )
    except KeyboardInterrupt:
        log_file.unlink(missing_ok=True)
