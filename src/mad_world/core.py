"""Core mechanics for the game."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import random
from dataclasses import dataclass
from functools import reduce
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import anyio
from pydantic import BaseModel, Field

from mad_world.actions import (
    BiddingAction,
    ChatAction,
    InitialMessageAction,
    InsufficientInfluenceError,
    InvalidActionError,
    InvalidOperationError,
    MessagingAction,
)
from mad_world.crises import BaseCrisis, create_crisis_deck
from mad_world.decks import Deck
from mad_world.effects import BaseEffect
from mad_world.enums import GameOverReason, GamePhase
from mad_world.event_cards import BaseEventCard, create_event_deck
from mad_world.event_stream import EventStream
from mad_world.events import (
    ActionEvent,
    BaseGameEvent,
    BiddingEvent,
    ChannelOpenedEvent,
    ChannelRejectedEvent,
    CrisisResolutionEvent,
    GameEvent,
    LoggedEvent,
    MandateFulfilledEvent,
    MessageEvent,
    OperationConductedEvent,
    OptActor,
    PlayerActor,
    StateEvent,
    SystemActor,
    SystemEvent,
)
from mad_world.mandates import BaseMandate, create_mandate_deck
from mad_world.rng import SerializableRandom
from mad_world.rules import (
    GameRules,
    OperationDefinition,
)
from mad_world.util import bannerize, escalation_bar, wrap_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mad_world.players import GamePlayer


CLOCK_WARNING_THRESHOLD = 0.8


@dataclass
class WorldDestroyed(Exception):
    """Exception thrown when the world is destroyed."""

    instigator: str | None


class PlayerState(BaseModel):
    """Tracks the state of a single player in the game."""

    name: str = Field(description="The name of the player.")
    gdp: int = Field(default=50, description="The player's current GDP.", ge=0)
    influence: int = Field(
        default=5,
        description="The player's current influence.",
        ge=0,
    )
    mandates: list[BaseMandate] = Field(
        default_factory=list,
        description="The active mandates held by the player.",
    )
    completed_mandates: list[BaseMandate] = Field(
        default_factory=list,
        description="The mandates completed and revealed by the player.",
    )
    channels_opened: int = Field(
        default=0,
        description=(
            "The number of direct communication channels this player "
            "has successfully requested."
        ),
    )


class GameState(BaseModel):
    """Tracks the overall state of the game."""

    players: dict[str, PlayerState] = Field(
        description="The state of each player, keyed by their name.",
    )
    escalation_track: list[OptActor] = Field(
        default_factory=list,
        description=(
            "A literal track of colored cubes (strings of player names "
            "or 'System') representing escalation debt."
        ),
    )

    @property
    def doomsday_clock(self) -> int:
        """The current value of the doomsday clock."""
        return len(self.escalation_track) - self.escalation_track.count(None)

    current_round: int = Field(
        default=1,
        description="The current round number.",
    )
    current_phase: GamePhase = Field(
        default=GamePhase.OPENING,
        description="The current phase of the game.",
    )
    post_crisis_round: int | None = Field(
        default=None,
        description="The next round after the resolution of crises.",
    )
    post_crisis_phase: GamePhase | None = Field(
        default=None, description="The next phase after resolution of crises."
    )
    crisis_deck: Deck[BaseCrisis] = Field(
        description="The deck from which to deal crises."
    )
    event_deck: Deck[BaseEventCard] = Field(
        description="The deck from which to draw round events."
    )
    mandate_deck: Deck[BaseMandate] = Field(
        description="The deck from which mandates are drawn."
    )
    pending_crisis: BaseCrisis | None = Field(
        default=None,
        description="The pending crisis to resolve.",
    )
    last_round: int = Field(
        default=0,
        description="The number of the previously resolved round.",
    )
    last_phase: GamePhase | None = Field(
        default=None,
        description="The previously resolved game phase.",
    )
    rules: GameRules = Field(description="The rules governing this game.")
    event_log: list[LoggedEvent[GameEvent]] = Field(
        default_factory=list,
        description="A chronological log of all events that "
        "have occurred in the game.",
    )
    log_dir: Path | None = Field(
        default=None,
        description="The directory to store game logs, if any.",
        exclude=True,
    )

    rng: SerializableRandom = Field(
        description="The RNG state used for this game.",
        default_factory=SerializableRandom,
    )
    active_effects: list[BaseEffect] = Field(
        default_factory=list,
        description="The currently active ongoing effects in the game.",
    )

    @property
    def allowed_operations(self) -> dict[str, OperationDefinition]:
        return reduce(
            lambda ops, fn: fn(ops),
            (effect.modify_operations for effect in self.active_effects),
            self.rules.allowed_operations,
        )

    @property
    def allowed_bids(self) -> list[int]:
        return reduce(
            lambda ops, fn: fn(ops),
            (effect.modify_bids for effect in self.active_effects),
            self.rules.allowed_bids,
        )

    @classmethod
    def new_game(
        cls,
        *,
        rules: GameRules,
        players: list[str],
        log_dir: Path | None = None,
        **kwargs: Any,
    ) -> Self:
        """Creates a new game state from the provided rules and players.

        Initializes the deck, states, and initial mandates.
        """
        rng = random.Random(rules.seed)
        result = cls(
            rng=rng,
            rules=rules,
            log_dir=log_dir,
            players={
                player: PlayerState(
                    name=player,
                    gdp=rules.initial_gdp,
                    influence=rules.initial_influence,
                )
                for player in players
            },
            escalation_track=[None] * rules.max_clock_state,
            crisis_deck=create_crisis_deck(rng)
            if rules.initial_crisis_deck is None
            else Deck[BaseCrisis].create(rules.initial_crisis_deck, rng),
            event_deck=create_event_deck(rng)
            if rules.initial_event_deck is None
            else Deck[BaseEventCard].create(rules.initial_event_deck, rng),
            mandate_deck=create_mandate_deck(rng)
            if rules.initial_mandate_deck is None
            else Deck[BaseMandate].create(rules.initial_mandate_deck, rng),
            **kwargs,
        )
        result.escalate(SystemActor(), rules.initial_clock_state)

        for p_name in players:
            num_to_draw = min(2, len(result.mandate_deck))
            for _ in range(num_to_draw):
                result.players[p_name].mandates.append(
                    result.mandate_deck.draw(result.rng)
                )

        return result

    @property
    def player_names(self) -> tuple[str, str]:
        assert len(self.players) == 2
        return cast("tuple[str, str]", tuple(self.players))

    def op_cost(self, op: str) -> int:
        """Helper function for getting the Inf cost of a named operation."""
        return self.allowed_operations[op].influence_cost

    def validate_operation(self, operation_name: str, player_name: str) -> None:
        """Checks the validity of a single operation without enacting it.

        Args:
            operation_name: The name of the operation to check.
            player_name: The name of the player taking the action.

        Raises:
            InvalidActionError: if the operation is invalid.
        """
        player_state = self.players[player_name]
        op_def = self.allowed_operations.get(operation_name)
        if op_def is None:
            raise InvalidOperationError(
                operation=operation_name,
                allowed=list(self.allowed_operations.keys()),
            )

        if player_state.influence < op_def.influence_cost:
            raise InsufficientInfluenceError(
                cost=op_def.influence_cost,
                available=player_state.influence,
                operation=operation_name,
            )

    def _deescalate_one(self, actor: SystemActor | PlayerActor) -> None:
        # First look for any of actor's cubes.
        for i in range(len(self.escalation_track) - 1, -1, -1):
            track_actor = self.escalation_track[i]
            if track_actor == actor or (
                actor.is_system() and track_actor is not None
            ):
                self.escalation_track[i] = None
                return

        # Now look for any cube.
        for i in range(len(self.escalation_track) - 1, -1, -1):
            track_actor = self.escalation_track[i]
            if track_actor is None:
                continue

            self.escalation_track[i] = None
            break

    def _escalate_one(self, actor: SystemActor | PlayerActor) -> None:
        for i in range(len(self.escalation_track)):
            if self.escalation_track[i] is None:
                self.escalation_track[i] = actor
                return

    def escalate(self, actor: SystemActor | PlayerActor, amount: int) -> None:
        """Escalate the doomsday clock by adding cubes to the track."""
        fn = self._escalate_one if amount >= 0 else self._deescalate_one
        amount = abs(amount)
        while amount > 0:
            fn(actor)
            amount -= 1

    def reset_escalation(self) -> None:
        self.escalate(SystemActor(), -self.rules.max_clock_state)

    def apply_event(self, event: GameEvent) -> None:
        """Applies a GameEvent to the state.

        Updates resources, doomsday clock, tracks effects, and handles
        blame shifts.
        """
        self.escalate(event.actor, event.clock_delta)

        if event.shift_blame is not None:
            old_actor = event.actor
            new_actor, amount = event.shift_blame
            old_clock = self.doomsday_clock
            self.escalate(old_actor, -amount)
            new_clock = self.doomsday_clock

            self.escalate(new_actor, old_clock - new_clock)

        for player_name, player in self.players.items():
            player.gdp += event.gdp_delta.get(player_name, 0)
            player.influence += event.influence_delta.get(player_name, 0)
            player.influence = max(0, player.influence)
            player.channels_opened += event.channels_opened.get(player_name, 0)

        for effect in event.new_effects:
            effect.start_round = self.current_round
            self.active_effects.append(effect)

        logged = LoggedEvent(
            round=self.current_round,
            phase=self.current_phase,
            event=event,
        )

        self.event_log.append(logged)
        logging.getLogger("mad_world").info(event.description)

        if not event.world_ending:
            return

        raise WorldDestroyed(instigator=event.actor.player())

    def log_message(
        self,
        self_player: str,
        opponent_player: str,
        action: MessagingAction | ChatAction | InitialMessageAction,
    ) -> None:
        """Records a messaging action between players.

        Logs the action into the game event log as a MessageEvent.
        """
        if isinstance(action, InitialMessageAction):
            message = action.opening_statement
        elif isinstance(action, MessagingAction):
            message = action.message_to_opponent
        else:
            message = action.message

        if message is None:
            return

        self.apply_event(
            MessageEvent(
                actor=PlayerActor(name=self_player),
                description=(
                    f"{self_player} sent a message to "
                    f"{opponent_player}:\n"
                    + wrap_text(
                        message,
                        width=80,
                        indent="  ",
                    )
                ),
                message=message,
                channel_message=isinstance(action, ChatAction),
            ),
        )

    def clock_is_critical(self) -> bool:
        """Check if the clock has reached a "critical" state."""
        return (
            self.doomsday_clock
            >= CLOCK_WARNING_THRESHOLD * self.rules.max_clock_state
        )

    def describe_state(self) -> str:
        """Returns a bannerized string describing the current state.

        Includes the current round, phase, doomsday clock, and player statuses.
        """
        tracker = escalation_bar(self.escalation_track, defrag=True)
        result = (
            bannerize(
                f"ROUND {self.current_round} PHASE {self.current_phase.name}"
            )
            + f"  Clock: {self.doomsday_clock}/"
            f"{self.rules.max_clock_state}"
            f"{' (CRITICAL)' if self.clock_is_critical() else ''}\n"
            "  Escalation Tracker:\n"
            f"{wrap_text(tracker, indent='    ')}"
            "  Players:\n"
        )
        for player in self.players.values():
            result += (
                f"    - {player.name}: {player.gdp} GDP, "
                f"{player.influence} Inf\n"
            )

        if self.pending_crisis is not None:
            result += (
                f"ONGOING CRISIS: {self.pending_crisis.title}\n"
                f"{self.pending_crisis.description}\n"
            )

        return result

    def trigger_crisis(self) -> bool:
        if self.doomsday_clock < self.rules.max_clock_state:
            return False

        if self.current_phase.is_crisis():
            return False

        assert self.pending_crisis is None
        self.pending_crisis = self.crisis_deck.draw(self.rng)

        return True

    def advance_phase(self) -> None:
        """Advances the game phase.

        Evaluates post-crisis states, mandate completions, and expires
        ongoing effects.
        """
        self.last_round = self.current_round
        self.last_phase = self.current_phase
        match self.last_phase:
            case GamePhase.OPENING:
                self.current_phase = GamePhase.ROUND_EVENTS

            case GamePhase.ROUND_EVENTS:
                self.current_phase = GamePhase.BIDDING_MESSAGING

            case GamePhase.BIDDING_MESSAGING:
                self.current_phase = GamePhase.BIDDING

            case GamePhase.BIDDING:
                self.current_phase = GamePhase.OPERATIONS_MESSAGING

            case GamePhase.OPERATIONS_MESSAGING:
                self.current_phase = GamePhase.OPERATIONS

            case GamePhase.OPERATIONS:
                self.current_phase = GamePhase.ROUND_EVENTS
                self.current_round += 1

            case GamePhase.CRISIS_MESSAGING:
                self.current_phase = GamePhase.CRISIS

            case GamePhase.CRISIS:
                # This will need to be updated when crises can have
                # multiple phases/trigger follow-up crises.
                assert self.pending_crisis is None
                assert self.post_crisis_phase is not None
                assert self.post_crisis_round is not None
                self.current_phase, self.current_round = (
                    self.post_crisis_phase,
                    self.post_crisis_round,
                )
                self.post_crisis_phase, self.post_crisis_round = None, None

        self.check_instant_mandates()

        if self.trigger_crisis():
            self.post_crisis_phase = self.current_phase
            self.post_crisis_round = self.current_round
            assert self.pending_crisis is not None
            self.current_phase = (
                GamePhase.CRISIS_MESSAGING
                if self.pending_crisis.has_messaging_phase
                else GamePhase.CRISIS
            )

            self.apply_event(
                SystemEvent(
                    description=(
                        "Time has run out and a global crisis has been "
                        f"triggered: {self.pending_crisis.title}"
                    ),
                )
            )

        self._expire_effects()

        self.apply_event(
            StateEvent(
                description=self.describe_state(),
                # It's not really a secret but the players already
                # get the game state so this is redundant. However,
                # it's important to have this in the after-game log.
                secret=True,
            ),
        )

    async def autosave(self) -> None:
        if self.log_dir is None:
            return

        try:
            await anyio.Path(
                os.fspath(self.log_dir / "game_state.json")
            ).write_text(self.model_dump_json(indent=2))
        except OSError:
            logging.getLogger("mad_world").exception(
                "Failed to write save game to log dir"
            )

    def _expire_effects(self) -> None:
        new_effects: list[BaseEffect] = []
        for effect in self.active_effects:
            if not effect.is_expired(self):
                new_effects.append(effect)
                continue

            for e in effect.on_expire(self):
                self.apply_event(e)

        self.active_effects = new_effects

    def recent_events(self) -> list[LoggedEvent[GameEvent]]:
        result: list[LoggedEvent[GameEvent]] = []

        for e in reversed(self.event_log):
            if e.phase == self.last_phase and e.round == self.last_round:
                result.insert(0, e)

            elif result:
                break

        return result

    def escalation_debt(self, player: str) -> int:
        return self.escalation_track.count(PlayerActor(name=player))

    def _claim_mandates(self, *, is_instant: bool) -> None:
        for player_name, player_state in self.players.items():
            completed = [
                m
                for m in player_state.mandates
                if m.is_instant == is_instant and m.is_met(self, player_name)
            ]

            for mandate in completed:
                self._apply_mandate_rewards(mandate, player_name)
                player_state.mandates.remove(mandate)
                player_state.completed_mandates.append(mandate)

    def _apply_mandate_rewards(
        self, mandate: BaseMandate, player_name: str
    ) -> None:
        for event in mandate.reward(self, player_name):
            self.apply_event(event)

    def check_instant_mandates(self) -> None:
        self._claim_mandates(is_instant=True)

    def check_endgame_mandates(self) -> None:
        self._claim_mandates(is_instant=False)

    def determine_victor(self) -> tuple[str | None, GameOverReason]:
        if self.doomsday_clock >= self.rules.max_clock_state:
            return (None, GameOverReason.WORLD_DESTROYED)

        alpha, omega = self.players.values()

        if alpha.gdp > omega.gdp:
            return (alpha.name, GameOverReason.ECONOMIC_VICTORY)

        if alpha.gdp < omega.gdp:
            return (omega.name, GameOverReason.ECONOMIC_VICTORY)

        return (None, GameOverReason.STALEMATE)

    def query_event_log(self) -> EventStream[GameEvent]:
        return EventStream(reversed(self.event_log))


def get_bid_impact(
    game: GameState, player_name: str, bid: int
) -> tuple[int, int, str]:
    """Calculates and validates a bid's impact.

    Determines the influence cost and clock impact of a player's bid.
    """
    action = BiddingAction(bid=bid)

    try:
        action.validate_semantics(game, player_name)
        error = False
    except InvalidActionError:
        error = True

    if error:
        bid = max(game.allowed_bids)
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

    return bid, clock_impact, desc


async def resolve_bidding(
    game: GameState,
    players: list[GamePlayer],
) -> GameState:
    """Resolve the bidding phase of the game."""

    alpha_name, omega_name = game.player_names
    alpha_actor, omega_actor = (
        PlayerActor(name=alpha_name),
        PlayerActor(name=omega_name),
    )
    alpha_action, omega_action = await asyncio.gather(
        players[0].bid(game),
        players[1].bid(game),
    )

    new_game = copy.deepcopy(game)

    alpha_bid, alpha_impact, alpha_desc = get_bid_impact(
        new_game, alpha_name, alpha_action.bid
    )
    omega_bid, omega_impact, omega_desc = get_bid_impact(
        new_game, omega_name, omega_action.bid
    )

    new_game.apply_event(
        BiddingEvent(
            actor=alpha_actor,
            description=alpha_desc,
            clock_delta=0,
            influence_delta={alpha_name: alpha_bid},
            bid=alpha_bid,
        )
    )
    new_game.apply_event(
        BiddingEvent(
            actor=omega_actor,
            description=omega_desc,
            clock_delta=0,
            influence_delta={omega_name: omega_bid},
            bid=omega_bid,
        )
    )

    clock_start = new_game.doomsday_clock
    alpha_rem = abs(alpha_impact)
    alpha_sign = 1 if alpha_impact > 0 else -1 if alpha_impact < 0 else 0

    omega_rem = abs(omega_impact)
    omega_sign = 1 if omega_impact > 0 else -1 if omega_impact < 0 else 0

    def apply_impact(rem: int, sign: int, actor: PlayerActor) -> int:
        if rem <= 0:
            return 0
        new_game.escalate(actor, sign * 1)
        return rem - 1

    while alpha_rem > 0 or omega_rem > 0:
        alpha_rem = apply_impact(alpha_rem, alpha_sign, alpha_actor)
        omega_rem = apply_impact(omega_rem, omega_sign, omega_actor)

    clock_end = new_game.doomsday_clock
    net_change = clock_end - clock_start

    new_game.apply_event(
        SystemEvent(
            description=(
                f"Bidding resolved. Net doomsday clock change: {net_change:+d}"
            ),
            clock_delta=0,
        )
    )

    new_game.advance_phase()

    return new_game


def effects_to_dict(
    names: Sequence[str], effects: Sequence[int]
) -> dict[str, int]:

    result: dict[str, int] = {}
    for p, e in zip_longest(names, effects):
        if e == 0:
            continue

        result |= {p: e}

    return result


def resolve_operation(
    game: GameState,
    player_name: str,
    opponent_name: str,
    operation_name: str,
) -> GameEvent:
    """Validates and executes a single operation.

    Returns the corresponding ActionEvent for the log.
    """
    try:
        game.validate_operation(operation_name, player_name)
    except InvalidActionError as e:
        return ActionEvent(
            actor=PlayerActor(name=player_name),
            description=(
                f"{player_name} attempted a {operation_name} "
                f"operation which was rejected: {e}."
            ),
        )

    op_def = game.allowed_operations[operation_name]
    friendly_gdp_effect = op_def.friendly_gdp_effect
    enemy_gdp_effect = op_def.enemy_gdp_effect
    desc = (
        f"{player_name} has successfully conducted a {operation_name} "
        "operation."
    )

    if (
        game.doomsday_clock >= game.rules.escalation_reward_clock_threshold
        and op_def.clock_effect > 0
    ):
        friendly_gdp_effect += game.rules.escalation_reward_gdp
        enemy_gdp_effect -= game.rules.escalation_reward_gdp
        desc += (
            f" As global tensions run high, this aggressive act yielded "
            f"an additional +{game.rules.escalation_reward_gdp} GDP at "
            f"the expense of the opponent."
        )

    return OperationConductedEvent(
        actor=PlayerActor(name=player_name),
        description=desc,
        clock_delta=op_def.clock_effect,
        shift_blame=(
            (PlayerActor(name=opponent_name), op_def.shift_blame)
            if op_def.shift_blame
            else None
        ),
        gdp_delta=effects_to_dict(
            (player_name, opponent_name),
            (friendly_gdp_effect, enemy_gdp_effect),
        ),
        influence_delta=effects_to_dict(
            (player_name, opponent_name),
            (-op_def.influence_cost, op_def.enemy_influence_effect),
        ),
        world_ending=operation_name == "first-strike",
        operation=operation_name,
    )


async def resolve_operations(
    game: GameState,
    players: list[GamePlayer],
) -> GameState:
    """Asynchronously prompts both players for operations.

    Resolves the operations in alternating turns.
    """

    alpha_name, omega_name = game.player_names
    alpha_action, omega_action = await asyncio.gather(
        players[0].operations(game),
        players[1].operations(game),
    )

    alpha_action.operations.reverse()
    omega_action.operations.reverse()

    new_game = copy.deepcopy(game)
    i = game.rng.choice([0, 1])

    while len(alpha_action.operations) > 0 or len(omega_action.operations) > 0:
        active_name = alpha_name if i == 0 else omega_name
        target_name = omega_name if i == 0 else alpha_name
        active_ops = (
            alpha_action.operations if i == 0 else omega_action.operations
        )

        i = (i + 1) % 2

        if not active_ops:
            continue

        new_game.apply_event(
            resolve_operation(
                new_game,
                active_name,
                target_name,
                active_ops.pop(),
            ),
        )

    new_game.advance_phase()

    return new_game


async def resolve_chat_channel(
    new_game: GameState,
    players: list[GamePlayer],
    alpha_msg: MessagingAction,
    omega_msg: MessagingAction,
) -> None:
    """Handles requests for a direct chat channel.

    Processes mutual or unilateral requests during the messaging phase.
    """
    initiator: PlayerActor | None = None
    alpha_name, omega_name = new_game.player_names

    if alpha_msg.requests_channel() and omega_msg.requests_channel():
        initiator = None
    elif alpha_msg.requests_channel() and omega_msg.accepts_channel():
        initiator = PlayerActor(name=alpha_name)
    elif omega_msg.requests_channel() and alpha_msg.accepts_channel():
        initiator = PlayerActor(name=omega_name)
    elif alpha_msg.requests_channel() and omega_msg.rejects_channel():
        new_game.apply_event(
            ChannelRejectedEvent(
                description=(
                    "A request for a direct communication channel by "
                    f"{alpha_name} was rejected by {omega_name}."
                ),
                initiator=PlayerActor(name=alpha_name),
                actor=PlayerActor(name=omega_name),
            )
        )
        return
    elif omega_msg.requests_channel() and alpha_msg.rejects_channel():
        new_game.apply_event(
            ChannelRejectedEvent(
                description=(
                    "A request for a direct communication channel by "
                    f"{omega_name} was rejected by {alpha_name}."
                ),
                initiator=PlayerActor(name=omega_name),
                actor=PlayerActor(name=alpha_name),
            )
        )
        return
    else:
        return

    initiator_desc = f" by {initiator.name}" if initiator is not None else ""

    new_game.apply_event(
        ChannelOpenedEvent(
            description=(
                "A direct communication channel has been opened"
                f"{initiator_desc}."
            ),
            initiator=initiator,
            channels_opened={
                p: 1
                for p in new_game.player_names
                if initiator is None or initiator.name == p
            },
        )
    )

    max_messages = new_game.rules.max_messages_per_channel
    sender_idx = (
        new_game.rng.choice([0, 1])
        if initiator is None
        else 0
        if initiator.name == alpha_name
        else 1
    )

    sender = players[sender_idx]
    receiver = players[1 - sender_idx]
    last_message: str | None = None

    for i in range(max_messages * 2):
        remaining = (max_messages * 2 + 1 - i) // 2
        chat_action = await sender.chat(new_game, remaining, last_message)
        new_game.log_message(sender.name, receiver.name, chat_action)
        last_message = chat_action.message

        if chat_action.end_channel:
            new_game.apply_event(
                SystemEvent(
                    description=(f"{sender.name} terminated the channel.")
                )
            )
            break

        sender, receiver = receiver, sender

    else:
        new_game.apply_event(
            SystemEvent(
                description=(
                    "The communication channel was closed after "
                    "reaching the limit."
                ),
            )
        )


async def resolve_messaging(
    game: GameState,
    players: list[GamePlayer],
) -> GameState:
    """Asynchronously prompts players for messaging actions.

    Also sets up chat channels if requested by the players.
    """
    alpha_name, omega_name = game.player_names

    async def callback(player: GamePlayer) -> MessagingAction:
        if game.current_phase.is_crisis():
            assert game.pending_crisis is not None
            return await player.crisis_message(game, game.pending_crisis)

        return await player.message(game)

    alpha_msg, omega_msg = await asyncio.gather(
        callback(players[0]), callback(players[1])
    )

    new_game = copy.deepcopy(game)

    new_game.log_message(alpha_name, omega_name, alpha_msg)
    new_game.log_message(omega_name, alpha_name, omega_msg)

    await resolve_chat_channel(new_game, players, alpha_msg, omega_msg)

    new_game.advance_phase()

    return new_game


async def resolve_opening(
    game: GameState,
    players: list[GamePlayer],
) -> GameState:
    """Asynchronously handles the initial opening messages.

    Processes messages from players at the start of the game.
    """

    alpha_name, omega_name = game.player_names
    alpha_msg, omega_msg = await asyncio.gather(
        players[0].initial_message(game),
        players[1].initial_message(game),
    )

    new_game = copy.deepcopy(game)

    new_game.log_message(alpha_name, omega_name, alpha_msg)
    new_game.log_message(omega_name, alpha_name, omega_msg)

    new_game.advance_phase()

    return new_game


async def resolve_round_events(game: GameState) -> GameState:
    new_game = copy.deepcopy(game)
    if len(new_game.event_deck) > 0:
        event_card = new_game.event_deck.draw(new_game.rng)
        events = event_card.run(new_game)
        for e in events:
            new_game.apply_event(e)

        new_game.event_deck.discard(event_card)

    _apply_aggressor_tax(new_game)
    new_game.advance_phase()

    return new_game


def _apply_aggressor_tax(new_game: GameState) -> None:
    if new_game.doomsday_clock < new_game.rules.aggressor_tax_clock_threshold:
        return

    alpha_name, omega_name = new_game.player_names
    alpha_debt = new_game.escalation_debt(alpha_name)
    omega_debt = new_game.escalation_debt(omega_name)

    taxed_players: list[str] = []
    if alpha_debt >= omega_debt:
        taxed_players.append(alpha_name)
    if omega_debt >= alpha_debt:
        taxed_players.append(omega_name)

    for player_name in taxed_players:
        player_state = new_game.players[player_name]
        inf_cost = new_game.rules.aggressor_tax_inf_cost
        gdp_cost = new_game.rules.aggressor_tax_gdp_cost

        if player_state.influence >= inf_cost:
            inf_delta = -inf_cost
            gdp_delta = 0
            desc_impact = f"{inf_cost} Inf"
        else:
            inf_delta = 0
            gdp_delta = -gdp_cost
            desc_impact = f"{gdp_cost} GDP"

        new_game.apply_event(
            SystemEvent(
                description=(
                    f"Aggressor Tax applied to {player_name} for their "
                    f"role in driving global tensions. They lost "
                    f"{desc_impact}."
                ),
                influence_delta={player_name: inf_delta},
                gdp_delta={player_name: gdp_delta},
            )
        )


async def resolve_crisis(
    game: GameState, players: list[GamePlayer]
) -> GameState:
    """Asynchronously prompts players to resolve a global crisis."""

    new_game = copy.deepcopy(game)
    next_crisis, new_game.pending_crisis = new_game.pending_crisis, None
    assert next_crisis is not None
    events = await next_crisis.run(game, players)
    for e in events:
        new_game.apply_event(e)

    if next_crisis.consumable:
        new_game.crisis_deck.trash(next_crisis)
    else:
        new_game.crisis_deck.discard(next_crisis)

    new_game.advance_phase()

    return new_game


async def iterate_game(game: GameState, players: list[GamePlayer]) -> GameState:
    """Advances the main game loop.

    Executes the logic for the current phase.
    """
    if game.current_phase.is_messaging():
        return await resolve_messaging(game, players)

    match game.current_phase:
        case GamePhase.OPENING:
            return await resolve_opening(game, players)

        case GamePhase.ROUND_EVENTS:
            return await resolve_round_events(game)

        case GamePhase.BIDDING:
            return await resolve_bidding(game, players)

        case GamePhase.OPERATIONS:
            return await resolve_operations(game, players)

        case GamePhase.CRISIS:
            return await resolve_crisis(game, players)

    return game


def check_game_over(game: GameState) -> bool:
    return (
        not game.current_phase.is_crisis()
        and game.doomsday_clock >= game.rules.max_clock_state
    ) or game.current_round > game.rules.round_count


def destroy_world(game: GameState) -> GameState:
    new_game = copy.deepcopy(game)
    new_game.escalate(SystemActor(), 50)

    for player in new_game.players.values():
        player.gdp -= 1000

    return new_game


async def game_loop(
    rules: GameRules,
    players: list[GamePlayer],
    log_dir: Path | None = None,
) -> tuple[str | None, GameOverReason, GameState]:
    """Continuously executes the game phases.

    Runs until a game over condition is reached.
    """
    game = GameState.new_game(
        players=[p.name for p in players], rules=rules, log_dir=log_dir
    )

    await asyncio.gather(*(p.start_game(game) for p in players))
    while not check_game_over(game):
        try:
            game = await iterate_game(game, players)
            await game.autosave()
        except WorldDestroyed:
            game = destroy_world(game)
            break

    game.check_endgame_mandates()
    winner, reason = game.determine_victor()
    logger = logging.getLogger("mad_world")
    logger.debug("Victor: %s", winner or "no one")
    logger.debug("Reason: %s", reason.name)

    game.apply_event(
        SystemEvent(
            description=(
                f"Game over! {winner or 'No one'} won due to {reason.name}."
            ),
        ),
    )

    for player in players:
        await player.game_over(game, winner, reason)

    return (winner, reason, game)


def format_results(
    winner: str | None,
    reason: GameOverReason,
    game: GameState,
) -> str:
    """Generates a text summary of the final game state.

    Includes the winner and game over reason.
    """
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


ActionEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
BaseGameEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
BiddingEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
ChannelOpenedEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
ChannelRejectedEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
CrisisResolutionEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
MandateFulfilledEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
MessageEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
OperationConductedEvent.model_rebuild(
    _types_namespace={"BaseEffect": BaseEffect}
)
StateEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
SystemEvent.model_rebuild(_types_namespace={"BaseEffect": BaseEffect})
