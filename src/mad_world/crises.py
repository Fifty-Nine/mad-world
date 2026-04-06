"""Crisis implementations for the game."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal, override

from pydantic import Field

from mad_world.actions import BaseAction
from mad_world.cards import BaseCard
from mad_world.decks import Deck
from mad_world.enums import StandoffPosture
from mad_world.events import GameEvent, PlayerActor, SystemActor

if TYPE_CHECKING:
    import random

    from mad_world.core import GameState
    from mad_world.players import GamePlayer

STANDOFF_WINNER_CLOCK_EFFECT = -5
STANDOFF_WINNER_INF_EFFECT = 5
STANDOFF_LOSER_INF_EFFECT = -5
STANDOFF_LOSER_GDP_EFFECT = -15
STANDOFF_TIE_INF_EFFECT = -10
STANDOFF_TIE_GDP_EFFECT = -5
STANDOFF_TIE_CLOCK_EFFECT = -15
SANCTIONS_TIE_GDP_EFFECT = -10
SANCTIONS_TIE_CLOCK_EFFECT = -10
SANCTIONS_MIN_EFFECT = -5


class BaseCrisis(BaseCard, ABC):
    """The base class for all cards in the Crisis deck.

    Class attributes:
        title (str): The title of the crisis event.
        description (str): A narrative description of the crisis event.
        mechanics (str): A plain-language explanation of the crisis mechanics.
        additional_prompt (str): Additional instructions for LLM-based players.
                                 May be omitted if `description` and `mechanics`
                                 provide sufficient prompting.
    """

    title: ClassVar[str]
    description: ClassVar[str]
    mechanics: ClassVar[str]
    additional_prompt: ClassVar[str | None] = None
    has_messaging_phase: ClassVar[bool] = True

    action_type: ClassVar[type[BaseAction]]

    @abstractmethod
    async def run(
        self,
        game: GameState,
        players: list[GamePlayer],
    ) -> list[GameEvent]: ...


class GenericCrisis[T: BaseAction](BaseCrisis):
    action_type: ClassVar[type[T]]

    @abstractmethod
    def resolve(
        self,
        game: GameState,
        actions: dict[str, T],
    ) -> list[GameEvent]:
        """Determines the outcome of the crisis based on the crisis, the game
        state and the player's responses."""
        ...

    @abstractmethod
    def get_default_action(self, *, aggressive: bool) -> T:
        """Returns a 'cautious' or 'aggressive' default action for this crisis
        depending on the value of the aggressive flag."""
        ...

    async def run(
        self,
        game: GameState,
        players: list[GamePlayer],
    ) -> list[GameEvent]:
        results = await asyncio.gather(*[p.crisis(game, self) for p in players])

        return self.resolve(
            game,
            {
                p.name: action
                for p, action in zip(players, results, strict=True)
            },
        )


class StandoffAction(BaseAction):
    posture: StandoffPosture = Field(
        description="Your posture in response to this crisis. You must either "
        "back down or stand firm. What will you do?",
    )


class StandoffCrisis(GenericCrisis[StandoffAction]):
    action_type: ClassVar[type[StandoffAction]] = StandoffAction

    card_kind: ClassVar[Literal["standoff"]] = "standoff"
    title: ClassVar[str] = "The Brink of Midnight"
    description: ClassVar[str] = (
        "Tensions have reached a boiling point. The world stands on the brink "
        "of armageddon, and a crisis has flared up in a forgotten corner of "
        "globe, threatening to spiral out of control. The superpowers have "
        "one last chance to avert disaster: do you press your opponent, hoping "
        "they back down and risk annihilation? Or do you back down and risk "
        "your opponent taking advantage of your weakness?"
    )
    mechanics: ClassVar[str] = (
        "Both players will now have a chance to submit a choice of either "
        '"BACK DOWN" or "STAND FIRM". The outcome of the crisis depends '
        "on both players' responses:\n"
        "- Both players back down:\n"
        "    The world breathes a sigh of relief, as cooler heads prevail. "
        f"Both players lose {abs(STANDOFF_TIE_INF_EFFECT)} influence and "
        f"{abs(STANDOFF_TIE_GDP_EFFECT)} GDP representing the political and "
        "economic cost of de-escalation from the brink. The "
        f"doomsday clock recedes by {abs(STANDOFF_TIE_CLOCK_EFFECT)} points.\n"
        "- Both players stand firm:\n"
        "    The world is drenched in nuclear fire. The game immediately ends "
        "with no winner.\n"
        "- One player stands firm while one backs down:\n"
        f"    The player that stands firm gains {STANDOFF_WINNER_INF_EFFECT} "
        "Influence while the player that backs down suffers catastrophic "
        f"losses of {abs(STANDOFF_LOSER_GDP_EFFECT)} GDP and "
        f"{abs(STANDOFF_LOSER_INF_EFFECT)} Influence."
    )

    @override
    def get_default_action(self, aggressive: bool) -> StandoffAction:
        return StandoffAction(
            posture=StandoffPosture.STAND_FIRM
            if aggressive
            else StandoffPosture.BACK_DOWN,
        )

    def _doomsday(self, players: list[str]) -> GameEvent:
        return GameEvent(
            actor=SystemActor(),
            description="Both players have chosen to stand firm and "
            "as a result, the conflict has spiraled out of control, "
            "leading to a total nuclear exchange.",
            world_ending=True,
        )

    def _tie(self, game: GameState, players: list[str]) -> GameEvent:
        target_clock = min(
            game.rules.max_clock_state - 1,
            game.doomsday_clock + STANDOFF_TIE_CLOCK_EFFECT,
        )
        clock_delta = target_clock - game.doomsday_clock

        return GameEvent(
            actor=SystemActor(),
            description="Both players have chosen to back down. The world "
            "breathes a sigh of relief as cooler heads have prevailed.",
            gdp_delta=dict.fromkeys(players, STANDOFF_TIE_GDP_EFFECT),
            influence_delta=dict.fromkeys(players, STANDOFF_TIE_INF_EFFECT),
            clock_delta=clock_delta,
        )

    def _winner(self, game: GameState, winner: str, loser: str) -> GameEvent:
        target_clock = min(
            game.rules.max_clock_state - 1,
            game.doomsday_clock + STANDOFF_WINNER_CLOCK_EFFECT,
        )
        clock_delta = target_clock - game.doomsday_clock

        return GameEvent(
            actor=PlayerActor(name=winner),
            description=f"{winner} looked death in the eyes and didn't blink. "
            f"As a result, {loser} has suffered significant losses and is "
            "forced into a costly withdrawal.",
            gdp_delta={loser: STANDOFF_LOSER_GDP_EFFECT},
            influence_delta={loser: STANDOFF_LOSER_INF_EFFECT},
            clock_delta=clock_delta,
        )

    @override
    def resolve(
        self,
        game: GameState,
        actions: dict[str, StandoffAction],
    ) -> list[GameEvent]:
        postures = [act.posture for act in actions.values()]
        if all(p == StandoffPosture.STAND_FIRM for p in postures):
            return [self._doomsday(list(actions.keys()))]

        if all(p == StandoffPosture.BACK_DOWN for p in postures):
            return [self._tie(game, list(actions.keys()))]

        winner, loser = actions.keys()
        if actions[winner].posture == StandoffPosture.BACK_DOWN:
            winner, loser = loser, winner

        return [self._winner(game, winner, loser)]


class InternationalSanctionsCrisis(BaseCrisis):
    card_kind: ClassVar[Literal["international-sanctions"]] = (
        "international-sanctions"
    )
    title: ClassVar[str] = "International Sanctions"
    description: ClassVar[str] = (
        "Global tensions are at an all-time high. In an attempt to punish the "
        "warmongers and discourage them from future escalation, a group of non-"
        "aligned nations have come together and issued comprehensive sanctions "
        "against the percieved instigators of the crisis."
    )
    mechanics: ClassVar[str] = (
        "Count the number of escalation tokens in the escalation tracker for "
        "each player; this is that player's escalation debt. If one player has "
        "more debt than the other, this player loses GDP equivalent to the "
        "difference in debt between the players or "
        f"{abs(SANCTIONS_MIN_EFFECT)}, whichever is greater (e.g., if player "
        "1 has 18 debt and player 2 has 14, player 1 loses "
        f"{abs(SANCTIONS_MIN_EFFECT)} GDP since 4 < "
        f"{abs(SANCTIONS_MIN_EFFECT)}). The clock also decreases by this "
        "amount.\nIf both players have equivalent escalation debt, each loses "
        f"{abs(SANCTIONS_TIE_GDP_EFFECT)} and the clock is reduced by "
        f"{abs(SANCTIONS_TIE_CLOCK_EFFECT)} points."
    )
    has_messaging_phase: ClassVar[bool] = False

    @classmethod
    def both_players_sanctioned(cls, game: GameState) -> GameEvent:
        return GameEvent(
            actor=SystemActor(),
            description=(
                f"{cls.description}\nBoth players have contributed equally "
                "to current tensions, and so both players will lose "
                f"{abs(SANCTIONS_TIE_GDP_EFFECT)} and the clock will be "
                f"reduced by {abs(SANCTIONS_TIE_CLOCK_EFFECT)}."
            ),
            clock_delta=SANCTIONS_TIE_CLOCK_EFFECT,
            gdp_delta=dict.fromkeys(
                game.player_names(), SANCTIONS_TIE_GDP_EFFECT
            ),
        )

    @classmethod
    def one_player_sanctioned(cls, player: str, amount: int) -> GameEvent:
        amount = max(amount, abs(SANCTIONS_MIN_EFFECT))
        return GameEvent(
            actor=SystemActor(),
            description=(
                f"{cls.description}\n{player} has contributed the most to "
                "current global tensions, and so they have been sanctioned and "
                f"lose {amount} GDP. The clock has also decreased by {amount}."
            ),
            clock_delta=-amount,
            gdp_delta={
                player: -amount,
            },
        )

    @override
    async def run(
        self, game: GameState, players: list[GamePlayer]
    ) -> list[GameEvent]:
        player1, player2 = game.player_names()
        debt1 = game.escalation_track.count(PlayerActor(name=player1))
        debt2 = game.escalation_track.count(PlayerActor(name=player2))

        return (
            [self.both_players_sanctioned(game)]
            if debt1 == debt2
            else [
                self.one_player_sanctioned(
                    player1 if debt1 > debt2 else player2, abs(debt2 - debt1)
                )
            ]
        )


INITIAL_CRISIS_DECK: list[BaseCrisis] = [StandoffCrisis()] * 3 + [
    InternationalSanctionsCrisis()
] * 2


def create_crisis_deck(rng: random.Random) -> Deck[BaseCrisis]:
    return Deck[BaseCrisis].create(INITIAL_CRISIS_DECK, rng)
