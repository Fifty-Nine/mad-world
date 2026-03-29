"""Crisis implementations for the game."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, Field

from mad_world.actions import BaseAction
from mad_world.enums import StandoffPosture
from mad_world.events import GameEvent, PlayerActor, SystemActor

if TYPE_CHECKING:
    from mad_world.core import GameState
    from mad_world.players import GamePlayer

STANDOFF_WINNER_CLOCK_EFFECT = -5
STANDOFF_WINNER_INF_EFFECT = 5
STANDOFF_LOSER_INF_EFFECT = -5
STANDOFF_LOSER_GDP_EFFECT = -15
STANDOFF_TIE_INF_EFFECT = -10
STANDOFF_TIE_GDP_EFFECT = -5
STANDOFF_TIE_CLOCK_EFFECT = -15


class BaseCrisis(BaseModel, ABC):
    title: str = Field(description="The title of the crisis event.")
    description: str = Field(
        description="A narrative description of the crisis event."
    )
    mechanics: str = Field(
        description="A plain-language explanation of the crisis mechanics."
    )
    additional_prompt: str | None = Field(
        description="Additional instructions for LLM-based players. "
        "May be omitted if description and mechanics are sufficient.",
        default=None,
    )

    @abstractmethod
    async def run(
        self, game: GameState, players: list[GamePlayer]
    ) -> list[GameEvent]: ...


class GenericCrisis[T: BaseAction](BaseCrisis):
    @abstractmethod
    def resolve(
        self, game: GameState, actions: dict[str, T]
    ) -> list[GameEvent]:
        """Determines the outcome of the crisis based on the crisis, the game
        state and the player's responses."""
        ...

    @abstractmethod
    def get_default_action(self, aggressive: bool) -> T:
        """Returns a 'cautious' or 'aggressive' default action for this crisis
        depending on the value of the aggressive flag."""
        ...

    async def run(
        self, game: GameState, players: list[GamePlayer]
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
        "back down or stand firm. What will you do?"
    )


class StandoffCrisis(GenericCrisis[StandoffAction]):
    title: str = "The Brink of Midnight"
    description: str = (
        "Tensions have reached a boiling point. The world stands on the brink "
        "of armageddon, and a crisis has flared up in a forgotten corner of "
        "globe, threatening to spiral out of control. The superpowers have "
        "one last chance to avert disaster: do you press your opponent, hoping "
        "they back down and risk annihilation? Or do you back down and risk "
        "your opponent taking advantage of your weakness?"
    )
    mechanics: str = (
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
            else StandoffPosture.BACK_DOWN
        )

    def _doomsday(self, players: list[str]) -> GameEvent:
        return GameEvent(
            actor=SystemActor(),
            description="Both players have chosen to stand firm and "
            "as a result, the conflict has spiraled out of control, "
            "leading to a total nuclear exchange.",
            gdp_delta=dict.fromkeys(players, -1000),
            clock_delta=50,
        )

    def _tie(self, players: list[str]) -> GameEvent:
        return GameEvent(
            actor=SystemActor(),
            description="Both players have chosen to back down. The world "
            "breathes a sigh of relief as cooler heads have prevailed.",
            gdp_delta=dict.fromkeys(players, STANDOFF_TIE_GDP_EFFECT),
            influence_delta=dict.fromkeys(players, STANDOFF_TIE_INF_EFFECT),
            clock_delta=STANDOFF_TIE_CLOCK_EFFECT,
        )

    def _winner(self, winner: str, loser: str) -> GameEvent:
        return GameEvent(
            actor=PlayerActor(name=winner),
            description=f"{winner} looked death in the eyes and didn't blink. "
            f"As a result, {loser} has suffered significant losses and is "
            "forced into a costly withdrawal.",
            gdp_delta={loser: STANDOFF_LOSER_GDP_EFFECT},
            influence_delta={loser: STANDOFF_LOSER_INF_EFFECT},
            clock_delta=STANDOFF_WINNER_CLOCK_EFFECT,
        )

    @override
    def resolve(
        self, game: GameState, actions: dict[str, StandoffAction]
    ) -> list[GameEvent]:
        postures = [act.posture for act in actions.values()]
        if all(p == StandoffPosture.STAND_FIRM for p in postures):
            return [self._doomsday(list(actions.keys()))]

        if all(p == StandoffPosture.BACK_DOWN for p in postures):
            return [self._tie(list(actions.keys()))]

        winner, loser = actions.keys()
        if actions[winner].posture == StandoffPosture.BACK_DOWN:
            winner, loser = loser, winner

        return [self._winner(winner, loser)]


CRISIS_DECK = [StandoffCrisis()]
