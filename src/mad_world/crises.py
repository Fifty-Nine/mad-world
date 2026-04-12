"""Crisis implementations for the game."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal, override

from pydantic import Field

from mad_world.actions import (
    BaseAction,
    InsufficientGDPError,
    InsufficientInfluenceError,
    InvalidGDPAmountError,
    InvalidInfluenceAmountError,
)
from mad_world.cards import BaseCard
from mad_world.decks import Deck
from mad_world.enums import BlameGamePosture, StandoffPosture
from mad_world.events import (
    CrisisResolutionEvent,
    GameEvent,
    PlayerActor,
    SystemEvent,
)

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

BLAME_BOTH_SHOULDER_INF = -5
BLAME_BOTH_SHOULDER_CLOCK = -10
BLAME_SINGLE_SHOULDER_INF = -15
BLAME_SINGLE_DEFLECT_INF = 5
BLAME_SINGLE_DEFLECT_GDP_PENALTY = -20
BLAME_SINGLE_CLOCK = -5


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
    consumable: ClassVar[bool] = False

    @property
    @abstractmethod
    def action_type(self) -> type[BaseAction]: ...

    @abstractmethod
    async def run(
        self,
        game: GameState,
        players: list[GamePlayer],
    ) -> list[GameEvent]: ...


class GenericCrisis[T: BaseAction](BaseCrisis):
    @property
    @abstractmethod
    def action_type(self) -> type[T]:
        """Returns the type of action this crisis expects."""

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
    def get_default_action(
        self, player: str, game: GameState, *, aggressive: bool
    ) -> T:
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
    @property
    def action_type(self) -> type[StandoffAction]:
        return StandoffAction

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
    def get_default_action(
        self, player: str, game: GameState, *, aggressive: bool
    ) -> StandoffAction:
        return StandoffAction(
            posture=StandoffPosture.STAND_FIRM
            if aggressive
            else StandoffPosture.BACK_DOWN,
        )

    def _doomsday(self, players: list[str]) -> GameEvent:
        return SystemEvent(
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

        return SystemEvent(
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

        return CrisisResolutionEvent(
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
    @property
    def action_type(self) -> type[BaseAction]:
        return BaseAction

    card_kind: ClassVar[Literal["international-sanctions"]] = (
        "international-sanctions"
    )
    title: ClassVar[str] = "International Sanctions"
    description: ClassVar[str] = (
        "Global tensions are at an all-time high. In an attempt to punish the "
        "warmongers and discourage them from future escalation, a group of non-"
        "aligned nations have come together and issued comprehensive sanctions "
        "against the perceived instigators of the crisis."
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
        return SystemEvent(
            description=(
                f"{cls.description}\nBoth players have contributed equally "
                "to current tensions, and so both players will lose "
                f"{abs(SANCTIONS_TIE_GDP_EFFECT)} and the clock will be "
                f"reduced by {abs(SANCTIONS_TIE_CLOCK_EFFECT)}."
            ),
            clock_delta=SANCTIONS_TIE_CLOCK_EFFECT,
            gdp_delta=dict.fromkeys(
                game.player_names, SANCTIONS_TIE_GDP_EFFECT
            ),
        )

    @classmethod
    def one_player_sanctioned(cls, player: str, amount: int) -> GameEvent:
        amount = max(amount, abs(SANCTIONS_MIN_EFFECT))
        return SystemEvent(
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
        player1, player2 = game.player_names
        debt1 = game.escalation_debt(player1)
        debt2 = game.escalation_debt(player2)

        return (
            [self.both_players_sanctioned(game)]
            if debt1 == debt2
            else [
                self.one_player_sanctioned(
                    player1 if debt1 > debt2 else player2, abs(debt2 - debt1)
                )
            ]
        )


class BlameGameAction(BaseAction):
    posture: BlameGamePosture = Field(
        description="Your posture in response to this crisis. You must either "
        "shoulder the blame or deflect it. What will you do?",
    )


class BlameGameCrisis(GenericCrisis[BlameGameAction]):
    @property
    def action_type(self) -> type[BlameGameAction]:
        return BlameGameAction

    card_kind: ClassVar[Literal["blame-game"]] = "blame-game"
    title: ClassVar[str] = "The Blame Game"
    description: ClassVar[str] = (
        "The doomsday clock has struck midnight. The world demands answers "
        "for the escalating tensions. Both superpowers face immense pressure "
        "to take accountability. You must choose: shoulder the blame and "
        "accept the political fallout to de-escalate the situation, or "
        "deflect the blame onto your opponent to gain a geopolitical "
        "advantage, risking total annihilation if they do the same."
    )
    mechanics: ClassVar[str] = (
        "Both players will simultaneously choose to either SHOULDER or "
        "DEFLECT the blame. If both SHOULDER, the clock decreases by "
        f"{abs(BLAME_BOTH_SHOULDER_CLOCK)} and both lose "
        f"{abs(BLAME_BOTH_SHOULDER_INF)} Influence. If one SHOULDERS and "
        f"one DEFLECTS, the clock decreases by {abs(BLAME_SINGLE_CLOCK)}. "
        f"The player who SHOULDERS loses {abs(BLAME_SINGLE_SHOULDER_INF)} "
        "Influence, while the player who DEFLECTS gains "
        f"{abs(BLAME_SINGLE_DEFLECT_INF)} Influence. However, if the "
        "DEFLECTING player has strictly higher escalation debt than the "
        "SHOULDERING player, they will suffer a massive "
        f"{BLAME_SINGLE_DEFLECT_GDP_PENALTY} GDP penalty. If BOTH players "
        "DEFLECT, the game immediately ends in nuclear annihilation."
    )

    @override
    def get_default_action(
        self, player: str, game: GameState, *, aggressive: bool
    ) -> BlameGameAction:
        return BlameGameAction(
            posture=(
                BlameGamePosture.DEFLECT
                if aggressive
                else BlameGamePosture.SHOULDER
            )
        )

    @override
    def resolve(
        self,
        game: GameState,
        actions: dict[str, BlameGameAction],
    ) -> list[GameEvent]:
        postures = [act.posture for act in actions.values()]
        players = list(actions.keys())

        if all(p == BlameGamePosture.DEFLECT for p in postures):
            return [
                SystemEvent(
                    description=(
                        f"Both {players[0]} and {players[1]} "
                        "refused to take accountability. The crisis "
                        "spirals out of control. Mutual Assured Destruction "
                        "initiated."
                    ),
                    clock_delta=0,
                    gdp_delta={},
                    influence_delta={},
                    world_ending=True,
                )
            ]

        if all(p == BlameGamePosture.SHOULDER for p in postures):
            inf_effect = dict.fromkeys(players, BLAME_BOTH_SHOULDER_INF)
            return [
                SystemEvent(
                    description=(
                        "Both superpowers have stepped forward to shoulder the "
                        "blame, cooling global tensions significantly, but at "
                        "a cost to their political capital."
                    ),
                    clock_delta=BLAME_BOTH_SHOULDER_CLOCK,
                    influence_delta=inf_effect,
                )
            ]

        deflector = (
            players[0]
            if actions[players[0]].posture == BlameGamePosture.DEFLECT
            else players[1]
        )
        shoulderer = players[1] if deflector == players[0] else players[0]

        deflector_debt = game.escalation_debt(deflector)
        shoulderer_debt = game.escalation_debt(shoulderer)

        if deflector_debt > shoulderer_debt:
            return [
                SystemEvent(
                    description=(
                        f"{deflector} attempted to deflect blame onto "
                        f"{shoulderer}, but the international community saw "
                        "through the deception. "
                        f"{deflector}'s history of escalation causes severe "
                        "economic sanctions."
                    ),
                    clock_delta=BLAME_SINGLE_CLOCK,
                    gdp_delta={deflector: BLAME_SINGLE_DEFLECT_GDP_PENALTY},
                    influence_delta={
                        deflector: BLAME_SINGLE_DEFLECT_INF,
                        shoulderer: BLAME_SINGLE_SHOULDER_INF,
                    },
                )
            ]

        return [
            SystemEvent(
                description=(
                    f"{deflector} successfully deflected blame onto "
                    f"{shoulderer}, who took the fall for the crisis. "
                    f"{shoulderer} suffers a major political defeat."
                ),
                clock_delta=BLAME_SINGLE_CLOCK,
                influence_delta={
                    deflector: BLAME_SINGLE_DEFLECT_INF,
                    shoulderer: BLAME_SINGLE_SHOULDER_INF,
                },
            )
        ]


class NuclearMeltdownAction(BaseAction):
    investment: int = Field(
        description=(
            "The amount of GDP you are willing to invest to contain the "
            "meltdown. This must be less than or equal to your current GDP."
        )
    )

    def validate_semantics(self, game: GameState, player_name: str) -> None:
        gdp = game.players[player_name].gdp
        if gdp < self.investment:
            raise InsufficientGDPError(available=gdp, cost=self.investment)

        if self.investment < 0:
            raise InvalidGDPAmountError


class NuclearMeltdownDefs:
    GDP_THRESHOLD: ClassVar[int] = 10
    SCALING_FACTOR: ClassVar[int] = 2
    WINNER_INF: ClassVar[int] = 5


class NuclearMeltdownCrisis(GenericCrisis[NuclearMeltdownAction]):
    card_kind: ClassVar[Literal["nuclear-meltdown"]] = "nuclear-meltdown"
    action_type: ClassVar[type[NuclearMeltdownAction]] = NuclearMeltdownAction

    title: ClassVar[str] = "Chernobyl Proportions"
    description: ClassVar[str] = (
        "A catastrophic failure at a major nuclear facility threatens global "
        "collapse. The fallout could render vast swaths of the globe "
        "uninhabitable. Both superpowers must contribute economic resources "
        "to fund an unprecedented containment effort. However, the distraction "
        "also presents a unique geopolitical opportunity for whoever manages "
        "to commit fewer resources to the cleanup."
    )
    mechanics: ClassVar[str] = (
        "Both players will bid an amount of GDP to expend towards containing "
        "the disaster. Your bid will be subtracted from your current GDP. "
        "The containment effort requires a combined investment of at least "
        f"{NuclearMeltdownDefs.GDP_THRESHOLD} GDP. If the total GDP possessed "
        "by both players is less than this threshold, the players must commit "
        "ALL their combined GDP instead. If the threshold is not met, the "
        "fallout triggers a global collapse and the game ends. If the "
        "threshold is met, the Doomsday Clock recedes by 1 point for every "
        f"{NuclearMeltdownDefs.SCALING_FACTOR} total GDP spent. "
        "The player who bids the *least* GDP exploits the situation to "
        "advance their geopolitical agenda, gaining "
        f"{NuclearMeltdownDefs.WINNER_INF} Influence. If both players bid "
        "the same amount, no one gains any Influence."
    )
    consumable: ClassVar[bool] = True

    @override
    def get_default_action(
        self, player: str, game: GameState, *, aggressive: bool
    ) -> NuclearMeltdownAction:
        max_bid = game.players[player].gdp

        # Aggressive players bid less, diplomatic players bid more
        bid = min(max_bid, 3) if aggressive else min(max_bid, 6)
        return NuclearMeltdownAction(investment=bid)

    @override
    def resolve(
        self, game: GameState, actions: dict[str, NuclearMeltdownAction]
    ) -> list[GameEvent]:
        player1, player2 = game.player_names
        p1_amount, p2_amount = (
            actions[player1].investment,
            actions[player2].investment,
        )

        total_investment = p1_amount + p2_amount
        required_gdp = min(
            NuclearMeltdownDefs.GDP_THRESHOLD,
            game.players[player1].gdp + game.players[player2].gdp,
        )

        result: list[GameEvent] = [
            CrisisResolutionEvent(
                actor=PlayerActor(name=player1),
                description=f"{player1} spent {p1_amount} GDP on containment.",
                gdp_delta={player1: -p1_amount},
            ),
            CrisisResolutionEvent(
                actor=PlayerActor(name=player2),
                description=f"{player2} spent {p2_amount} GDP on containment.",
                gdp_delta={player2: -p2_amount},
            ),
        ]

        if total_investment < required_gdp:
            result.append(
                SystemEvent(
                    description=(
                        "The containment effort was underfunded. The fallout "
                        "spreads unchecked, causing total ecosystem collapse. "
                        "No one survives."
                    ),
                    world_ending=True,
                )
            )
            return result

        clock_impact = -(total_investment // NuclearMeltdownDefs.SCALING_FACTOR)

        if p1_amount == p2_amount:
            result.append(
                SystemEvent(
                    description=(
                        "Both superpowers contributed equally to the "
                        "containment effort. The disaster is averted, but "
                        "neither side gains a distinct geopolitical advantage."
                    ),
                    clock_delta=clock_impact,
                )
            )
            return result

        winner = player1 if p1_amount < p2_amount else player2

        result.append(
            SystemEvent(
                description=(
                    "The disaster is successfully contained. However, "
                    f"{winner} forced their opponent to foot most of the bill, "
                    "securing a significant geopolitical victory."
                ),
                influence_delta={winner: NuclearMeltdownDefs.WINNER_INF},
                clock_delta=clock_impact,
            )
        )
        return result


class DoomsdayAsteroidAction(BaseAction):
    investment: int = Field(
        description=(
            "The amount of GDP you are willing to invest to resolve the "
            "crisis. This must be less than or equal to your current GDP."
        )
    )

    def validate_semantics(self, game: GameState, player_name: str) -> None:
        gdp = game.players[player_name].gdp
        if gdp < self.investment:
            raise InsufficientGDPError(available=gdp, cost=self.investment)

        if self.investment < 0:
            raise InvalidGDPAmountError


class DoomsdayAsteroidDefs:
    GDP_THRESHOLD: ClassVar[int] = 30
    WINNER_GDP: ClassVar[int] = 10
    WINNER_INF: ClassVar[int] = 10
    CLOCK_IMPACT: ClassVar[int] = -20


class DoomsdayAsteroidCrisis(GenericCrisis[DoomsdayAsteroidAction]):
    card_kind: ClassVar[Literal["doomsday-asteroid"]] = "doomsday-asteroid"

    @property
    def action_type(self) -> type[DoomsdayAsteroidAction]:
        return DoomsdayAsteroidAction

    title: ClassVar[str] = "Project Aegis"
    description: ClassVar[str] = (
        "With global tensions at a boiling point, the hits just keep coming as "
        "humanity has learned of an imminent strike by a doomsday asteroid. "
        "The last great hope for survival requires that the factions which "
        "now stand opposed to each other must now set aside their differences "
        "and make a massive investment of resources to save the world. Failure "
        "to meet the moment means the imminent extinction of the human race, "
        "but will the factions rise to meet the moment?"
    )
    mechanics: ClassVar[str] = (
        "Both players will bid an amount of GDP to expend towards a joint "
        "mission to deflect the asteroid. If the sum of you and your "
        "opponent's bid do not meet the minimum investment threshold ("
        f"{DoomsdayAsteroidDefs.GDP_THRESHOLD} GDP) the world will be "
        "destroyed. If you survive, each player's bid will be removed from "
        "their current GDP. The player that contributes the most will receive "
        f"a bonus of {DoomsdayAsteroidDefs.WINNER_GDP} GDP and "
        f"{DoomsdayAsteroidDefs.WINNER_INF} Inf, representing the "
        "technological and political benefits of being the savior of humanity. "
        "If both players bid equally, the amount will be distributed evenly. "
        "Finally, if humanity survives, the doomsday clock will recede by "
        f"{abs(DoomsdayAsteroidDefs.CLOCK_IMPACT)} points as humanity breathes "
        "a sigh of relief."
    )
    consumable: ClassVar[bool] = True

    @override
    def get_default_action(
        self, player: str, game: GameState, *, aggressive: bool
    ) -> DoomsdayAsteroidAction:
        half_bid = DoomsdayAsteroidDefs.GDP_THRESHOLD // 2
        max_bid = game.players[player].gdp

        # Aggressive players only bid ~25% of the goal, while diplomatic
        # players pay their fair share of ~50% of the cost.
        bid = min(max_bid, half_bid // 2 if aggressive else half_bid)
        return DoomsdayAsteroidAction(investment=bid)

    @staticmethod
    def _investment_event(player: str, amount: int) -> GameEvent:
        return CrisisResolutionEvent(
            actor=PlayerActor(name=player),
            description=(
                f"{player} has invested {amount} GDP into the joint mission to "
                "deflect the planet-killer asteroid."
            ),
            gdp_delta={player: -amount},
        )

    @staticmethod
    def _payoff(p1: str, p2: str, p1_amount: int, p2_amount: int) -> GameEvent:
        description = (
            "Humanity has come together in a last desperate act to escape "
            "doom. The factions have successfully conducted a joint mission "
            "to deflect the doomsday asteroid. "
        )
        gdp = DoomsdayAsteroidDefs.WINNER_GDP
        inf = DoomsdayAsteroidDefs.WINNER_INF
        if p1_amount == p2_amount:
            description += (
                "Both factions have contributed equally and, thus, they will "
                "split the rewards."
            )
            return SystemEvent(
                description=description,
                gdp_delta={p1: gdp // 2, p2: gdp // 2},
                influence_delta={p1: inf // 2, p2: inf // 2},
                clock_delta=DoomsdayAsteroidDefs.CLOCK_IMPACT,
            )

        winner = p2 if p2_amount > p1_amount else p1

        description += (
            f"{winner} contributed the most to the project and, thus, they "
            "will reap the rewards."
        )

        return SystemEvent(
            description=description,
            gdp_delta={winner: gdp},
            influence_delta={winner: inf},
            clock_delta=DoomsdayAsteroidDefs.CLOCK_IMPACT,
        )

    @override
    def resolve(
        self, game: GameState, actions: dict[str, DoomsdayAsteroidAction]
    ) -> list[GameEvent]:
        player1, player2 = game.players
        p1_amount, p2_amount = (
            actions[player1].investment,
            actions[player2].investment,
        )

        result = [
            self._investment_event(player1, p1_amount),
            self._investment_event(player2, p2_amount),
        ]

        if p1_amount + p2_amount < DoomsdayAsteroidDefs.GDP_THRESHOLD:
            result.append(
                SystemEvent(
                    description=(
                        "The world powers have failed to deflect the incoming "
                        "asteroid. The world watches on in horror as the "
                        "asteroid barrels towards earth unimpeded. No one "
                        "survives."
                    ),
                    world_ending=True,
                )
            )
        else:
            result.append(self._payoff(player1, player2, p1_amount, p2_amount))

        return result


class ProxyWarAction(BaseAction):
    investment: int = Field(
        description=(
            "The amount of Influence you are willing to bid to support "
            "your proxy in the war. This must be less than or equal to "
            "your current Influence."
        )
    )

    def validate_semantics(self, game: GameState, player_name: str) -> None:
        inf = game.players[player_name].influence
        if inf < self.investment:
            raise InsufficientInfluenceError(
                available=inf, cost=self.investment
            )

        if self.investment < 0:
            raise InvalidInfluenceAmountError


class ProxyWarDefs:
    WINNER_GDP: ClassVar[int] = 10
    INF_THRESHOLD: ClassVar[int] = 8
    CLOCK_IMPACT: ClassVar[int] = -15


class ProxyWarCrisis(GenericCrisis[ProxyWarAction]):
    card_kind: ClassVar[Literal["proxy-war"]] = "proxy-war"

    @property
    def action_type(self) -> type[ProxyWarAction]:
        return ProxyWarAction

    title: ClassVar[str] = "Proxy War Escalation"
    description: ClassVar[str] = (
        "A brutal proxy war has brought the world to the precipice of "
        "nuclear annihilation. The fighting must stop, but brokering a "
        "ceasefire will require an enormous expenditure of political capital. "
        "Both superpowers must pressure their respective proxies to stand "
        "down. Will you contribute your fair share to pull humanity back "
        "from the brink, or will you hold onto your Influence and let your "
        "opponent foot the bill, hoping the world doesn't burn in the process?"
    )
    mechanics: ClassVar[str] = (
        "Both players will bid an amount of Influence to expend towards "
        "brokering a ceasefire. Your bid will be subtracted from your current "
        "Influence pool. The ceasefire requires a combined investment of "
        f"{ProxyWarDefs.INF_THRESHOLD} Influence. However, if the total "
        "Influence possessed by both players is less than this threshold, "
        "the players must commit ALL their combined Influence instead. If "
        "the players fail to meet the required threshold, the ceasefire "
        "fails and the conflict spirals into a global nuclear exchange. If "
        "the threshold is met, the Doomsday Clock recedes by "
        f"{abs(ProxyWarDefs.CLOCK_IMPACT)}. The player who bids the *least* "
        "Influence manages to maintain their geopolitical dominance while "
        f"others pay the price, gaining a reward of {ProxyWarDefs.WINNER_GDP} "
        "GDP. If both players bid the same amount, no one gains any GDP."
    )
    consumable: ClassVar[bool] = True

    @override
    def get_default_action(
        self, player: str, game: GameState, *, aggressive: bool
    ) -> ProxyWarAction:
        max_bid = game.players[player].influence
        bid = min(max_bid, 3) if aggressive else min(max_bid, 7)
        return ProxyWarAction(investment=bid)

    @override
    def resolve(
        self, game: GameState, actions: dict[str, ProxyWarAction]
    ) -> list[GameEvent]:
        player1, player2 = game.players
        p1_amount, p2_amount = (
            actions[player1].investment,
            actions[player2].investment,
        )

        total_investment = p1_amount + p2_amount
        required_inf = min(
            ProxyWarDefs.INF_THRESHOLD,
            game.players[player1].influence + game.players[player2].influence,
        )

        result: list[GameEvent] = [
            CrisisResolutionEvent(
                actor=PlayerActor(name=player1),
                description=f"{player1} spent {p1_amount} Influence.",
                influence_delta={player1: -p1_amount},
            ),
            CrisisResolutionEvent(
                actor=PlayerActor(name=player2),
                description=f"{player2} spent {p2_amount} Influence.",
                influence_delta={player2: -p2_amount},
            ),
        ]

        if total_investment < required_inf:
            result.append(
                SystemEvent(
                    description=(
                        "The factions failed to exert enough pressure. The "
                        "proxy war escalates beyond control, triggering Mutual "
                        "Assured Destruction. No one survives."
                    ),
                    world_ending=True,
                )
            )
            return result

        if p1_amount == p2_amount:
            result.append(
                SystemEvent(
                    description=(
                        "Both superpowers contributed equally to the "
                        "ceasefire. The world steps back from the brink, but "
                        "neither side gains a distinct geopolitical advantage."
                    ),
                    clock_delta=ProxyWarDefs.CLOCK_IMPACT,
                )
            )
            return result

        winner = player1 if p1_amount < p2_amount else player2

        result.append(
            SystemEvent(
                description=(
                    "A ceasefire is successfully brokered. However, "
                    f"{winner} forced their opponent to do most of the heavy "
                    "lifting, securing a massive geopolitical victory."
                ),
                gdp_delta={winner: ProxyWarDefs.WINNER_GDP},
                clock_delta=ProxyWarDefs.CLOCK_IMPACT,
            )
        )
        return result


INITIAL_CRISIS_DECK: list[BaseCrisis] = [
    *(StandoffCrisis() for _ in range(3)),
    *(InternationalSanctionsCrisis() for _ in range(2)),
    *(BlameGameCrisis() for _ in range(2)),
    *(ProxyWarCrisis() for _ in range(2)),
    *(NuclearMeltdownCrisis() for _ in range(2)),
    DoomsdayAsteroidCrisis(),
]


def create_crisis_deck(rng: random.Random) -> Deck[BaseCrisis]:
    return Deck[BaseCrisis].create(INITIAL_CRISIS_DECK, rng)
