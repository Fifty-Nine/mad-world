"""Rules and static definitions for the game."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mad_world.util import cost_or_gain, get_doomsday_bids, increase_or_decrease

if TYPE_CHECKING:
    from collections.abc import Callable


class OperationDefinition(BaseModel):
    """Tracks the definition of a single operation type."""

    name: str = Field(description="The name of the operation.")
    description: str = Field(
        description="A brief description of the operation.",
    )
    influence_cost: int = Field(
        description="The influence cost of the operation.",
    )
    enemy_influence_effect: int = Field(
        description="The effect on the opposing player's influence.",
        default=0,
    )
    clock_effect: int = Field(
        description="The clock impact of the operation.",
        default=0,
    )
    friendly_gdp_effect: int = Field(
        description="The GDP impact on the acting player.",
        default=0,
    )
    enemy_gdp_effect: int = Field(
        description="The GDP impact on the opposing player.",
        default=0,
    )

    @staticmethod
    def format_one(val: int, field: str, desc_fn: Callable[[int], str]) -> str:
        if val == 0:
            return ""

        return f"  {field} {desc_fn(val)}: {abs(val)}\n"

    def format(self, *, verbose: bool, indent: str = "") -> str:
        result = f"{self.name}:\n"

        if verbose:
            result += "  Description:\n" + "    " + self.description + "\n"

        if self.name != "first-strike":
            result += self.format_one(-self.influence_cost, "Inf", cost_or_gain)
            result += self.format_one(
                self.enemy_influence_effect,
                "Opponent Inf",
                increase_or_decrease,
            )
            result += self.format_one(
                self.clock_effect,
                "Clock",
                increase_or_decrease,
            )
            result += self.format_one(
                self.friendly_gdp_effect,
                "GDP",
                increase_or_decrease,
            )
            result += self.format_one(
                self.enemy_gdp_effect,
                "Opponent GDP",
                increase_or_decrease,
            )
        else:
            result += "  Cost: everything\n"
            result += (
                "  Gain: a legacy of ashes, but at "
                "least your opponent doesn't win.\n"
            )

        return textwrap.indent(result, indent)


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
        clock_effect=2,
    ),
    "unilateral-drawdown": OperationDefinition(
        name="unilateral-drawdown",
        description=(
            "Expending massive political capital to walk back from the brink "
            "of nuclear war. Generates zero economic value but has a massive "
            "clock impact."
        ),
        influence_cost=5,
        clock_effect=-9,
    ),
    "stand-down": OperationDefinition(
        name="stand-down",
        description=(
            "Unilaterally concede geopolitical territory or scale back "
            "military readiness. Sacrifices economic standing to generate "
            "immediate diplomatic capital and cool global tensions."
        ),
        influence_cost=-3,
        enemy_influence_effect=-1,
        friendly_gdp_effect=-5,
        clock_effect=-2,
    ),
    "conventional-offensive": OperationDefinition(
        name="conventional-offensive",
        description=(
            "Commit to a widespread, bloody conventional war in a regional "
            "theater. Causes massive economic disruption domestically and "
            "internationally, while pushing the world to the brink of total "
            "war."
        ),
        influence_cost=2,
        friendly_gdp_effect=-4,
        enemy_gdp_effect=-12,
        clock_effect=4,
    ),
    "first-strike": OperationDefinition(
        name="first-strike",
        description=(
            "Launch a first strike against your opponent, immediately "
            "triggering MAD and ending the game."
        ),
        influence_cost=0,
    ),
}


class GameRules(BaseModel):
    """Tracks the rules of a game."""

    initial_gdp: int = Field(default=50, description="Initial GDP value.")
    initial_influence: int = Field(
        default=5,
        description="Initial influence value.",
    )
    initial_clock_state: int = Field(
        default=0,
        description="Initial doomsday clock value.",
    )
    max_clock_state: int = Field(
        default=30,
        description="Maximum doomsday clock value.",
    )
    round_count: int = Field(
        default=10,
        description="Maximum number of rounds.",
    )
    de_escalate_impact: int = Field(
        default=-1,
        description="The clock impact of a de-escalatory bid.",
    )
    allowed_operations: dict[str, OperationDefinition] = Field(
        default=DEFAULT_OPERATIONS,
        description="The set of operations allowed in the game.",
    )
    allowed_bids: list[int] = Field(
        default=[0, 1, 3, 5, 10],
        description="The set of bids allowed in the game.",
    )

    def get_doomsday_bids(
        self,
        clock: int,
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Compute bids that are risky or deadly given the current clock.

        Args:
            clock: The current doomsday clock value.

        Returns:
            A tuple of (risky_bids, deadly_bids).
            risky_bids is a list of (bid, obid) where bid + obid >= limit.
            deadly_bids is a list of bids that unilaterally >= limit.
        """
        return get_doomsday_bids(
            clock,
            self.max_clock_state,
            self.de_escalate_impact,
            self.allowed_bids,
        )


DEFAULT_RULES: GameRules = GameRules()
