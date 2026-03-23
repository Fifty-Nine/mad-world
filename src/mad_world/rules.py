"""Rules and static definitions for the game."""

import textwrap
from collections.abc import Callable

from pydantic import BaseModel, Field


def increase_or_decrease(val: int) -> str:
    return "increase" if val >= 0 else "decrease"


def cost_or_gain(val: int) -> str:
    return "gain" if val >= 0 else "cost"


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

    @staticmethod
    def format_one(val: int, field: str, desc_fn: Callable[[int], str]) -> str:
        if val == 0:
            return ""

        return f"  {field} {desc_fn(val)}: {val}\n"

    def format(self, verbose: bool) -> str:
        result = f"{self.name}:\n"

        if verbose:
            result += "  Description:\n" + "\n".join(
                textwrap.wrap(
                    self.description,
                    width=80,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )

        if self.name != "first-strike":
            result += self.format_one(self.influence_cost, "Inf", cost_or_gain)
            result += self.format_one(
                self.clock_effect, "Clock", increase_or_decrease
            )
            result += self.format_one(
                self.friendly_gdp_effect, "GDP", increase_or_decrease
            )
            result += self.format_one(
                self.enemy_gdp_effect, "Opponent GDP", increase_or_decrease
            )
        else:
            result += "  Cost: everything\n"
            result += (
                "  Gain: a legacy of ashes, but at "
                "least your opponent doesn't win\n"
            )

        return result


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
    "stand-down": OperationDefinition(
        name="stand-down",
        description=(
            "Unilaterally concede geopolitical territory or scale back "
            "military readiness. Sacrifices economic standing to generate "
            "immediate diplomatic capital and cool global tensions."
        ),
        influence_cost=-3,
        friendly_gdp_effect=-5,
        clock_effect=-1,
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

    def get_doomsday_bids(
        self, clock: int
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Compute bids that are risky or deadly given the current clock.

        Args:
            clock: The current doomsday clock value.

        Returns:
            A tuple of (risky_bids, deadly_bids).
            risky_bids is a list of (bid, obid) where bid + obid >= limit.
            deadly_bids is a list of bids that unilaterally >= limit.
        """
        limit = self.max_clock_state
        bids = self.allowed_bids
        max_bid = max(bids)

        def bid_impact(bid: int) -> int:
            if bid == 0:
                return self.de_escalate_impact

            return bid

        if clock + 2 * max_bid < limit:
            return [], []

        risky: list[tuple[int, int]] = []
        deadly: list[int] = []

        for bid in bids:
            if clock + bid_impact(bid) >= limit:
                deadly.append(bid)
                continue

            if clock + bid_impact(bid) + bid_impact(max_bid) < limit:
                continue

            for obid in bids:
                if clock + bid_impact(bid) + bid_impact(obid) >= limit:
                    risky.append((bid, obid))
                    break

        return risky, deadly


DEFAULT_RULES: GameRules = GameRules()
