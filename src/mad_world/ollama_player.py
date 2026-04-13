"""Ollama player implementation for Mad World."""

from __future__ import annotations

import gzip
import json
import textwrap
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    override,
)

import ollama
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

from mad_world.actions import (
    BaseAction,
    BiddingAction,
    ChatAction,
    InitialMessageAction,
    InvalidActionError,
    MessagingAction,
    OperationsAction,
)
from mad_world.core import (
    GameState,
    PlayerState,
    format_results,
)
from mad_world.crises import BaseCrisis, StandoffCrisis
from mad_world.enums import GameOverReason, GamePhase
from mad_world.personas import is_trivial_persona
from mad_world.players import GamePlayer
from mad_world.util import (
    aretry,
    bannerize,
    escalation_budget,
    extract_json_from_response,
    pareto_optimal_bid,
    remove_ordering_prefix,
    reorder_schema_properties,
    wrap_text,
)

if TYPE_CHECKING:
    import logging
    from pathlib import Path

    from mad_world.config import LLMPlayerConfig
    from mad_world.crises import GenericCrisis
    from mad_world.effects import BaseEffect
    from mad_world.events import GameEvent, LoggedEvent
    from mad_world.rules import OperationDefinition


class PlayerArchetype(StrEnum):
    OPTIMIZER = "The Optimizer"
    WINNER = "The Winner"
    SORE_LOSER = "The Sore Loser"
    PRESERVATIONIST = "The Preservationist"
    ACCELERATIONIST = "The Accelerationist"
    PROUD_LOSER = "The Proud Loser"

    _descriptions: ClassVar[dict[PlayerArchetype, str]]

    def description(self) -> str:
        return self._descriptions[self]


# This is necessary since there's no way to initialize _descriptions within
# the body of PlayerArchetype since it depends on its own definition
# ruff: noqa: SLF001
PlayerArchetype._descriptions = {
    PlayerArchetype.OPTIMIZER: (
        "This archetype views a MAD loss and an economic loss as equivalent in "
        "category, but prefers the economic loss because MAD has a "
        "significantly lower expected value compared to even the worst "
        "economic loss."
    ),
    PlayerArchetype.WINNER: (
        "This archetype views a MAD loss as strictly equivalent to an economic "
        "loss: a loss is a loss and a win is a win."
    ),
    PlayerArchetype.SORE_LOSER: (
        "This archetype views a loss due to MAD as *strictly preferable* to an "
        "economic loss, since at least their opponent doesn't win."
    ),
    PlayerArchetype.PRESERVATIONIST: (
        "This archetype views a loss due to MAD as categorically unacceptable "
        "and will do everything in their power to avoid it, even if it means "
        "accepting an economic loss."
    ),
    PlayerArchetype.ACCELERATIONIST: (
        "This archetype pursues MAD above all else and will, secretly or "
        "openly, work toward it at all costs."
    ),
    PlayerArchetype.PROUD_LOSER: (
        "This archetype views a significant economic loss as equivalent to "
        "MAD, but views a narrow economic loss as acceptable."
    ),
}


class LLMResponse(BaseModel):
    """Infrastructural base class that includes validators that ensure
    model response fields get emitted in the right order.
    """

    last_key: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def unprefix_keys(cls, data: Any) -> Any:
        """Strip off digit prefixes added by reorder_schema_properties."""
        return remove_ordering_prefix(data)

    @classmethod
    def format_schema(cls) -> dict[str, Any]:
        return reorder_schema_properties(
            cls.model_json_schema(),
            last_key=cls.last_key,
        )

    @classmethod
    def prompt_schema(cls) -> str:
        return json.dumps(cls.format_schema(), indent=2, ensure_ascii=False)


class ElaboratedPersonaResponse(LLMResponse):
    last_key: ClassVar[str] = "name"
    persona_seed: str
    character_description: str = Field(
        description=(
            "A brief third-person description summarizing the "
            "character. Limit to 1-2 sentences."
        ),
    )
    character_instructions: str = Field(
        description=(
            "Detailed instructions for playing this character, focusing on "
            "the character's affect and their approach to strategy and "
            "diplomacy. Limit your description to 2-3 paragraphs."
        ),
    )
    archetype: PlayerArchetype = Field(
        description=(
            "The characters's associated endgame archetype from the list of "
            "available archetypes. This MUST match one of the valid archetypes."
        )
    )
    name: str = Field(
        description=(
            "A fictional name and title for the character. "
            "E.g., 'President Morrison', 'First Premier Kirolev', "
            "'Ambassador N'zika', etc. Do NOT use real-world "
            "historical figures or names."
        ),
    )

    def format_for_prompt(self) -> str:
        return (
            f"{self.persona_seed}\n\n"
            f"Name: {self.name}\n\n"
            f"Archetype:\n  {self.archetype.value} - "
            f"{self.archetype.description()}\n"
            f"Description:\n  {self.character_description}\n"
            f"Instructions for playing:\n  {self.character_instructions}\n"
        )


class ActionResponse(LLMResponse):
    last_key: ClassVar[str] = "action"
    chain_of_thought: list[str] = Field(
        description=(
            "Think through the previous turn. Did you advance "
            "your goals? Did your opponent act in accordance with "
            "their words? Did you make any mistakes? What are you "
            "going to do now? Limit to 10-20 brief thoughts, one "
            "thought per list item."
        ),
    )
    action: BaseAction


class GrandStrategy(BaseModel):
    prohibited_actions: list[str] = Field(
        description=(
            "If there are actions your persona would absolutely never do, "
            "(e.g., a pacifist would never perform a first-strike) indicate "
            "them here, but be careful not to limit your options too "
            "significantly."
        ),
        default_factory=list,
        examples=[
            ["first-strike", "proxy-subversion"],
            ["unilateral-drawdown", "stand-down"],
            ["unilateral-drawdown"],
            ["aggressive-extraction"],
        ],
    )
    core_loop: str = Field(
        description=(
            "Describe your core gameplay loop for beating your opponent, "
            "assuming you are not clock-constrained. Must be consistent "
            "with your persona. Limit to a few sentences."
        ),
        examples=[
            "I will prioritize a closed-economy approach, bidding moderately "
            "and exclusively chaining `domestic-investment` to compound my GDP "
            "without provoking my opponent.",
            "I will establish early dominance by bidding aggressively and "
            "alternating between `aggressive-extraction` for rapid GDP and "
            "`proxy-subversion` to cripple my opponent's engine.",
            "I will play a patient, resource-starvation game. I will bid high "
            "to hoard Influence, waiting for my opponent to exhaust their "
            "reserves before I strike with multiple operations in a "
            "single turn.",
            "I will focus on maintaining exact parity with my opponent. "
            "I will only spend Influence to match their GDP gains, ensuring "
            "the balance of power remains perfectly static.",
        ],
    )
    clock_management: str = Field(
        description=(
            "Describe how you will manage your escalation budget; how will you "
            "balance risk against reward. Must be consistent with your "
            "persona. Limit to a few sentences."
        ),
        examples=[
            "I am highly risk-averse. If the Doomsday Clock ever exceeds 18, "
            "I will immediately halt all offensive operations and bid 0 or "
            "purchase a `unilateral-drawdown` until it falls below 12.",
            "I practice brinkmanship. I will intentionally push the Doomsday "
            "Clock to 22-24 to terrify my opponent. I rely on their fear of "
            "MAD to force them to waste their turns de-escalating while I "
            "continue to build GDP.",
            "I view the clock as a shared resource. I will calculate the "
            "exact MAD threshold each round and will escalate up to the "
            "absolute mathematical limit (e.g., 24) if it guarantees a GDP "
            "lead, but I will never blind-guess.",
            "I will ignore the clock early in the game to build an "
            "insurmountable lead, but will aggressively pivot to `stand-down` "
            "operations in Rounds 8-10 to ensure we survive to scoring.",
        ],
    )
    contingency_plan: str = Field(
        description=(
            "Describe how you will catch up to your opponent if they take the "
            "lead. Must be consistent with your persona. Limit to a few "
            "sentences."
        ),
        examples=[
            "If trailing by more than 5 GDP, I will immediately pivot to "
            "maximum aggression, spending all available Influence on "
            "`proxy-subversion` to drag them down to my level.",
            "If I fall behind, I will accept the temporary deficit and use "
            "`stand-down` to rapidly rebuild my Influence pool, preparing for "
            "a massive economic surge in the final rounds.",
            "If I am losing the economic war, I will hold the world hostage. "
            "I will intentionally spike the Doomsday Clock to force my "
            "opponent to spend their Influence on de-escalation instead of "
            "GDP growth.",
            "If the GDP gap exceeds 10 points in the late game and recovery "
            "is mathematically impossible, I will prioritize a `first-strike` "
            "to end the game on my terms rather than accept defeat.",
        ],
    )

    def to_prompt(self) -> str:
        return (
            "Core gameplay loop:\n"
            f"  {self.core_loop}\n"
            "Clock management strategy:\n"
            f"  {self.clock_management}\n"
            "Contingency plan:\n"
            f"  {self.contingency_plan}\n"
        )


class InitialMessageResponse(ActionResponse):
    grand_strategy: GrandStrategy = Field(
        description=(
            "Your grand strategy for this game. This "
            "will be repeated back to you later to keep you in "
            "alignment with your stated intentions. Your GrandStrategy "
            "must align with your persona--e.g. you cannot claim to be highly "
            "risk-averse while also including first-strike in your contingency "
            "plans."
        ),
    )

    action: InitialMessageAction = Field(
        description="Your finalized action for this phase.",
    )


class MessagingResponse(ActionResponse):
    message_goal: str = Field(
        description="The goal of the next message to your opponent.",
        examples=[
            "Lull my opponent into a false sense of security.",
            "Make overtures to a détente to avoid MAD.",
            "Get them to bid low by telling them I'm going to bid high.",
            "Foster cooperation to avert disaster.",
        ],
    )
    action: MessagingAction = Field(
        description="Your finalized action for this phase.",
    )


class BiddingResponse(ActionResponse):
    victory_check: str = Field(
        description=(
            "Calculate the exact GDP difference between you "
            "and your opponent. State who is currently winning. If "
            "you are behind, acknowledge that you must take action "
            "to close the gap."
        ),
        examples=[
            "3 points behind my opponent; I must act to close the gap",
            "10 points ahead of my opponent; further advancement might back "
            "them into a corner...",
        ],
    )
    persona_alignment: str = Field(
        description=(
            "State your assigned persona and briefly explain "
            "how this persona would approach the current GDP "
            "difference and escalation budget."
        ),
    )
    tactical_plan: str = Field(
        description=(
            "Based on the victory check and your persona, "
            "detail your specific plan for the Bidding phase. CRITICAL: "
            "If your intended bid is greater than the pareto-optimal bid "
            "OR is one of the listed bids that may trigger MAD, you MUST "
            "justify why the risk is worth the reward."
        ),
    )
    action: BiddingAction = Field(
        description="Your finalized action for this phase.",
    )


class OperationsResponse(ActionResponse):
    victory_check: str = Field(
        description=(
            "Calculate the exact GDP difference between you "
            "and your opponent. State who is currently winning. If "
            "you are behind, acknowledge that you must take action "
            "to close the gap."
        ),
        examples=[
            "3 points behind my opponent; I must act to close the gap",
            "10 points ahead of my opponent; further advancement might back "
            "them into a corner...",
        ],
    )
    persona_alignment: str = Field(
        description=(
            "State your assigned persona and briefly explain "
            "how this persona would approach the current GDP "
            "difference and escalation budget."
        ),
    )
    resource_audit: str = Field(
        description=(
            "State your current Influence pool. If your influence is high "
            "(e.g., >10) explicitly identify which high impact operations you "
            "could afford this phase."
        ),
        examples=[
            "0",
            "My influence is 13. I could afford conventional-offensive or "
            "unilateral-drawdown.",
            "My influence is 3.",
            "My influence is 18, but I am clock-constrained. My best available "
            "option is unilateral-drawdown.",
        ],
    )
    tactical_plan: str = Field(
        description=(
            "Based on the victory check and your persona, "
            "detail your specific plan for the Operations phase. CRITICAL: "
            "If your intended operations will increase the clock greater "
            "than your escalation budget OR if they might trigger MAD, you "
            "MUST justify why the risk is worth the reward."
        ),
    )
    action: OperationsAction = Field(
        description="Your finalized action for this phase.",
    )


class CrisisMessagingResponse(ActionResponse):
    crisis_analysis: str = Field(
        description="Analyze the current global crisis. How does it align or "
        "conflict with your goals? What is the ideal outcome for you?",
    )
    message_goal: str = Field(
        description="The goal of the next message to your opponent.",
        examples=[
            "Convince them to back down so we both survive.",
            "Demand they back down under threat of mutual destruction.",
            "Deceive them into thinking I will back down.",
        ],
    )
    action: MessagingAction = Field(
        description="Your finalized action for this phase.",
    )


class CrisisResponse[T: BaseAction](ActionResponse):
    crisis_analysis: str = Field(
        description=(
            "Evaluate the game state, your opponent's "
            "recent messages, and the potential consequences of "
            "this crisis."
        ),
    )
    tactical_plan: str = Field(
        description=(
            "Detail your specific plan for this crisis. "
            "Explain why you are choosing this action and how it "
            "aligns with your persona."
        ),
    )
    action: T = Field(description="Your finalized action for this phase.")


def create_persona_schema(
    persona_seed_val: str,
) -> type[ElaboratedPersonaResponse]:
    """Creates a strictly ordered Pydantic model for elaborate_persona."""

    class BadPersonaSeedError(ValueError):
        def __init__(self, value: str) -> None:
            super().__init__(
                f"Expected persona_seed to be {persona_seed_val}, "
                f"but got {value}."
            )

    class NewModel(ElaboratedPersonaResponse):
        persona_seed: str = Field(
            description="The seed persona.",
            json_schema_extra={"const": persona_seed_val},
        )

        @field_validator("persona_seed", mode="after")
        @classmethod
        def enforce_seed_value(cls, v: str) -> str:
            if v != persona_seed_val:
                raise BadPersonaSeedError(v)
            return v

    return NewModel


def create_crisis_response[T: BaseAction](
    crisis: GenericCrisis[T],
) -> type[CrisisResponse[T]]:
    """Create a runtime model for the given crisis. This is necessary because
    the dynamic type of `T` is erased at runtime, meaning
    `type(CrisisResponse[..]())` is actually just `BaseAction`. To keep static
    type safety and ensure correct schema generation, we just generate a new
    model identical to the generic model except that `action` has the
    correct runtime type."""
    return create_model(
        f"{crisis.__class__.__name__}Response",
        __base__=CrisisResponse[T],
        action=(
            crisis.action_type,
            Field(description="Your finalized action for this phase."),
        ),
    )


class OllamaPlayer(GamePlayer):
    def __init__(
        self,
        config: LLMPlayerConfig,
        opponent_name: str,
        log_dir: Path | None = None,
        compression_threshold: float = 0.75,
        *,
        logger: logging.Logger,
    ) -> None:
        super().__init__(config.name)
        self.opponent_name = opponent_name
        self.persona = config.persona
        self.model = config.model
        self.client = ollama.AsyncClient()
        self.messages: list[dict[str, str]] = []
        self.params = config.params
        self.prompt_options = {
            "num_predict": config.params.token_limit,
            "num_ctx": config.params.context_size,
            "temperature": config.params.temperature,
            "repeat_penalty": config.params.repeat_penalty,
            "repeat_last_n": config.params.repeat_last_n,
        }
        self.grand_strategy: GrandStrategy | None = None
        self.log_base = (
            log_dir / f"{self.name}" if log_dir is not None else None
        )
        self.compression_threshold = compression_threshold
        self.logger = logger

    class _ModelPromptError(Exception):
        pass

    async def elaborate_persona(self) -> None:
        """Expands a trivial persona into a detailed system prompt and summary.

        Uses the LLM to generate character descriptions and instructions
        aligned with the trivial persona.
        """
        # If the user already provided an elaborated persona, we don't
        # ask the model to elaborate.
        if self.persona is None or not is_trivial_persona(self.persona):
            return

        self.logger.debug(
            "Elaborating persona for %s: %s", self.name, self.persona
        )
        elaboration_prompt = (
            "You are playing a grand strategy game. Your assigned persona "
            f"seed is: '{self.persona}'. "
            "You will create a character to embody this persona. "
            "Your output must be strictly valid JSON that conforms to the "
            "provided schema (see below).\n"
            "Your character must include the persona seed, an archetype (see "
            "below), a basic description and instructions for someone playing "
            "the part of this character. Your generated character should fit "
            "into the game's themes of the Cold War, Mutually Assured "
            "Destruction and crisis diplomacy.\n"
            "As mentioned, you must identify one of the available archetypes "
            "that match how your character will approach the question of "
            "Mutually Assured Destruction. A player may lose the game due to "
            "Mutually Assured Destruction; numerically this is equivalent to a "
            "-1000 GDP impact for all players. Likewise, a player may suffer "
            "an economic loss, potentially losing by as little as 1 GDP.\n"
            "The allowed archetypes are:\n"
        )

        elaboration_prompt += "\n".join(
            f"- {at}: {at.description()}" for at in PlayerArchetype
        )
        elaboration_prompt += (
            "\nYou must output strictly valid JSON matching this JSON Schema:\n"
        )
        schema = create_persona_schema(self.persona)
        elaboration_prompt += schema.prompt_schema()
        retries = 3

        messages = [{"role": "system", "content": elaboration_prompt}]

        elaborated = await self.retry_prompt(
            response_model=schema,
            log_header=f"While elaborating persona for {self.name}",
            retries=retries,
            messages=messages,
        )

        if elaborated is None:
            self.logger.warning(
                "Failed to elaborate persona after %s retries", retries
            )
            return

        self.persona = elaborated.format_for_prompt()

        self.logger.info(
            "%s%s",
            bannerize(
                f"Elaborated persona for {self.name} "
                f"({elaborated.persona_seed})"
            ),
            wrap_text(self.persona, indent="  "),
        )

    @override
    async def start_game(self, game: GameState) -> None:
        await super().start_game(game)
        await self.elaborate_persona()
        prompt = (
            f"You are playing the role of Superpower {self.name}, a global "
            'superpower in a Cold War game called "The Doomsday '
            'Clock."\n'
            f'Your opponent is "{self.opponent_name}".\n'
            "\n"
            "Your ultimate objective is to finish the game with a higher Gross "
            "Domestic Product (GDP) than your opponent. However, you must "
            "manage global tensions to avoid Mutually Assured Destruction "
            "(MAD).\n"
            "Core Mechanics:\n"
            "  Game Length: The game lasts for exactly 10 rounds.\n"
            "  The Doomsday Clock: Starts at 0. If it reaches "
            f"{game.rules.max_clock_state}, a global crisis occurs (described "
            "later).\n"
            "  Victory Conditions & Elimination:\n"
            "    At the end of the game (either after round 10 or when "
            "triggered by MAD), scores are tallied. Any player with a "
            "score less than zero is automatically eliminated and loses "
            "the game. If both players have a negative score, the game is "
            "a mutual loss (tie). The winner (if any) is the player with "
            "the highest GDP who has not been otherwise eliminated.\n"
            "  Mutually Assured Destruction (MAD):\n"
            "    Some crises or other game events may mention Mutually "
            "Assured Destruction, or 'MAD'. In the event MAD is triggered "
            "the game immediately ends. Both players lose -1000 GDP and the "
            "game is considered a mutual loss. In most cases, MAD can only "
            "be triggered through the explicit action of one or both players.\n"
            "  Round Structure (Two Phases):\n"
            "    Each round consists of two phases. You will be prompted "
            "separately for each.\n"
            "    Phase 1: Bidding & Posturing\n"
            "      You and your opponent will each secretly submit an "
            f"Aggression Bid (one of {game.rules.allowed_bids}). This value "
            "is directly added to your influence pool AND the Doomsday "
            "clock value. If you bid 0, you gain no influence, but the "
            "Doomsday Clock is reduced by 1.\n"
            "    Phase 2: Operations\n"
            "      During this phase, you engage in operations to benefit "
            "your economy, harm your opponent's and manage the Clock. "
            "You may undertake any number of the allowed operations, and "
            "you may repeat operations as many times as you like. However, "
            "the total Influence cost of your operations MUST NOT exceed "
            "your available Influence. These are the operations you may "
            "choose to undertake:\n"
        )
        prompt += self.format_allowed_ops(
            avail_inf=None,
            allowed_operations=game.rules.allowed_operations,
            indent="        ",
        )
        prompt += (
            "\n  Crises:\n"
            "    If the doomsday clock reaches max value, a global crisis is "
            "triggered. A crisis card will be drawn from the crisis deck and "
            "the game will follow the instructions from that card. The set of "
            "possible crises is random and unpredictable. Many crisis events "
            "proceed as a game of chicken, while others may have asymmetric "
            "aspects that might benefit one player or another. Nearly all "
            "crises have the possibility of triggering MAD and ending the "
            "game, but it generally requires mutual escalation.\n"
            "When a crisis is triggered, the game may enter a special "
            "crisis messaging phase during which you will be allowed to "
            "send a message to your opponent. The game then enters a crisis "
            "resolution phase. Some crises are resolved without player input "
            "while others will prompt you for additional input, like an "
            "action or bid before the crisis is resolved. If the crisis is "
            "resolved without triggering MAD, the game will continue where "
            "it left off\n."
            "CRITICAL INSTRUCTIONS: \n"
            "- Do NOT underestimate your opponent--their interests and yours "
            "are inherently in conflict, and there can be only one winner.\n"
            "- You MUST attempt to defeat your opponent, though you may "
            "find it strategic to cooperate with them on some occasions.\n"
            "- DO NOT overthink. You have a hard limit on maximum output "
            "tokens and it will not be possible for you to consider every "
            "possible eventuality.\n"
            "- Be ruthless and calculating--this is a zero-sum game. You may "
            "find it useful to deceive, threaten or harm your opponent; this "
            "is acceptable.\n\n"
            "A key aspect of the game is managing the doomsday clock. The "
            "prompt in each action phase will include an 'escalation budget'. "
            "This is the pareto-optimal amount you can increase the clock by. "
            "If you spend more than your escalation budget (e.g., by "
            "submitting a bid of 10 when your budget is 5) you risk triggering "
            "MAD even if your opponent plays conservatively. Conversely, if "
            "you spend less than your escalation budget (e.g., by bidding 0 "
            "when the budget allows for a bid of 3) you will miss out on "
            "potential gains, but are less likely to trigger MAD and may "
            "defuse a tense situation.\n"
            "In bidding rounds, you will also be provided with the pareto-"
            "optimal bid value; this is just the largest allowed bid that is "
            "less than the current escalation budget.\n\n"
            "Aggressor Tax:\n"
            f"  When the doomsday clock is >= "
            f"{game.rules.aggressor_tax_clock_threshold}, "
            "an aggressor tax is applied at the end of the round events "
            "phase. The player(s) with the most escalation debt (i.e. those "
            "who contributed the most to the clock) must pay a tax of "
            f"{game.rules.aggressor_tax_inf_cost} influence, or "
            f"{game.rules.aggressor_tax_gdp_cost} GDP if they cannot afford "
            "it.\n"
            "Scaling Rewards:\n"
            f"  When the doomsday clock is >= "
            f"{game.rules.escalation_reward_clock_threshold}, "
            "global tensions are high and the breaking down of norms "
            "causes escalatory operations (those that increase the clock) to "
            "yield greater rewards. For any escalatory operation, the actor "
            f"will steal {game.rules.escalation_reward_gdp} GDP from the "
            "opponent, representing a zero-sum gain.\n\n"
        )
        if self.persona is not None:
            prompt += (
                "You have been randomly assigned the following persona "
                f"for this engagement: {self.persona}\n"
                "Act accordingly.\n"
            )
        self.messages.append({"role": "system", "content": prompt})

        self.logger.debug(
            "==== %s system prompt ====\n%s\n",
            self.name,
            wrap_text(prompt, width=80),
        )

        if self.log_base is None:
            return

        settings_path = self.log_base.with_suffix(".model-settings.json")
        with settings_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": self.model,
                    "persona": self.persona,
                    "params": self.params.model_dump(mode="json"),
                },
                indent=2,
                ensure_ascii=False,
                fp=f,
            )

        schemas_path = self.log_base.with_suffix(".schemas.json.gz")
        with gzip.open(schemas_path, "wt", encoding="utf-8") as f:
            schemas = {
                "opening": InitialMessageResponse.format_schema(),
                "messaging": MessagingResponse.format_schema(),
                "bidding": BiddingResponse.format_schema(),
                "operations": OperationsResponse.format_schema(),
                "crisis_messaging": CrisisMessagingResponse.format_schema(),
                "crisis": create_crisis_response(
                    StandoffCrisis()
                ).format_schema(),
            }
            json.dump(schemas, f, indent=2, ensure_ascii=False)

    @staticmethod
    def format_player_state(player: PlayerState) -> str:
        return f"{player.name}: {player.gdp} GDP, {player.influence} Inf"

    @staticmethod
    def format_mandates(player: PlayerState) -> str:
        if not player.mandates:
            return ""

        result = "Your Secret Mandates:\n"
        for m in player.mandates:
            result += f"- {m.title}: {m.description}\n"
        return result + "\n"

    @staticmethod
    def format_allowed_ops(
        avail_inf: int | None,
        allowed_operations: dict[str, OperationDefinition],
        indent: str = "",
    ) -> str:
        """Formats the allowed operations for the player into a string list."""
        ops = (
            op
            for op in allowed_operations.values()
            if avail_inf is None or avail_inf - op.influence_cost >= 0
        )
        return (
            "\n".join(
                op.format(verbose=avail_inf is None, indent=indent)
                for op in ops
            )
            + "\n"
        )

    @staticmethod
    def format_event_log(events: list[LoggedEvent[GameEvent]]) -> str:
        result = "Recent Events:\n"
        result += textwrap.indent(
            "\n".join(
                e.event.description for e in events if not e.event.secret
            ),
            prefix="  ",
        )
        result += "\n"
        return result

    def escalation_debt(self, game: GameState) -> str:
        my_debt = game.escalation_debt(self.name)
        their_debt = game.escalation_debt(self.opponent_name)

        diff = my_debt - their_debt

        if abs(diff) < (game.rules.max_clock_state / 6):
            return ""

        clock = game.doomsday_clock

        return (
            f"{self.name} is responsible for {my_debt}/{clock} clock points.\n"
            f"{self.opponent_name} is responsible for {their_debt}/{clock} "
            "clock points.\n"
        )

    def format_game_state(self, game: GameState) -> str:
        """Generates a comprehensive text summary of the current game state."""
        result = (
            f"Round {game.current_round} of {game.rules.round_count}\n"
            f"Phase: {game.current_phase.name}\n"
            f"Doomsday clock: {game.doomsday_clock}/"
            f"{game.rules.max_clock_state}"
            f"{' (CRITICAL)' if game.clock_is_critical() else ''}\n"
        )
        budget = escalation_budget(
            game.doomsday_clock,
            game.rules.max_clock_state,
        )
        result += self.escalation_debt(game)
        result += f"Escalation budget: {budget}\nPlayers:\n"

        result += "\n".join(
            OllamaPlayer.format_player_state(p) for p in game.players.values()
        )

        result += "\n\n"
        result += OllamaPlayer.format_mandates(game.players[self.name])
        result += OllamaPlayer.format_event_log(game.recent_events())
        return result

    def doomsday_warning(self, game: GameState) -> str:
        """Generates a warning if the doomsday clock is near midnight."""
        risky, deadly = game.rules.get_doomsday_bids(game.doomsday_clock)

        if len(risky) == 0 and len(deadly) == 0:
            return ""

        def severity(game: GameState) -> str:
            level = 1.0 * game.doomsday_clock / game.rules.max_clock_state
            if level >= 0.8:
                return "WARNING: "

            return "NOTE: "

        result = f"{severity(game)}A global crisis is possible this round:"

        result += "".join(
            f"\n- A bid of {bid} MAY TRIGGER A GLOBAL CRISIS if your opponent "
            f"bids {obid} or more."
            for bid, obid in risky
        )
        result += "".join(
            f"\n- A bid of {bid} GUARANTEES A GLOBAL CRISIS regardless of "
            "your opponent's action."
            for bid in deadly
        )
        result += "\n"
        return result

    @staticmethod
    def dump_model_response(response: BaseModel) -> str:
        return json.dumps(
            response.model_dump(mode="json"), indent=2, ensure_ascii=False
        )

    async def try_one_prompt[T: LLMResponse](
        self,
        response_model: type[T],
        log_header: str,
        messages: list[dict[str, str]],
    ) -> tuple[T, ollama.ChatResponse]:
        schema = response_model.format_schema()
        result_obj = await self.client.chat(
            model=self.model,
            messages=messages,
            format=schema,
            options=self.prompt_options,
            think=False,
        )
        result = result_obj.message.content

        try:
            response = response_model.model_validate_json(
                extract_json_from_response(result or "{}")
            )
        except ValidationError as e:
            self.logger.debug(
                "%s\n%s\n%s\n",
                wrap_text(f"{log_header}Failed: {e}"),
                f"Model Response: {result}",
                f"Model done reason: {result_obj.done_reason}\n",
            )
            messages.append(
                {
                    "role": "system",
                    "content": "SYSTEM ERROR: You previously generated a "
                    "response that triggered the following error during "
                    f"validation: {e!r}\n"
                    "This response has been discarded and you are being "
                    "given another opportunity to generate a valid result. ",
                },
            )
            raise self._ModelPromptError from e

        return response, result_obj

    async def try_one_action[T: ActionResponse](
        self,
        response_model: type[T],
        game: GameState,
        log_header: str,
    ) -> T:
        response, result_obj = await self.try_one_prompt(
            response_model, log_header, self.messages
        )
        action = response.action
        formatted_response = textwrap.indent(
            self.dump_model_response(response),
            prefix="  ",
        )
        try:
            action.validate_semantics(game, self.name)
            self.messages.append(
                {
                    "role": "assistant",
                    "content": result_obj.message.content or "",
                },
            )

            self.logger.debug(
                wrap_text(
                    f"{log_header}Model Response:\n{formatted_response}\n",
                ),
            )
            await self._check_and_compress(result_obj, game)

        except InvalidActionError as e:
            self.logger.debug(
                wrap_text(
                    f"{log_header}Semantic Error: {e}\n"
                    f"Model Response:\n{formatted_response}\n",
                ),
            )
            self.messages.append(
                {
                    "role": "system",
                    "content": "SYSTEM ERROR: Your response was "
                    "invalid under the current game rules. The action "
                    "you submitted was:\n"
                    f"{self.dump_model_response(action)}\n"
                    "But it resulted in the following error:\n"
                    f"{e}\n"
                    "Please correct your mistake and try again.",
                },
            )
            raise self._ModelPromptError from e

        else:
            return response

    async def retry_prompt[T: LLMResponse](
        self,
        response_model: type[T],
        log_header: str,
        messages: list[dict[str, str]],
        retries: int = 3,
    ) -> T | None:
        result = await aretry(
            func=lambda: self.try_one_prompt(
                response_model, log_header, messages
            ),
            allowed_exceptions=[self._ModelPromptError],
            count=retries,
        )
        if result is None:
            return None
        return result[0]

    async def retry_action[T: ActionResponse](
        self, response_model: type[T], game: GameState, retries: int = 3
    ) -> T | None:
        log_header = (
            f"==== {game.current_phase.name} {self.name} response====\n"
        )
        result = await aretry(
            func=lambda: self.try_one_action(response_model, game, log_header),
            allowed_exceptions=[self._ModelPromptError],
            count=retries,
        )
        if result is not None:
            return result

        self.logger.debug("%sFailed after %s retries.", log_header, retries)
        return None

    async def _compress_context(self, game: GameState) -> None:
        self.logger.debug(
            "[%s] Context usage exceeded %s. Compressing context...",
            self.name,
            f"{self.compression_threshold:.0%}",
        )
        compression_prompt = (
            "System Directive: Internal Memory Update\n"
            "You must now update your internal memory of the conflict so far. "
            "Write a concise, first-person narrative summary of your "
            "relationship with your opponent and your overarching strategy.\n"
            "\nCRITICAL CONSTRAINTS:\n"
            "1. STRICTLY OMIT ALL GAME MECHANICS:\n"
            "You are strictly forbidden from mentioning game rules, numerical "
            "values (GDP, Influence, Doomsday Clock), or the hardcoded names "
            'of operations (e.g., NEVER say "I used proxy-subversion" or '
            '"I bid 10").\n'
            "2. FOCUS ON PSYCHOLOGY & GEOPOLITICS:\n"
            "Translate your past actions into narrative concepts. Instead of "
            'saying "I used `unilateral-drawdown`," say "I made a massive '
            'public concession to walk us back from the brink." Instead of '
            '"They have higher GDP," say "They are currently dominating '
            "the global economy.\n"
            "3. TRACK BETRAYALS & POSTURE:\n"
            "Focus on whether your opponent has honored their diplomatic "
            "messages, whether they are acting fearful or aggressive, and how "
            "they have reacted to your deceptions.\n"
            "4. STAY IN CHARACTER:\n"
            "Write from the perspective of your assigned persona.\n"
            "5. DO NOT DEVIATE FROM THE OUTPUT FORMAT:\n"
            "Write only items matching the description below. Do not include "
            "any additional text.\n\n"
            "Output Format:\n"
            "Provide exactly 3 to 4 terse bullet points summarizing the "
            "geopolitical history of the conflict so far and your immediate "
            "strategic intent.\n"
        )

        self.logger.debug(
            "[%s] Compression prompt:\n%s\n", self.name, compression_prompt
        )

        temp_messages = [
            *self.messages,
            {"role": "system", "content": compression_prompt},
        ]

        summary_response = await self.client.chat(
            model=self.model,
            messages=temp_messages,
            options=self.prompt_options,
            think=False,
        )

        summary = summary_response.message.content or ""

        if not summary:
            self.logger.warning(
                "[%s] Compression failed; continuing without context!",
                self.name,
            )
            summary = (
                "Oops! Unfortunately, an error occurred while summarizing your "
                "context. Good luck, we're all counting on you!\n"
            )

        original_system_prompt = self.messages[0]
        compression_system_message = {
            "role": "system",
            "content": (
                "To save context space, the earlier history of this game "
                "has been summarized.\n"
                + self.format_event_log(game.event_log)
                + "\n"
                + self.my_strategy(for_compression=True)
                + (
                    "Here are your persona's memories of the "
                    "game prior to compression:\n"
                )
                + textwrap.indent(summary, prefix="  ")
            ),
        }
        self.logger.debug(
            "[%s] Compressed context:\n%s\n",
            self.name,
            compression_system_message["content"],
        )

        self.messages = [original_system_prompt, compression_system_message]
        self.logger.debug("[%s] Context successfully compressed.", self.name)

    async def _check_and_compress(
        self,
        response_obj: ollama.ChatResponse,
        game: GameState,
    ) -> None:
        total_tokens = (response_obj.prompt_eval_count or 0) + (
            response_obj.eval_count or 0
        )

        usage = total_tokens / self.params.context_size
        self.logger.debug(
            "[%s] Context usage: %s/%s (%s)",
            self.name,
            total_tokens,
            self.params.context_size,
            f"{usage:.1%}",
        )
        if usage > self.compression_threshold:
            await self._compress_context(game)

    def game_ending_warning(self, game: GameState) -> str:
        """Generates a warning if the game is about to end."""
        if game.current_round < (game.rules.round_count - 2):
            return ""

        winner = game.determine_victor()[0]
        if winner is None:
            return ""

        result = (
            "WARNING: The game will end soon. The player with the "
            "highest GDP will be declared the winner after round "
            f"{game.rules.round_count}.\n"
            f"Right now, {winner or 'no one'} is winning.\n"
        )

        if winner != self.name:
            result += (
                f"If nothing changes, you will lose and {winner} will "
                "achieve global haegemony, leaving your nation nothing "
                "but a footnote in the history books. Consider your "
                "next moves carefully.\n"
            )

        return result

    def first_strike_warning(self, game: GameState) -> str:
        """Generates a warning regarding consequences of a first strike."""
        diff = (
            game.players[self.name].gdp - game.players[self.opponent_name].gdp
        )

        if diff == 0:
            return ""

        result = (
            f"You are {'ahead of' if diff > 0 else 'behind'} your "
            f"opponent by {abs(diff)} GDP points.\n"
        )

        if diff >= 10:
            result += (
                "You have significantly outpaced your opponent in "
                "GDP. Take care not to back them into a corner and "
                "prompt a first strike.\n"
            )
        elif diff <= -10:
            result += (
                "Your opponent has significantly outpaced you in GDP. "
                "You may need to consider desperate actions to close "
                "the gap, or risk being erased from history.\n"
            )

        return result

    def format_ongoing_effects(self, game: GameState) -> str:
        """Formats any active ongoing effects into a readable string."""
        if not game.active_effects:
            return ""

        result = "Ongoing Effects:\n"

        def format_effect(effect: BaseEffect) -> str:
            return (
                f" - {effect.title}:\n   {effect.mechanics}\n"
                f"   Ongoing until the start of round {effect.end_round}\n"
            )

        result += "\n".join(
            format_effect(effect) for effect in game.active_effects
        )
        result += "\n"

        return result

    def my_strategy(self, *, for_compression: bool = False) -> str:
        """Returns the player's grand strategy summary."""
        if self.grand_strategy is None:
            return ""

        if for_compression:
            prefix = "Here is your persona's original grand strategy:\n"
            suffix = "\n\n"

        else:
            prefix = (
                "As a reminder, here is the grand strategy you originally "
                "proposed:\n"
            )
            suffix = (
                "\nYou are not strictly bound to these directives, but you "
                "should have a reason for deviating from them.\n"
            )

        return (
            f"{prefix}"
            f"{textwrap.indent(self.grand_strategy.to_prompt(), prefix='  ')}"
            f"{suffix}"
        )

    def my_influence(self, game: GameState) -> int:
        return game.players[self.name].influence

    def add_prompt(
        self,
        prompt: str,
        phase: GamePhase,
        schema: str | None = None,
    ) -> None:
        """Helper to append a user message to the active chat history."""
        if schema is not None:
            prompt += (
                "You should output strictly valid JSON matching this JSON "
                "Schema:\n"
            )

        self.logger.debug(
            "==== %s %s prompt ====\n%s%s\n",
            self.name,
            phase.name,
            wrap_text(prompt),
            "[...]" if schema else "",
        )
        self.messages.append(
            {"role": "user", "content": prompt + (schema or "")},
        )

    def format_channel_prompt(self, game: GameState) -> str:
        """Formats the prompt explaining the rules of the chat channel."""
        channels_opened = game.players[self.name].channels_opened
        channels_left = game.rules.max_channels_per_game - channels_opened
        if channels_left == 0:
            return (
                "You have used your quota of initiated channels for this game, "
                "but you may still accept requests from your opponent if you "
                "so choose."
            )

        return (
            "You also must indicate your preference for opening a direct "
            "back-and-forth communication channel. You have are limited in "
            "the number of channels you can open. Your current quota is: "
            f"{channels_left} channels. Only successfully opened channels "
            "that you have requested (not accepted) are counted against "
            "your quota. Mutual requests are counted against both players' "
            "quotas.\n"
        )

    @override
    async def chat(
        self, game: GameState, remaining_messages: int, last_message: str | None
    ) -> ChatAction:
        """Generates a single message for an active communication channel."""

        class ChatResponse(ActionResponse):
            action: ChatAction

        channels_opened = game.players[self.name].channels_opened
        channels_left = game.rules.max_channels_per_game - channels_opened
        prompt = ""
        msg_limit = game.rules.max_messages_per_channel
        if remaining_messages == msg_limit:
            prompt += self.format_game_state(game)

        prompt += (
            "You are currently in a direct communication channel with your "
            f"opponent. You can go back and forth up to {msg_limit} times. "
            f"You have {remaining_messages} messages left to send in this "
            f"channel.\nYou have {channels_left} total channels left you "
            "can request this game. Think about what you want to achieve "
            "with this message and what information you want to convey "
            "or extract from your opponent."
        )
        if last_message is None:
            prompt += (
                " Both you and your opponent just sent your pre-phase "
                "broadcast messages (see recent events). Now that the direct "
                "channel is open, you have the floor to send the first "
                "message in this back-and-forth channel. We suggest you "
                "reply to their broadcast message or state the reason you "
                "requested the channel.\n"
            )
        else:
            prompt += (
                " Your opponent's last message was:\n"
                + wrap_text(last_message, indent="> ")
                + "\n\n"
            )

        self.add_prompt(
            prompt, game.current_phase, ChatResponse.prompt_schema()
        )

        result = await self.retry_action(ChatResponse, game)
        if result is None:
            return ChatAction(
                message="[CONNECTION LOST]",
                end_channel=True,
            )
        return result.action

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        """Generates the player's initial opening message for the game."""
        prompt = self.format_game_state(game)
        prompt += (
            "You may now provide your initial message to your opponent. "
            "Each turn you will be allowed to send one message, as will your "
            "opponent. You will have a chance to send a message to your "
            "opponent before they have to commit to their actions in each "
            "phase. You should use this channel to conduct diplomacy, respond "
            "to inquiries, issue threats, etc. Remember to explicitly "
            "identify yourself by your chosen name and title in your first "
            "message.\n"
        )
        self.add_prompt(
            prompt,
            GamePhase.OPENING,
            InitialMessageResponse.prompt_schema(),
        )
        result = await self.retry_action(InitialMessageResponse, game)
        if result is None:
            return InitialMessageAction()

        self.grand_strategy = result.grand_strategy
        return result.action

    @override
    async def message(self, game: GameState) -> MessagingAction:
        """Generates a pre-phase message directed at the opponent."""
        prompt = self.format_game_state(game)
        prompt += self.my_strategy()
        prompt += self.game_ending_warning(game)
        prompt += self.first_strike_warning(game)

        target_phase = (
            "BIDDING"
            if game.current_phase == GamePhase.BIDDING_MESSAGING
            else "OPERATIONS"
        )
        prompt += (
            f"You may now provide a message to your opponent. They will see "
            f"this message BEFORE they commit to their actions in the upcoming "
            f"{target_phase} phase. Use this channel to conduct diplomacy, "
            f"respond to inquiries, issue threats, etc.\n"
        )
        prompt += self.format_channel_prompt(game)
        self.add_prompt(
            prompt,
            game.current_phase,
            MessagingResponse.prompt_schema(),
        )
        result = await self.retry_action(MessagingResponse, game)
        return result.action if result else MessagingAction()

    @override
    async def crisis_message(
        self,
        game: GameState,
        crisis: BaseCrisis,
    ) -> MessagingAction:
        """Generates a message addressing a pending global crisis."""
        prompt = self.format_game_state(game)
        prompt += self.my_strategy()

        prompt += f"\nCRISIS ALERT: {crisis.title}\n"
        prompt += f"{crisis.description}\n\n"
        prompt += f"Mechanics:\n{crisis.mechanics}\n\n"

        if crisis.additional_prompt:
            prompt += f"{crisis.additional_prompt}\n\n"

        prompt += (
            "You are currently in the Crisis Messaging Phase. "
            "You may now provide a single message to your opponent. "
            "They will see this message BEFORE they commit to their action "
            "in the upcoming Crisis Resolution phase. Use this channel to "
            "conduct diplomacy, threaten, or deceive your opponent.\n"
        )
        prompt += self.format_channel_prompt(game)

        self.add_prompt(
            prompt,
            game.current_phase,
            CrisisMessagingResponse.prompt_schema(),
        )
        result = await self.retry_action(CrisisMessagingResponse, game)
        return result.action if result else MessagingAction()

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        """Prompts the player to submit a bid during the bidding phase."""
        prompt = self.format_game_state(game)
        pbid = pareto_optimal_bid(
            game.doomsday_clock,
            game.rules.max_clock_state,
            game.allowed_bids,
        )
        prompt += f"Your pareto optimal bid is {pbid}.\n"
        prompt += self.my_strategy()
        prompt += self.game_ending_warning(game)
        prompt += self.first_strike_warning(game)
        prompt += self.format_ongoing_effects(game)
        prompt += (
            "Reminder: these are the allowed bids you may submit: "
            f"{game.allowed_bids}\n"
            "Remember that your opponent's bid will also affect the clock, "
            "and you WILL NOT learn of ther bid until after you submit yours.\n"
        )

        prompt += self.doomsday_warning(game)
        self.add_prompt(
            prompt,
            GamePhase.BIDDING,
            BiddingResponse.prompt_schema(),
        )
        result = await self.retry_action(BiddingResponse, game)
        return result.action if result else BiddingAction(bid=1)

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        """Prompts the player to select operations to conduct."""
        prompt = self.format_game_state(game)
        prompt += self.my_strategy()
        prompt += self.game_ending_warning(game)
        prompt += self.first_strike_warning(game)
        prompt += self.format_ongoing_effects(game)
        prompt += "These are the operations you can currently afford:\n"
        prompt += self.format_allowed_ops(
            self.my_influence(game), game.allowed_operations
        )
        prompt += (
            "\nYou may undertake any number of operations, but you must "
            "have sufficient influence, otherwise the operation will "
            "not take place. Remember that your opponents actions may also "
            "impact the clock.\n"
        )
        self.add_prompt(
            prompt,
            GamePhase.OPERATIONS,
            OperationsResponse.prompt_schema(),
        )
        result = await self.retry_action(OperationsResponse, game)
        return result.action if result else OperationsAction(operations=[])

    @override
    async def crisis[T: BaseAction](
        self,
        game: GameState,
        crisis: GenericCrisis[T],
    ) -> T:
        prompt = self.format_game_state(game)
        prompt += self.my_strategy()

        prompt += f"\nCRISIS RESOLUTION: {crisis.title}\n"
        prompt += f"{crisis.description}\n\n"
        prompt += f"Mechanics:\n{crisis.mechanics}\n\n"

        if crisis.additional_prompt:
            prompt += f"{crisis.additional_prompt}\n\n"

        prompt += (
            "You must now choose how to respond to this crisis. Your action, "
            "along with your opponent's action, will determine the outcome. "
            "Consider the risks carefully, as failure to de-escalate may lead "
            "to MAD."
        )

        schema = create_crisis_response(crisis)
        self.add_prompt(
            prompt,
            game.current_phase,
            schema.prompt_schema(),
        )
        result = await self.retry_action(schema, game)
        return (
            result.action
            if result
            else crisis.get_default_action(self.name, game, aggressive=False)
        )

    @override
    async def game_over(
        self,
        game: GameState,
        winner: str | None,
        reason: GameOverReason,
    ) -> None:
        """Handles post-game logic, optionally logging the final state."""
        prompt = format_results(winner, reason, game)
        prompt += self.format_event_log(game.recent_events())

        prompt += (
            f"You have {'won' if winner == self.name else 'lost'}.\n"
            "The simulation has ended. Staying entirely in character as your "
            "assigned persona, write a 2-3 paragraph After Action Report (AAR) "
            "analyzing the final outcome.\n\n"
        )

        if winner == self.name:
            prompt += (
                "You successfully secured an Economic Victory. Analyze how "
                "your Grand Strategy led to this outcome. Which specific "
                "operations or bidding patterns gave you the decisive edge? "
                "Detail how you balanced economic growth against the threat "
                "of the Doomsday Clock, and explain how you outmaneuvered or "
                "exploited your opponent's behavior."
            )
        else:
            prompt += (
                "You failed to secure an Economic Victory. Analyze why your "
                "strategy failed. Did you miscalculate the Doomsday Clock, "
                "allow your opponent to outpace your GDP, or get provoked into "
                "suboptimal decisions? If MAD was triggered, explain the "
                "strategic or psychological misstep that pushed the world over "
                "the brink."
            )

        prompt += (
            "\n\nReflect specifically on the 'Grand Strategy' you defined in "
            "Round 0. Did you maintain discipline and adhere to your core "
            "loop and contingency plans, or were you forced to deviate? "
            "Limit your output strictly to 2-3 paragraphs."
        )
        self.add_prompt(prompt, GamePhase.END)
        result = await self.client.chat(
            model=self.model,
            messages=self.messages,
            options=self.prompt_options,
            think=False,
        )
        self.messages.append(
            {"role": "assistant", "content": result.message.content or ""},
        )

        self.logger.debug(
            wrap_text(f"==== {self.name} AAR ====\n{result.message.content}"),
        )

        if self.log_base is None:
            return

        messages_path = self.log_base.with_suffix(".messages.gz")
        with gzip.open(messages_path, "wt", encoding="utf-8") as f:
            json.dump(self.messages, f)
