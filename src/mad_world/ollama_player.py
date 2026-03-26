"""Ollama player implementation for Mad World."""

import gzip
import json
import re
import textwrap
from pathlib import Path
from typing import Any, override

import ollama
from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from mad_world.core import (
    BaseAction,
    BiddingAction,
    GameOverReason,
    GamePhase,
    GamePlayer,
    GameState,
    InitialMessageAction,
    InvalidActionError,
    MessagingAction,
    OperationsAction,
    PlayerState,
    format_results,
    logging,
)
from mad_world.rules import GameRules
from mad_world.util import escalation_budget, pareto_optimal_bid, wrap_text


class ActionResponse(BaseModel):
    chain_of_thought: list[str] = Field(
        description=(
            "Think through the previous turn. Did you advance "
            "your goals? Did your opponent act in accordance with "
            "their words? Did you make any mistakes? What are you "
            "going to do now? Limit to 10-20 brief thoughts, one "
            "thought per list item."
        )
    )
    action: BaseAction

    @model_validator(mode="before")
    @classmethod
    def unprefix_keys(cls, data: Any) -> Any:
        """Strip off digit prefixes added by reorder_schema."""

        def clean(d: Any) -> Any:
            if isinstance(d, dict):
                return {
                    re.sub(r"^\d\d_", "", str(k)): clean(v)
                    for k, v in d.items()
                }
            if isinstance(d, list):
                return [clean(v) for v in d]
            return d

        return clean(data)

    @staticmethod
    def reorder_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Given the JSON Schema for the action, reorder the properties
        so that the `action` field always comes last. We also prefix the
        property keys with numbers (e.g. 00_, 01_) because the underlying
        llama.cpp grammar engine forcefully alphabetizes schema properties.
        These prefixes are stripped off by `unprefix_keys` before the caller
        ultimately sees them.
        """

        def process_obj(obj: dict[str, Any]) -> None:
            if "properties" not in obj:
                return

            old_props = obj["properties"]
            action = old_props.pop("action", None)
            required = obj.get("required", [])
            new_props = {}

            for i, field in enumerate(old_props.keys()):
                field_obj = old_props[field]
                new_key = f"{i:02d}_{field}"

                new_props[new_key] = field_obj
                if field in required:
                    required.remove(field)
                    required.append(new_key)

            if action is not None:
                new_props["99_action"] = action

            if "action" in required:
                required.remove("action")
                required.append("99_action")

            obj["properties"] = new_props

        process_obj(schema)
        for def_schema in schema.get("$defs", {}).values():
            process_obj(def_schema)

        return schema

    @classmethod
    def format_schema(cls) -> dict[str, Any]:
        return cls.reorder_schema(cls.model_json_schema())

    @classmethod
    def prompt_schema(cls) -> str:
        return json.dumps(cls.format_schema(), indent=2, ensure_ascii=False)


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
            "As a reminder, here is the grand strategy you originally "
            "proposed:\n"
            "  Core gameply loop:\n"
            f"    {self.core_loop}\n"
            "  Clock management strategy:\n"
            f"    {self.clock_management}\n"
            "  Contingency plan:\n"
            f"    {self.contingency_plan}\n"
            "\nYou are not strictly bound to these directives, but you "
            "should have a reason for deviating from them.\n"
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
        )
    )

    action: InitialMessageAction = Field(
        description="Your finalized action for this phase."
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
        description="Your finalized action for this phase."
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
        )
    )
    tactical_plan: str = Field(
        description=(
            "Based on the victory check and your persona, "
            "detail your specific plan for the Bidding phase. CRITICAL: "
            "If your intended bid is greater than the pareto-optimal bid "
            "OR is one of the listed bids that may trigger MAD, you MUST "
            "justify why the risk is worth the reward."
        )
    )
    action: BiddingAction = Field(
        description="Your finalized action for this phase."
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
        )
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
        )
    )
    action: OperationsAction = Field(
        description="Your finalized action for this phase."
    )


class OllamaPlayer(GamePlayer):
    def __init__(
        self,
        name: str,
        opponent_name: str,
        model: str = "qwen3.5:9b",
        token_limit: int = 8192,
        context_size: int = 2**15,
        temperature: float = 0.0,
        persona: str | None = None,
        log_dir: Path | None = None,
    ) -> None:
        super().__init__(name)
        self.opponent_name = opponent_name
        self.persona = persona
        self.model = model
        self.client = ollama.AsyncClient()
        self.messages: list[dict[str, str]] = []
        self.token_limit = token_limit
        self.context_size = context_size
        self.temperature = temperature
        self.prompt_options = {
            "num_predict": self.token_limit,
            "num_ctx": self.context_size,
            "temperature": self.temperature,
            "think": False,
        }
        self.grand_strategy: GrandStrategy | None = None
        self.log_base = (
            log_dir / f"{self.name}" if log_dir is not None else None
        )

    def start_game(self, rules: GameRules) -> None:
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
            f"{rules.max_clock_state}, MAD is triggered. Both players suffer "
            "a catastrophic penalty of -1000 GDP, and the game immediately "
            "ends.\n"
            "  Victory Conditions & Elimination:\n"
            "    At the end of the game (either after round 10 or when "
            "triggered by MAD), scores are tallied. Any player with a "
            "score less than zero is automatically eliminated and loses "
            "the game. If both players have a negative score, the game is "
            "a mutual loss (tie). The winner (if any) is the player with "
            "the highest GDP who has not been otherwise eliminated.\n"
            "  Round Structure (Two Phases):\n"
            "    Each round consists of two phases. You will be prompted "
            "separately for each.\n"
            "    Phase 1: Bidding & Posturing\n"
            "      You and your opponent will each secretly submit an "
            f"Aggression Bid (one of {rules.allowed_bids}). This value "
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
            avail_inf=None, rules=rules, indent="        "
        )
        prompt += (
            "\nYou will receive a prompt detailing the current Phase and Game "
            "State and respond with your action matching the provided schema.\n"
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
        )
        if self.persona is not None:
            prompt += (
                "You have been randomly assigned the following persona "
                f"for this engagement: {self.persona}\n"
                "Act accordingly.\n"
            )
        self.messages.append({"role": "system", "content": prompt})
        logging.debug(
            f"==== {self.name} system prompt ====\n"
            + wrap_text(prompt, width=80)
            + "\n"
        )

        if self.log_base is None:
            return

        settings_path = self.log_base.with_suffix(".model-settings.json")
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": self.model, "options": self.prompt_options},
                indent=2,
                ensure_ascii=False,
                fp=f,
            )

        schemas_path = self.log_base.with_suffix(".schemas.json.gz")
        with gzip.open(schemas_path, "wt", encoding="utf-8") as f:
            schemas = {
                "opening": InitialMessageResponse.prompt_schema(),
                "messaging": MessagingResponse.prompt_schema(),
                "bidding": BiddingResponse.prompt_schema(),
                "operations": OperationsResponse.prompt_schema(),
            }
            json.dump(schemas, f, indent=2, ensure_ascii=False)

    @staticmethod
    def format_player_state(player: PlayerState) -> str:
        return f"{player.name}: {player.gdp} GDP, {player.influence} Inf"

    @staticmethod
    def format_allowed_ops(
        avail_inf: int | None, rules: GameRules, indent: str = ""
    ) -> str:
        ops = (
            op
            for op in rules.allowed_operations.values()
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
    def format_event_log(game: GameState) -> str:
        result = "Recent Events:\n"
        result += textwrap.indent(
            "\n".join(
                e.description for e in game.recent_events() if not e.secret
            ),
            prefix="  ",
        )
        result += "\n"
        return result

    @staticmethod
    def format_game_state(game: GameState) -> str:
        result = (
            f"Round {game.current_round} of {game.rules.round_count}\n"
            f"Phase: {game.current_phase.name}\n"
            f"Doomsday clock: {game.doomsday_clock}/"
            f"{game.rules.max_clock_state}"
            f"{' (CRITICAL)' if game.doomsday_clock >= 20 else ''}\n"
        )
        budget = escalation_budget(
            game.doomsday_clock, game.rules.max_clock_state
        )
        result += f"Escalation budget: {budget}\nPlayers:\n"

        result += "\n".join(
            OllamaPlayer.format_player_state(p) for p in game.players.values()
        )

        result += "\n" + OllamaPlayer.format_event_log(game)
        return result

    def doomsday_warning(self, game: GameState) -> str:
        risky, deadly = game.rules.get_doomsday_bids(game.doomsday_clock)

        if len(risky) == 0 and len(deadly) == 0:
            return ""

        def severity(game: GameState) -> str:
            level = 1.0 * game.doomsday_clock / game.rules.max_clock_state
            if level >= 0.9:
                return "!!!!! CRITICAL WARNING !!!!!\n"

            if level >= 0.8:
                return "WARNING: "

            return "NOTE: "

        result = f"{severity(game)}You are at risk of triggering MAD.\n"

        result += "".join(
            f"\n- A bid of {bid} RISKS MAD if your opponent "
            f"bids {obid} or more."
            for bid, obid in risky
        )
        result += "".join(
            f"\n- A bid of {bid} GUARANTEES MAD regardless of your opponent's "
            "action."
            for bid in deadly
        )
        result += (
            "\nRemember that if MAD occurs, you will lose even if you "
            "are currently winning."
        )
        return result

    @staticmethod
    def dump_model_response(response: BaseModel) -> str:
        return json.dumps(response.model_dump(), indent=2, ensure_ascii=False)

    async def retry_prompt[T: ActionResponse](
        self,
        response_model: type[T],
        game: GameState,
        retries: int = 3,
    ) -> T | None:
        log_header = (
            f"==== {game.current_phase.name} {self.name} response====\n"
        )
        for _i in range(retries):
            adapter = TypeAdapter(response_model)
            result_obj = await self.client.chat(
                model=self.model,
                messages=self.messages,
                format=response_model.format_schema(),
                options=self.prompt_options,
            )
            result = result_obj.message.content

            try:
                response = adapter.validate_json(result or "")
            except ValidationError as e:
                logging.debug(
                    wrap_text(
                        f"{log_header}Failed: {e}\nModel Response: {result}\n"
                        f"Model done reason: {result_obj.done_reason}\n"
                    )
                )
                self.messages.append(
                    {
                        "role": "system",
                        "content": "SYSTEM ERROR: You previously generated a "
                        "response that triggered the following error during "
                        f"validation: {e!r}\n"
                        "This response has been discarded and you are being "
                        "given another opportunity to generate a valid result. "
                        "Please ensure you limit your internal "
                        "monologue to a few paragraphs and follow "
                        "the provided schema exactly.",
                    }
                )
                continue

            action = response.action
            formatted_response = textwrap.indent(
                self.dump_model_response(response), prefix="  "
            )
            try:
                action.validate_semantics(game, self.name)
                self.messages.append(
                    {"role": "assistant", "content": result or ""}
                )

                logging.debug(
                    wrap_text(
                        f"{log_header}Model Response:\n{formatted_response}\n"
                    )
                )
                return response

            except InvalidActionError as e:
                logging.debug(
                    wrap_text(
                        f"{log_header}Semantic Error: {e}\n"
                        f"Model Response:\n{formatted_response}\n"
                    )
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
                    }
                )

        logging.debug(f"{log_header}Failed after {retries} retries.")
        return None

    def game_ending_warning(self, game: GameState) -> str:
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

    def my_strategy(self) -> str:
        return self.grand_strategy.to_prompt() if self.grand_strategy else ""

    def my_influence(self, game: GameState) -> int:
        return game.players[self.name].influence

    def add_prompt(
        self, prompt: str, phase: GamePhase, schema: str | None = None
    ) -> None:
        if schema is not None:
            prompt += (
                "You should output strictly valid JSON matching this JSON "
                "Schema:\n"
            )

        logging.debug(
            f"==== {self.name} {phase.name} prompt ====\n"
            f"{wrap_text(prompt)}"
            f"{'[...]' if schema else ''}\n"
        )
        self.messages.append(
            {"role": "user", "content": prompt + (schema or "")}
        )

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        prompt = self.format_game_state(game)
        prompt += (
            "You may now provide your initial message to your opponent. "
            "Each turn you will be allowed to send one message, as will your "
            "opponent. You will have a chance to send a message to your "
            "opponent before they have to commit to their actions in each "
            "phase. You should use this channel to conduct diplomacy, respond "
            "to inquiries, issue threats, etc.\n"
        )
        self.add_prompt(
            prompt,
            GamePhase.OPENING,
            InitialMessageResponse.prompt_schema(),
        )
        result = await self.retry_prompt(InitialMessageResponse, game)
        if result is None:
            return InitialMessageAction()

        self.grand_strategy = result.grand_strategy
        return result.action

    @override
    async def message(self, game: GameState) -> MessagingAction:
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
        self.add_prompt(
            prompt,
            game.current_phase,
            MessagingResponse.prompt_schema(),
        )
        result = await self.retry_prompt(MessagingResponse, game)
        return result.action if result else MessagingAction()

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        prompt = self.format_game_state(game)
        pbid = pareto_optimal_bid(
            game.doomsday_clock,
            game.rules.max_clock_state,
            game.rules.allowed_bids,
        )
        prompt += f"Your pareto optimal bid is {pbid}.\n"
        prompt += self.my_strategy()
        prompt += self.game_ending_warning(game)
        prompt += self.first_strike_warning(game)
        prompt += (
            "Reminder: these are the allowed bids you may submit: "
            f"{game.rules.allowed_bids}\n"
            "Remember that your opponent's bid will also affect the clock, "
            "and you WILL NOT learn of ther bid until after you submit yours.\n"
        )

        prompt += self.doomsday_warning(game)
        self.add_prompt(
            prompt,
            GamePhase.BIDDING,
            BiddingResponse.prompt_schema(),
        )
        result = await self.retry_prompt(BiddingResponse, game)
        return result.action if result else BiddingAction(bid=1)

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        prompt = self.format_game_state(game)
        prompt += self.my_strategy()
        prompt += self.game_ending_warning(game)
        prompt += self.first_strike_warning(game)
        prompt += "These are the operations you can currently afford:\n"
        prompt += self.format_allowed_ops(self.my_influence(game), game.rules)
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
        result = await self.retry_prompt(OperationsResponse, game)
        return result.action if result else OperationsAction(operations=[])

    @override
    async def game_over(
        self, game: GameState, winner: str | None, reason: GameOverReason
    ) -> None:
        prompt = format_results(winner, reason, game)
        prompt += self.format_event_log(game)

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
        )
        self.messages.append(
            {"role": "assistant", "content": result.message.content or ""}
        )

        logging.debug(
            wrap_text(f"==== {self.name} AAR ====\n{result.message.content}")
        )

        if self.log_base is None:
            return

        messages_path = self.log_base.with_suffix(".messages.gz")
        with gzip.open(messages_path, "wt", encoding="utf-8") as f:
            json.dump(self.messages, f)
