"""Human player implementation for Mad World."""
# ruff: noqa: T201

from __future__ import annotations

import inspect
from enum import Enum
from typing import TYPE_CHECKING, Any, override

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import DummyCompleter, WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout
from pydantic import ValidationError

from mad_world.actions import (
    BaseAction,
    BiddingAction,
    ChatAction,
    InitialMessageAction,
    InvalidActionError,
    MessagingAction,
    OperationsAction,
)
from mad_world.players import GamePlayer

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic.fields import FieldInfo

    from mad_world.core import GameState
    from mad_world.crises import BaseCrisis, GenericCrisis


class FinishInput(Exception):
    """Raised to indicate that the user has finished entering input."""


class HumanPlayer(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.session: PromptSession[str] = PromptSession()
        self.operations_completer: WordCompleter | None = None

    @override
    async def start_game(self, game: GameState) -> None:
        await super().start_game(game)
        self.operations_completer = WordCompleter(
            list(game.rules.allowed_operations.keys()),
        )

    def _print_mandates(self, game: GameState) -> None:
        player_state = game.players[self.name]
        if player_state.mandates:
            print("\n--- Secret Mandates ---")
            for m in player_state.mandates:
                print(f"- {m.title}: {m.description}")
            print("-----------------------\n")

    async def prompt_user[T: BaseAction](
        self,
        game: GameState,
        prompt: str,
        parse: Callable[[str], T],
        completer: WordCompleter | None = None,
    ) -> T | None:
        self._print_mandates(game)
        try:
            with patch_stdout():
                user_input = await self.session.prompt_async(
                    prompt,
                    completer=completer or DummyCompleter(),
                )

            action = parse(user_input)
            action.validate_semantics(game, self.name)

        except ValueError:
            print("Invalid entry. Please enter a valid response.")

        except InvalidActionError as e:
            print(e)

        else:
            return action

        return None

    async def retry_prompt[T: BaseAction](
        self,
        game: GameState,
        prompt: str,
        parse: Callable[[str], T],
        completer: WordCompleter | None = None,
    ) -> T:
        while True:
            result = await self.prompt_user(game, prompt, parse, completer)
            if result is not None:
                return result

    def _validate_action(self, action: BaseAction, game: GameState) -> None:
        action.validate_semantics(game, self.name)

    async def _get_action[T: BaseAction](
        self, game: GameState, action_class: type[T]
    ) -> T:
        while True:
            try:
                action = await self._prompt_crisis_action(action_class)
                self._validate_action(action, game)
            except (ValidationError, InvalidActionError) as e:
                print(f"Invalid input: {e}")
                print("Please try again.")
            else:
                return action

    @override
    async def chat(
        self, game: GameState, remaining_messages: int
    ) -> ChatAction:
        print("\n" + "=" * 40)
        print(
            "Direct back-and-forth communication channel. You have "
            f"{remaining_messages} messages left."
        )
        return await self._get_action(game, ChatAction)

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        print(game.describe_state())
        print(f"\n[{self.name}] Initial Message Phase")
        return await self._get_action(game, InitialMessageAction)

    @override
    async def message(self, game: GameState) -> MessagingAction:
        print(f"\n{game.describe_state()}")
        print(f"\n[{self.name}] Messaging Phase")
        return await self._get_action(game, MessagingAction)

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        print(f"\n{game.describe_state()}")
        print(f"\n[{self.name}] Bidding Phase")
        print(f"Allowed bids: {game.allowed_bids}")
        return await self.retry_prompt(
            game,
            "Enter your bid: ",
            lambda text: BiddingAction(bid=int(text)),
        )

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        print(f"\n{game.describe_state()}")
        print(f"\n[{self.name}] Operations Phase")
        print("Available Operations:")
        for op_name, op_def in game.allowed_operations.items():
            print(f"  - {op_name} (Cost: {op_def.influence_cost})")

        def parse_op(text: str) -> OperationsAction:
            if not text.strip():
                raise FinishInput

            return OperationsAction(operations=[text.strip()])

        ops = []
        while True:
            try:
                result = await self.retry_prompt(
                    game,
                    "Enter operation name to add (or press Enter to finish): ",
                    parse_op,
                    self.operations_completer,
                )
                ops.append(result.operations[0])
            except FinishInput:
                break

        return OperationsAction(operations=ops)

    def _extract_enum_type(self, annotation: Any) -> type[Enum] | None:
        """Extract an Enum class from a type annotation."""
        if inspect.isclass(annotation) and issubclass(annotation, Enum):
            return annotation

        for arg in getattr(annotation, "__args__", []):
            if inspect.isclass(arg) and issubclass(arg, Enum):
                return arg

        return None

    async def _prompt_enum_field(
        self, prompt_text: str, enum_type: type[Enum]
    ) -> Any:
        valid_values = [f"{e.name} ({e.value})" for e in enum_type]
        prompt_text += f"\n[Valid values: {', '.join(valid_values)}]: "

        completer = WordCompleter(
            [e.name for e in enum_type] + [str(e.value) for e in enum_type],
            ignore_case=True,
        )

        with patch_stdout():
            user_input = await self.session.prompt_async(
                prompt_text, completer=completer
            )

        user_input = user_input.strip()
        if not user_input:
            return None

        if user_input.isdigit():
            return int(user_input)

        enum_member = getattr(
            enum_type, user_input.replace(" ", "_").upper(), None
        )
        if enum_member is not None:
            return enum_member.value

        return user_input

    async def _prompt_standard_field(self, prompt_text: str) -> Any:
        prompt_text += ": "

        with patch_stdout():
            user_input = await self.session.prompt_async(
                prompt_text, completer=DummyCompleter()
            )

        return user_input.strip() or None

    async def _prompt_crisis_field(
        self, field_name: str, field_info: FieldInfo
    ) -> Any:
        prompt_text = f"Enter value for '{field_name}'"
        if field_info.description:
            prompt_text += f" ({field_info.description})"

        enum_type = self._extract_enum_type(field_info.annotation)
        if enum_type is not None:
            return await self._prompt_enum_field(prompt_text, enum_type)

        return await self._prompt_standard_field(prompt_text)

    async def _prompt_crisis_action[T: BaseAction](
        self, action_class: type[T]
    ) -> T:
        field_values: dict[str, Any] = {}
        for field_name, field_info in action_class.model_fields.items():
            if (
                val := await self._prompt_crisis_field(field_name, field_info)
            ) is not None:
                field_values[field_name] = val
        return action_class(**field_values)

    @override
    async def crisis_message(
        self,
        game: GameState,
        crisis: BaseCrisis,
    ) -> MessagingAction:
        print(f"\n{game.describe_state()}")
        print(f"\n[{self.name}] Crisis Alert: {crisis.title}")
        print(crisis.description)
        print(f"Mechanics: {crisis.mechanics}")

        print(f"\n[{self.name}] Crisis Messaging Phase")
        return await self._get_action(game, MessagingAction)

    @override
    async def crisis[T: BaseAction](
        self,
        game: GameState,
        crisis: GenericCrisis[T],
    ) -> T:
        print(f"\n{game.describe_state()}")
        print(f"\n[{self.name}] Crisis Phase: {crisis.title}")

        while True:
            try:
                action = await self._prompt_crisis_action(crisis.action_type)
                action.validate_semantics(game, self.name)
            except (ValidationError, InvalidActionError) as e:
                print(f"Invalid input: {e}")
                print("Please try again.")
            else:
                return action
