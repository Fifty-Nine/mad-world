"""Human player implementation for Mad World."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout

from mad_world.actions import (
    BaseAction,
    BiddingAction,
    InitialMessageAction,
    InvalidActionError,
    MessagingAction,
    OperationsAction,
)
from mad_world.players import GamePlayer

if TYPE_CHECKING:
    from collections.abc import Callable

    from mad_world.core import GameState
    from mad_world.crises import GenericCrisis
    from mad_world.rules import GameRules


class FinishInput(Exception):
    """Raised to indicate that the user has finished entering input."""

    pass


class HumanPlayer(GamePlayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.session: PromptSession[str] = PromptSession()
        self.operations_completer: WordCompleter | None = None

    def start_game(self, game: GameRules) -> None:
        self.operations_completer = WordCompleter(
            list(game.allowed_operations.keys()),
        )

    async def prompt_user[T: BaseAction](
        self,
        game: GameState,
        prompt: str,
        parse: Callable[[str], T],
        completer: WordCompleter | None = None,
    ) -> T | None:
        try:
            with patch_stdout():
                user_input = await self.session.prompt_async(
                    prompt,
                    completer=completer,
                )

            action = parse(user_input)
            action.validate_semantics(game, self.name)

            return action

        except ValueError:
            print("Invalid entry. Please enter a valid response.")

        except InvalidActionError as e:
            print(e)

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

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        print(game.describe_state())
        print(f"\n[{self.name}] Initial Message Phase")
        return await self.retry_prompt(
            game,
            "Enter your initial message to your opponent: ",
            lambda m: InitialMessageAction(message_to_opponent=m),
        )

    @override
    async def message(self, game: GameState) -> MessagingAction:
        print(f"\n{game.describe_state()}")
        print(f"\n[{self.name}] Messaging Phase")
        return await self.retry_prompt(
            game,
            "Enter a message to your opponent (or press Enter to skip): ",
            lambda m: MessagingAction(
                message_to_opponent=m.strip() if m.strip() else None,
            ),
        )

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        print(f"\n{game.describe_state()}")
        print(f"\n[{self.name}] Bidding Phase")
        print(f"Allowed bids: {game.rules.allowed_bids}")
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
        for op_name, op_def in game.rules.allowed_operations.items():
            print(f"  - {op_name} (Cost: {op_def.influence_cost})")

        def parse_op(text: str) -> OperationsAction:
            if not text.strip():
                raise FinishInput()

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

    @override
    async def crisis[T: BaseAction](
        self,
        game: GameState,
        crisis: GenericCrisis[T],
    ) -> T:
        # FIXME
        return crisis.get_default_action(aggressive=True)  # pragma: no cover
