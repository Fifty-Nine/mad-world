"""Tests for the human_player module."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prompt_toolkit.completion import DummyCompleter, WordCompleter
from pydantic import Field

from mad_world.actions import (
    BaseAction,
    BiddingAction,
    InitialMessageAction,
)
from mad_world.crises import GenericCrisis, StandoffCrisis
from mad_world.enums import StandoffPosture
from mad_world.human_player import HumanPlayer

if TYPE_CHECKING:
    from collections.abc import Generator

    from mad_world.core import GameState


@contextlib.contextmanager
def mock_human_input(
    player: HumanPlayer,
    return_value: Any = None,
    side_effect: Any = None,
) -> Generator[AsyncMock, None, None]:
    """Context manager to mock human player input."""
    mock_prompt = AsyncMock(return_value=return_value, side_effect=side_effect)
    with (
        patch.object(player.session, "prompt_async", mock_prompt),
        patch("mad_world.human_player.patch_stdout", MagicMock()),
    ):
        yield mock_prompt


@pytest.mark.asyncio
async def test_human_player_initial_message(basic_game: GameState) -> None:
    player = HumanPlayer("Alpha")

    with mock_human_input(player, return_value="Hello World"):
        action = await player.initial_message(basic_game)

    assert action.message_to_opponent == "Hello World"


@pytest.mark.asyncio
async def test_human_player_message(basic_game: GameState) -> None:
    player = HumanPlayer("Alpha")

    with mock_human_input(player, return_value="  I am here  "):
        action = await player.message(basic_game)

    assert action.message_to_opponent == "I am here"

    with mock_human_input(player, return_value=""):
        action = await player.message(basic_game)

    assert action.message_to_opponent is None


@pytest.mark.asyncio
async def test_human_player_bid(basic_game: GameState) -> None:
    player = HumanPlayer("Alpha")

    with mock_human_input(player, return_value="3"):
        action = await player.bid(basic_game)

    assert action.bid == 3


@pytest.mark.asyncio
async def test_human_player_operations(basic_game: GameState) -> None:
    player = HumanPlayer("Alpha")
    player.start_game(basic_game.rules)

    side_effect = ["domestic-investment", "aggressive-extraction", ""]
    with mock_human_input(player, side_effect=side_effect) as mock_prompt:
        action = await player.operations(basic_game)

    assert action.operations == ["domestic-investment", "aggressive-extraction"]
    assert mock_prompt.call_count == 3


@pytest.mark.asyncio
async def test_human_player_prompt_user_invalid_then_valid(
    basic_game: GameState,
) -> None:
    player = HumanPlayer("Alpha")

    def parse_with_error(t: str) -> InitialMessageAction:
        if t == "error":
            raise ValueError("test error")  # noqa: TRY003
        return InitialMessageAction(message_to_opponent=t)

    side_effect = ["error", "valid"]
    with mock_human_input(player, side_effect=side_effect):
        action = await player.retry_prompt(
            basic_game,
            "Prompt: ",
            parse_with_error,
        )

    assert action.message_to_opponent == "valid"


@pytest.mark.asyncio
async def test_human_player_prompt_user_invalid_semantics(
    basic_game: GameState,
) -> None:
    player = HumanPlayer("Alpha")

    basic_game.rules.allowed_bids = [0, 1, 2, 3, 5]

    side_effect = ["10", "2"]
    with mock_human_input(player, side_effect=side_effect):
        action = await player.retry_prompt(
            basic_game,
            "Bid: ",
            lambda t: BiddingAction(bid=int(t)),
        )

    assert action.bid == 2


@pytest.mark.asyncio
async def test_human_player_crisis(basic_game: GameState) -> None:
    player = HumanPlayer("Alpha")

    crisis = StandoffCrisis()

    # Test with valid integer string input
    with mock_human_input(player, side_effect=["1"]):
        action = await player.crisis(basic_game, crisis)

    assert action.posture == StandoffPosture.BACK_DOWN

    # Test with valid enum name input (case-insensitive)
    with mock_human_input(player, side_effect=["sTaNd_fIrM"]):
        action = await player.crisis(basic_game, crisis)

    assert action.posture == StandoffPosture.STAND_FIRM

    # Test with invalid input then valid input
    with mock_human_input(player, side_effect=["", "invalid", "1"]):
        action = await player.crisis(basic_game, crisis)

    assert action.posture == StandoffPosture.BACK_DOWN

    # Test EOFError propagates
    with (
        mock_human_input(player, side_effect=EOFError),
        pytest.raises(EOFError),
    ):
        await player.crisis(basic_game, crisis)


@pytest.mark.asyncio
async def test_human_player_crisis_coverage(basic_game: GameState) -> None:
    player = HumanPlayer("Alpha")

    class DummyAction(BaseAction):
        text_field: str
        list_field: list[int] = Field(default_factory=list)
        opt_enum: StandoffPosture | None = None

    class DummyCrisis(GenericCrisis[DummyAction]):
        card_kind: ClassVar[Literal["Dummy"]] = "Dummy"
        title: ClassVar[str] = "Dummy"
        description: ClassVar[str] = "Dummy"
        mechanics: ClassVar[str] = "Dummy"

        @override
        def get_action_type(self) -> type[DummyAction]:
            return DummyAction

        @override
        def resolve(
            self, game: GameState, actions: dict[str, DummyAction]
        ) -> list[Any]:
            return []

        @override
        def get_default_action(self, *, aggressive: bool) -> DummyAction:
            return DummyAction(text_field="default")

    crisis = DummyCrisis()

    # Test non-enum string field and list field
    with mock_human_input(player, side_effect=["hello", "", "1"]):
        action = await player.crisis(basic_game, crisis)

    assert action.text_field == "hello"
    assert action.list_field == []
    assert action.opt_enum == StandoffPosture.BACK_DOWN


@pytest.mark.asyncio
async def test_human_player_crisis_message(basic_game: GameState) -> None:
    player = HumanPlayer("Alpha")
    crisis = StandoffCrisis()

    with mock_human_input(player, return_value="A message to my opponent"):
        action = await player.crisis_message(basic_game, crisis)

    assert action.message_to_opponent == "A message to my opponent"

    with mock_human_input(player, return_value=""):
        action = await player.crisis_message(basic_game, crisis)

    assert action.message_to_opponent is None


@pytest.mark.asyncio
async def test_human_player_completer_leak(basic_game: GameState) -> None:
    player = HumanPlayer("Alpha")
    player.start_game(basic_game.rules)

    with mock_human_input(
        player, side_effect=["domestic-investment", ""]
    ) as mock_prompt:
        await player.operations(basic_game)

        assert mock_prompt.call_count == 2
        for call in mock_prompt.call_args_list:
            assert isinstance(call.kwargs.get("completer"), WordCompleter)

    with mock_human_input(player, side_effect=["My message"]) as mock_prompt:
        await player.message(basic_game)

        assert mock_prompt.call_count == 1
        last_call_kwargs = mock_prompt.call_args.kwargs
        assert isinstance(last_call_kwargs.get("completer"), DummyCompleter)
