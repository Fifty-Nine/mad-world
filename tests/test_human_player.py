"""Tests for the human_player module."""

import contextlib
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_world.core import (
    BiddingAction,
    GameState,
    InitialMessageAction,
)
from mad_world.human_player import HumanPlayer


@contextlib.contextmanager
def mock_human_input(
    player: HumanPlayer, return_value: Any = None, side_effect: Any = None
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
            raise ValueError("test error")
        return InitialMessageAction(message_to_opponent=t)

    side_effect = ["error", "valid"]
    with mock_human_input(player, side_effect=side_effect):
        action = await player.retry_prompt(
            basic_game, "Prompt: ", parse_with_error
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
            basic_game, "Bid: ", lambda t: BiddingAction(bid=int(t))
        )

    assert action.bid == 2
