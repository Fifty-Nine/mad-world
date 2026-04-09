from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_world.actions import BaseAction, InvalidActionError
from mad_world.config import LLMParams, LLMPlayerConfig
from mad_world.events import ActionEvent, PlayerActor
from mad_world.ollama_player import ActionResponse, OllamaPlayer

if TYPE_CHECKING:
    from mad_world.core import GameState


@pytest.fixture
def mock_logger() -> logging.Logger:
    return logging.getLogger("test_logger")


@pytest.fixture
def player_config() -> LLMPlayerConfig:
    params = LLMParams(
        token_limit=100,
        context_size=200,
        temperature=0.7,
        repeat_penalty=1.1,
        repeat_last_n=64,
    )
    return LLMPlayerConfig(
        name="TestPlayer",
        persona="TestPersona",
        model="test-model",
        params=params,
    )


def test_ollama_player_init(player_config: Any, mock_logger: Any) -> None:
    player = OllamaPlayer(
        config=player_config,
        opponent_name="Opponent",
        logger=mock_logger,
    )
    assert player.name == "TestPlayer"
    assert player.opponent_name == "Opponent"
    assert player.persona == "TestPersona"
    assert player.model == "test-model"
    assert player.prompt_options["num_predict"] == 100
    assert player.logger == mock_logger


@pytest.fixture
def test_player(player_config: Any, mock_logger: Any) -> Any:
    player = OllamaPlayer(
        config=player_config,
        opponent_name="Opponent",
        logger=mock_logger,
    )
    player.client = AsyncMock()
    return player


@pytest.mark.asyncio
async def test_start_game(
    player_config: Any, mock_logger: Any, tmp_path: Any
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    player = OllamaPlayer(
        config=player_config,
        opponent_name="Opponent",
        log_dir=log_dir,
        logger=mock_logger,
    )

    mock_rules = MagicMock()
    mock_rules.max_clock_state = 24
    mock_rules.allowed_bids = [0, 1, 2]
    mock_rules.allowed_operations = {}

    await player.start_game(mock_rules)

    assert len(player.messages) == 1
    assert player.messages[0]["role"] == "system"
    assert "Superpower TestPlayer" in player.messages[0]["content"]
    assert "Opponent" in player.messages[0]["content"]
    assert "TestPersona" in player.messages[0]["content"]

    # Check log files
    settings_file = log_dir / "TestPlayer.model-settings.json"
    assert settings_file.exists()

    schemas_file = log_dir / "TestPlayer.schemas.json.gz"
    assert schemas_file.exists()


@pytest.mark.asyncio
async def test_retry_prompt_success(
    test_player: Any, basic_game: GameState
) -> None:

    class DummyAction(BaseAction):
        def validate_semantics(
            self, game: GameState, current_player: str
        ) -> None:
            pass

    class DummyResponse(ActionResponse):
        action: DummyAction

    test_player.client.chat.return_value.message.content = json.dumps(
        {"chain_of_thought": ["thought"], "action": {}}
    )
    test_player.client.chat.return_value.prompt_eval_count = 10
    test_player.client.chat.return_value.eval_count = 10

    test_player.messages = []

    response = await test_player.retry_prompt(DummyResponse, basic_game)
    assert response is not None
    assert isinstance(response.action, DummyAction)


@pytest.mark.asyncio
async def test_retry_prompt_validation_error(
    test_player: Any, basic_game: GameState
) -> None:

    class DummyAction(BaseAction):
        def validate_semantics(
            self, game: GameState, current_player: str
        ) -> None:
            pass

    class DummyResponse(ActionResponse):
        action: DummyAction

    # Missing required 'action' field triggers ValidationError
    test_player.client.chat.return_value.message.content = json.dumps(
        {"chain_of_thought": ["thought"]}
    )

    test_player.messages = []

    response = await test_player.retry_prompt(
        DummyResponse, basic_game, retries=2
    )
    assert response is None
    assert test_player.client.chat.call_count == 2

    # System errors added to history
    assert len(test_player.messages) == 2
    assert test_player.messages[0]["role"] == "system"
    assert "SYSTEM ERROR" in test_player.messages[0]["content"]


@pytest.mark.asyncio
async def test_retry_prompt_semantic_error(
    test_player: Any, basic_game: GameState
) -> None:

    class DummyAction(BaseAction):
        def validate_semantics(
            self, game: GameState, current_player: str
        ) -> None:
            raise InvalidActionError("Test semantics failed")  # noqa: TRY003

    class DummyResponse(ActionResponse):
        action: DummyAction

    test_player.client.chat.return_value.message.content = json.dumps(
        {"chain_of_thought": ["thought"], "action": {}}
    )

    test_player.messages = []

    response = await test_player.retry_prompt(
        DummyResponse, basic_game, retries=2
    )
    assert response is None
    assert test_player.client.chat.call_count == 2

    assert len(test_player.messages) == 2
    assert test_player.messages[0]["role"] == "system"
    assert "SYSTEM ERROR" in test_player.messages[0]["content"]
    assert "Test semantics failed" in test_player.messages[0]["content"]


@pytest.mark.asyncio
async def test_check_and_compress(
    test_player: Any, basic_game: GameState
) -> None:
    # Force compression
    test_player.compression_threshold = 0.5
    test_player.params.context_size = 100

    mock_response_obj = MagicMock()
    mock_response_obj.prompt_eval_count = 60
    mock_response_obj.eval_count = 10

    test_player.messages = [
        {"role": "system", "content": "original system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    # Mock compression chat response
    test_player.client.chat.return_value.message.content = "summarized history"

    await test_player._check_and_compress(mock_response_obj, basic_game)

    assert test_player.client.chat.call_count == 1

    assert len(test_player.messages) == 2
    assert test_player.messages[0]["content"] == "original system"
    assert "summarized history" in test_player.messages[1]["content"]


@pytest.mark.asyncio
async def test_check_and_compress_no_compression(
    test_player: Any, basic_game: GameState
) -> None:
    test_player.compression_threshold = 0.9
    test_player.params.context_size = 100

    mock_response_obj = MagicMock()
    mock_response_obj.prompt_eval_count = 10
    mock_response_obj.eval_count = 10

    await test_player._check_and_compress(mock_response_obj, basic_game)

    assert test_player.client.chat.call_count == 0


def test_first_strike_warning(test_player: Any, basic_game: GameState) -> None:
    test_player.name = "Alpha"
    test_player.opponent_name = "Omega"

    # Set to tied
    basic_game.players["Alpha"].gdp = 10
    basic_game.players["Omega"].gdp = 10
    assert not test_player.first_strike_warning(basic_game)

    # Set to ahead
    basic_game.players["Alpha"].gdp = 25
    assert bool(test_player.first_strike_warning(basic_game))

    # Set to behind
    basic_game.players["Alpha"].gdp = 10
    basic_game.players["Omega"].gdp = 25
    assert bool(test_player.first_strike_warning(basic_game))


def test_game_ending_warning(test_player: Any, basic_game: GameState) -> None:
    test_player.name = "Alpha"
    test_player.opponent_name = "Omega"
    basic_game.rules.round_count = 10

    # test early game
    basic_game.current_round = 2
    assert not test_player.game_ending_warning(basic_game)

    # test late game, winning
    basic_game.current_round = 9
    basic_game.players["Alpha"].gdp = 100
    assert bool(test_player.game_ending_warning(basic_game))

    # test late game, losing
    basic_game.players["Alpha"].gdp = 10
    basic_game.players["Omega"].gdp = 100
    assert bool(test_player.game_ending_warning(basic_game))


def test_doomsday_warning(test_player: Any, basic_game: GameState) -> None:
    # `doomsday_clock` is computed from `escalation_track`
    # By default, basic_game has an empty track (doomsday clock = 0)
    assert not test_player.doomsday_warning(basic_game)

    # Test near limit where warnings should trigger
    basic_game.escalation_track = [PlayerActor(name="Alpha")] * 22
    assert bool(test_player.doomsday_warning(basic_game))


def test_escalation_debt(test_player: Any, basic_game: GameState) -> None:
    test_player.name = "Alpha"
    test_player.opponent_name = "Omega"

    # No debt
    assert not test_player.escalation_debt(basic_game)

    # High debt difference
    # The debt calculation divides by max_clock_state / 6.
    # With a max_clock_state of 24, diff must be >= 4.
    # Let's add multiple clock_delta events to build up debt

    for _ in range(10):
        basic_game.event_log.append(
            ActionEvent(
                actor=PlayerActor(name="Alpha"),
                description="Alpha bid high",
                clock_delta=1,
            )
        )
        basic_game.escalation_track.append(PlayerActor(name="Alpha"))

    assert bool(test_player.escalation_debt(basic_game))


def test_format_game_state(test_player: Any, basic_game: GameState) -> None:
    test_player.name = "Alpha"
    test_player.opponent_name = "Omega"

    state_str = test_player.format_game_state(basic_game)

    # We just want to ensure it formats effectively without crashing
    # and returns a valid non-empty string state
    assert isinstance(state_str, str)
    assert bool(state_str)
