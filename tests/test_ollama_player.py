from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from mad_world.actions import (
    BaseAction,
    BiddingAction,
    InitialMessageAction,
    InvalidActionError,
    MessagingAction,
    OperationsAction,
)
from mad_world.config import LLMParams, LLMPlayerConfig
from mad_world.core import GameState, PlayerState
from mad_world.crises import StandoffCrisis
from mad_world.enums import GameOverReason, GamePhase
from mad_world.events import ActionEvent, PlayerActor
from mad_world.mandates import PacifistUtopiaMandate
from mad_world.ollama_player import (
    ActionResponse,
    GrandStrategy,
    OllamaPlayer,
    PlayerArchetype,
    create_persona_schema,
)

if TYPE_CHECKING:
    from mad_world.rules import GameRules


@pytest.fixture
def mock_logger() -> logging.Logger:
    return logging.getLogger("test_logger")


def test_format_mandates_empty() -> None:
    player = PlayerState(name="Alpha", gdp=42, influence=3, mandates=[])
    formatted = OllamaPlayer.format_mandates(player)
    assert formatted == ""


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
        name="Alpha",
        persona="TestPersona",
        model="test-model",
        params=params,
    )


def test_ollama_player_init(
    player_config: LLMPlayerConfig, mock_logger: Any
) -> None:
    player = OllamaPlayer(
        config=player_config,
        opponent_name="Omega",
        logger=mock_logger,
    )
    assert player.name == "Alpha"
    assert player.opponent_name == "Omega"
    assert player.persona == "TestPersona"
    assert player.model == "test-model"
    assert player.prompt_options["num_predict"] == 100
    assert player.logger == mock_logger


@pytest.fixture
def test_player(player_config: LLMPlayerConfig, mock_logger: Any) -> Any:
    player = OllamaPlayer(
        config=player_config,
        opponent_name="Opponent",
        logger=mock_logger,
    )
    player.client = AsyncMock()
    return player


@pytest.mark.asyncio
async def test_elaborate_persona(test_player: Any) -> None:

    test_player.persona = "Earnest Pacifist"
    test_player.client.chat.return_value.message.content = json.dumps(
        {
            "00_persona_seed": "Earnest Pacifist",
            "01_character_description": "He is a detailed pacifist.",
            "02_character_instructions": "Do things pacifistly.",
            "03_archetype": PlayerArchetype.PRESERVATIONIST.value,
            "99_name": "General Pacifisto",
        }
    )
    await test_player.elaborate_persona()
    assert test_player.client.chat.call_count == 1
    assert "He is a detailed pacifist." in test_player.persona
    assert "General Pacifisto" in test_player.persona

    # Test JSON parse error recovery (should just ignore and not crash)
    test_player.persona = "Earnest Pacifist"
    test_player.client.chat.reset_mock()
    test_player.client.chat.return_value.message.content = "Invalid JSON"
    await test_player.elaborate_persona()
    assert test_player.client.chat.call_count == 3
    # Persona remains unelaborated
    assert test_player.persona == "Earnest Pacifist"

    # Test when already elaborated
    test_player.client.chat.reset_mock()
    test_player.persona = "Already\n\nElaborated"
    await test_player.elaborate_persona()
    assert test_player.client.chat.call_count == 0

    # Test when None
    test_player.client.chat.reset_mock()
    test_player.persona = None
    await test_player.elaborate_persona()
    assert test_player.client.chat.call_count == 0


def test_elaborated_persona_schema_type() -> None:
    schema_type = create_persona_schema("Big Dummy")
    schema = schema_type.model_json_schema()

    assert schema["properties"]["persona_seed"]["const"] == "Big Dummy"

    base_args: dict[str, Any] = {
        "character_description": "",
        "character_instructions": "",
        "archetype": PlayerArchetype.SORE_LOSER,
        "name": "General Truly N. Competent",
    }

    with pytest.raises(ValidationError, match="Expected persona_seed"):
        schema_type(persona_seed="not good!", **base_args)

    valid = schema_type(persona_seed="Big Dummy", **base_args)

    dict_val = valid.model_dump()
    dict_val["persona_seed"] = "invalid"
    with pytest.raises(ValidationError, match="Expected persona_seed"):
        schema_type.model_validate_json(json.dumps(dict_val))


@pytest.mark.asyncio
async def test_start_game(
    player_config: LLMPlayerConfig,
    mock_logger: Any,
    tmp_path: Any,
    stable_rules: GameRules,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    player = OllamaPlayer(
        config=player_config,
        opponent_name="Opponent",
        log_dir=log_dir,
        logger=mock_logger,
    )

    stable_rules.max_clock_state = 24
    stable_rules.allowed_bids = [0, 1, 2]
    stable_rules.allowed_operations = {}

    mock_game = MagicMock(spec=GameState)
    mock_game.players = {"Alpha": MagicMock()}
    mock_game.rules = stable_rules

    await player.start_game(mock_game)

    assert len(player.messages) == 1
    assert player.messages[0]["role"] == "system"
    assert "Superpower Alpha" in player.messages[0]["content"]
    assert "Opponent" in player.messages[0]["content"]
    assert "TestPersona" in player.messages[0]["content"]

    # Check log files
    settings_file = log_dir / "Alpha.model-settings.json"
    assert settings_file.exists()

    schemas_file = log_dir / "Alpha.schemas.json.gz"
    assert schemas_file.exists()

    # Test start_game without log_base
    player_no_log = OllamaPlayer(
        config=player_config,
        opponent_name="Opponent",
        log_dir=None,
        logger=mock_logger,
    )
    player_no_log.client = AsyncMock()
    await player_no_log.start_game(mock_game)


@pytest.mark.asyncio
async def test_retry_action_success(
    test_player: Any, basic_game: GameState
) -> None:

    class DummyAction(BaseAction):
        def validate_semantics(self, game: GameState, player_name: str) -> None:
            pass

    class DummyResponse(ActionResponse):
        action: DummyAction

    test_player.client.chat.return_value.message.content = json.dumps(
        {"chain_of_thought": ["thought"], "action": {}}
    )
    test_player.client.chat.return_value.prompt_eval_count = 10
    test_player.client.chat.return_value.eval_count = 10

    test_player.messages = []

    response = await test_player.retry_action(DummyResponse, basic_game)
    assert response is not None
    assert isinstance(response.action, DummyAction)


@pytest.mark.asyncio
async def test_retry_action_validation_error(
    test_player: Any, basic_game: GameState
) -> None:

    class DummyAction(BaseAction):
        def validate_semantics(self, game: GameState, player_name: str) -> None:
            pass

    class DummyResponse(ActionResponse):
        action: DummyAction

    # Missing required 'action' field triggers ValidationError
    test_player.client.chat.return_value.message.content = json.dumps(
        {"chain_of_thought": ["thought"]}
    )

    test_player.messages = []

    response = await test_player.retry_action(
        DummyResponse, basic_game, retries=2
    )
    assert response is None
    assert test_player.client.chat.call_count == 2

    # System errors added to history
    assert len(test_player.messages) == 2
    assert test_player.messages[0]["role"] == "system"
    assert "SYSTEM ERROR" in test_player.messages[0]["content"]


@pytest.mark.asyncio
async def test_retry_action_semantic_error(
    test_player: Any, basic_game: GameState
) -> None:

    class DummyAction(BaseAction):
        def validate_semantics(self, game: GameState, player_name: str) -> None:
            raise InvalidActionError("Test semantics failed")  # noqa: TRY003

    class DummyResponse(ActionResponse):
        action: DummyAction

    test_player.client.chat.return_value.message.content = json.dumps(
        {"chain_of_thought": ["thought"], "action": {}}
    )

    test_player.messages = []

    response = await test_player.retry_action(
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

    # Test fallback when compression fails (returns empty string)
    test_player.messages = [
        {"role": "system", "content": "original system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    test_player.client.chat.return_value.message.content = ""
    await test_player._check_and_compress(mock_response_obj, basic_game)
    assert "Oops!" in test_player.messages[1]["content"]
    assert test_player.client.chat.call_count == 2


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

    # late game, but tied
    basic_game.players["Alpha"].gdp = 10
    basic_game.players["Omega"].gdp = 10
    assert not test_player.game_ending_warning(basic_game)


def test_game_ending_warning_does_not_mutate_game(
    test_player: Any, basic_game: GameState
) -> None:
    test_player.name = "Alpha"
    test_player.opponent_name = "Omega"
    basic_game.rules.round_count = 10
    basic_game.current_round = 9
    basic_game.players["Alpha"].gdp = 100

    mandate = PacifistUtopiaMandate()
    basic_game.players["Alpha"].mandates.append(mandate)

    assert len(basic_game.players["Alpha"].completed_mandates) == 0
    test_player.game_ending_warning(basic_game)
    assert len(basic_game.players["Alpha"].completed_mandates) == 0


def test_doomsday_warning(test_player: Any, basic_game: GameState) -> None:
    assert not test_player.doomsday_warning(basic_game)

    mock_rules = MagicMock()
    mock_rules.max_clock_state = 24
    mock_rules.get_doomsday_bids.return_value = ([(1, 2)], [(3,)])

    mock_game = MagicMock()
    mock_game.rules = mock_rules

    # Test boundary condition and multiple states
    for clock_val, expected in [
        (24, "WARNING: "),
        (22, "WARNING: "),
        (12, "NOTE: "),
    ]:
        mock_game.doomsday_clock = clock_val
        assert expected in test_player.doomsday_warning(mock_game)


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


def test_format_ongoing_effects_empty(
    test_player: Any, basic_game: GameState
) -> None:
    basic_game.active_effects = []
    result = test_player.format_ongoing_effects(basic_game)
    assert result == ""


def test_format_ongoing_effects_with_effects(
    test_player: Any, basic_game: GameState
) -> None:
    mock_effect_1 = MagicMock()
    mock_effect_1.title = "Effect 1"
    mock_effect_1.mechanics = "Does a thing"
    mock_effect_1.end_round = 3

    mock_effect_2 = MagicMock()
    mock_effect_2.title = "Effect 2"
    mock_effect_2.mechanics = "Does another thing"
    mock_effect_2.end_round = 5

    basic_game.active_effects = [mock_effect_1, mock_effect_2]

    result = test_player.format_ongoing_effects(basic_game)

    expected = (
        "Ongoing Effects:\n"
        " - Effect 1:\n"
        "   Does a thing\n"
        "   Ongoing until the start of round 3\n"
        "\n"
        " - Effect 2:\n"
        "   Does another thing\n"
        "   Ongoing until the start of round 5\n"
        "\n"
    )
    assert result == expected


@pytest.mark.asyncio
async def test_initial_message(test_player: Any, basic_game: GameState) -> None:
    test_player.retry_action = AsyncMock()

    mock_response = MagicMock()
    mock_response.grand_strategy = MagicMock()
    mock_response.action = InitialMessageAction()
    test_player.retry_action.return_value = mock_response

    action = await test_player.initial_message(basic_game)

    assert test_player.retry_action.call_count == 1
    assert test_player.grand_strategy == mock_response.grand_strategy
    assert action == mock_response.action

    test_player.retry_action.return_value = None
    action = await test_player.initial_message(basic_game)
    assert isinstance(action, InitialMessageAction)


@pytest.mark.asyncio
async def test_message(test_player: Any, basic_game: GameState) -> None:
    test_player.name = "Alpha"
    test_player.opponent_name = "Omega"
    test_player.retry_action = AsyncMock()

    mock_response = MagicMock()
    mock_response.action = MessagingAction()
    test_player.retry_action.return_value = mock_response

    basic_game.current_phase = GamePhase.BIDDING_MESSAGING
    action = await test_player.message(basic_game)
    assert test_player.retry_action.call_count == 1
    assert action == mock_response.action

    basic_game.current_phase = GamePhase.OPERATIONS_MESSAGING
    action = await test_player.message(basic_game)
    assert test_player.retry_action.call_count == 2
    assert action == mock_response.action

    test_player.retry_action.return_value = None
    action = await test_player.message(basic_game)
    assert isinstance(action, MessagingAction)


@pytest.mark.asyncio
async def test_crisis_message(test_player: Any, basic_game: GameState) -> None:
    test_player.retry_action = AsyncMock()

    mock_response = MagicMock()
    mock_response.action = MessagingAction()
    test_player.retry_action.return_value = mock_response

    crisis = StandoffCrisis()
    with patch.object(StandoffCrisis, "additional_prompt", "Custom Prompt!"):
        action = await test_player.crisis_message(basic_game, crisis)
    assert test_player.retry_action.call_count == 1
    assert action == mock_response.action

    test_player.retry_action.return_value = None
    action = await test_player.crisis_message(basic_game, crisis)
    assert isinstance(action, MessagingAction)


@pytest.mark.asyncio
async def test_bid(test_player: Any, basic_game: GameState) -> None:
    test_player.name = "Alpha"
    test_player.opponent_name = "Omega"
    test_player.retry_action = AsyncMock()

    mock_response = MagicMock()
    mock_response.action = BiddingAction(bid=2)
    test_player.retry_action.return_value = mock_response

    action = await test_player.bid(basic_game)
    assert test_player.retry_action.call_count == 1
    assert action == mock_response.action

    test_player.retry_action.return_value = None
    action = await test_player.bid(basic_game)
    assert isinstance(action, BiddingAction)
    assert action.bid == 1


@pytest.mark.asyncio
async def test_operations(test_player: Any, basic_game: GameState) -> None:
    test_player.name = "Alpha"
    test_player.opponent_name = "Omega"
    test_player.retry_action = AsyncMock()

    mock_response = MagicMock()
    mock_response.action = OperationsAction(operations=["op1"])
    test_player.retry_action.return_value = mock_response

    action = await test_player.operations(basic_game)
    assert test_player.retry_action.call_count == 1
    assert action == mock_response.action

    test_player.retry_action.return_value = None
    action = await test_player.operations(basic_game)
    assert isinstance(action, OperationsAction)
    assert action.operations == []


@pytest.mark.asyncio
async def test_crisis(test_player: Any, basic_game: GameState) -> None:
    test_player.retry_action = AsyncMock()

    mock_response = MagicMock()
    mock_response.action = BiddingAction(bid=2)
    test_player.retry_action.return_value = mock_response

    crisis = StandoffCrisis()
    with patch.object(StandoffCrisis, "additional_prompt", "Custom Prompt!"):
        action = await test_player.crisis(basic_game, crisis)
    assert test_player.retry_action.call_count == 1
    assert action == mock_response.action

    test_player.retry_action.return_value = None
    action = await test_player.crisis(basic_game, crisis)
    assert isinstance(action, BaseAction)


@pytest.mark.asyncio
async def test_game_over(
    test_player: Any, basic_game: GameState, tmp_path: Any
) -> None:
    test_player.log_base = tmp_path / "test_player"
    test_player.client.chat.return_value.message.content = "AAR Output"

    await test_player.game_over(
        basic_game, test_player.name, GameOverReason.ECONOMIC_VICTORY
    )
    assert test_player.client.chat.call_count == 1
    assert "AAR Output" in test_player.messages[-1]["content"]

    test_player.client.chat.reset_mock()
    await test_player.game_over(
        basic_game, "Opponent", GameOverReason.WORLD_DESTROYED
    )
    assert test_player.client.chat.call_count == 1

    assert test_player.log_base.with_suffix(".messages.gz").exists()

    test_player.log_base = None
    await test_player.game_over(
        basic_game, "Opponent", GameOverReason.WORLD_DESTROYED
    )


def test_my_strategy(test_player: Any) -> None:
    assert test_player.my_strategy() == ""

    test_player.grand_strategy = GrandStrategy(
        prohibited_actions=[],
        core_loop="loop",
        clock_management="clock",
        contingency_plan="plan",
    )

    strategy_text = test_player.my_strategy()
    assert "loop" in strategy_text
    assert "clock" in strategy_text

    strategy_text_compressed = test_player.my_strategy(for_compression=True)
    assert "original grand strategy" in strategy_text_compressed
