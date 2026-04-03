from __future__ import annotations

import logging
import re
from argparse import ArgumentError
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_world.__main__ import (
    amain,
    coerce_bool_response,
    create_log_session_dir,
    get_player,
    main,
    prompt_bool,
    prompt_bool_once,
    random_persona,
    run_game,
    setup_logging,
)
from mad_world.config import (
    HumanPlayerConfig,
    LLMPlayerConfig,
    TrivialPlayerConfig,
)
from mad_world.enums import GameOverReason
from mad_world.human_player import HumanPlayer
from mad_world.ollama_player import OllamaPlayer
from mad_world.trivial_players import CrazyIvan

if TYPE_CHECKING:
    from pathlib import Path


def test_random_persona() -> None:
    persona = random_persona()
    assert isinstance(persona, str)
    assert " " in persona
    assert re.match(r"^([A-Z][a-z]+) ([A-Z][a-z]+)$", persona) is not None


def test_create_log_session_dir(tmp_path: Path) -> None:
    timestamp = datetime(2026, 3, 25, 20, 50, 23)
    alpha_config = LLMPlayerConfig(
        name="Alpha",
        model="ModelA",
        persona="PersonaA",
    )
    omega_config = LLMPlayerConfig(
        name="Omega",
        model="ModelB",
        persona="PersonaB",
    )
    log_dir = create_log_session_dir(
        tmp_path,
        alpha_config,
        omega_config,
        timestamp=timestamp,
    )

    expected_name = (
        "Alpha-PersonaA-ModelA-vs-Omega-PersonaB-ModelB.2026-03-25T20-50-23"
    )
    assert log_dir.name == expected_name
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_create_log_session_dir_long_personas(tmp_path: Path) -> None:
    timestamp = datetime(2026, 3, 25, 20, 50, 23)
    alpha_config = LLMPlayerConfig(
        name="Alpha",
        model="ModelA",
        persona="PersonaA\nExtra persona text",
    )
    omega_config = LLMPlayerConfig(
        name="Omega",
        model="ModelB",
        persona="PersonaB\n" + "\n".join(["  another line"] * 100),
    )
    log_dir = create_log_session_dir(
        tmp_path,
        alpha_config,
        omega_config,
        timestamp=timestamp,
    )

    expected_name = (
        "Alpha-PersonaA-ModelA-vs-Omega-PersonaB-ModelB.2026-03-25T20-50-23"
    )
    assert log_dir.name == expected_name
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_setup_logging(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    logger = setup_logging("INFO", log_dir)

    # Check if handlers were added
    handler_types = [type(h) for h in logger.handlers]
    assert logging.FileHandler in handler_types
    assert logging.StreamHandler in handler_types

    debug_file = log_dir / "debug.txt"
    log_file = log_dir / "log.txt"
    assert debug_file.exists()
    assert log_file.exists()


def test_get_player(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    player_human = get_player(
        HumanPlayerConfig(name="Alpha"),
        "Omega",
        log_dir,
        logging.getLogger(__name__),
    )
    assert isinstance(player_human, HumanPlayer)

    # Test trivial players
    player_crazy = get_player(
        TrivialPlayerConfig(name="Alpha", bot_name="CrazyIvan"),
        "Omega",
        log_dir,
        logging.getLogger(__name__),
    )

    assert isinstance(player_crazy, CrazyIvan)
    assert player_crazy.name == "Alpha"

    # Test LLM player
    player_ollama = get_player(
        LLMPlayerConfig(
            name="Alpha",
            model="gemma3:12b",
            persona="Persona",
        ),
        "Omega",
        log_dir,
        logging.getLogger(__name__),
    )
    assert isinstance(player_ollama, OllamaPlayer)
    assert player_ollama.model == "gemma3:12b"

    # Test unknown bot
    with pytest.raises(ValueError, match="Unknown trivial player bot name"):
        get_player(
            TrivialPlayerConfig(name="Alpha", bot_name="Unknown"),
            "Omega",
            log_dir,
            logging.getLogger(__name__),
        )


@pytest.mark.asyncio
@patch("mad_world.__main__.game_loop", new_callable=AsyncMock)
async def test_amain_success(mock_game_loop: AsyncMock, tmp_path: Path) -> None:
    # Mock game_loop return value
    mock_state = MagicMock()
    mock_state.current_round = 10
    mock_state.players = {}
    mock_game_loop.return_value = (
        "Alpha",
        GameOverReason.ECONOMIC_VICTORY,
        mock_state,
    )

    alpha_config = HumanPlayerConfig(name="Alpha")
    omega_config = HumanPlayerConfig(name="Omega")

    await amain(
        alpha_config,
        omega_config,
        log_dir=tmp_path,
        logger=logging.getLogger(__name__),
    )

    assert mock_game_loop.called


@patch("mad_world.__main__.amain", new_callable=AsyncMock)
@patch("mad_world.__main__.shutil.rmtree")
def test_run_game_keyboard_interrupt(
    mock_rmtree: MagicMock,
    mock_amain: AsyncMock,
    tmp_path: Path,
) -> None:
    mock_amain.side_effect = KeyboardInterrupt()

    alpha_config = HumanPlayerConfig(name="Alpha")
    omega_config = HumanPlayerConfig(name="Omega")

    run_game(
        alpha_config,
        omega_config,
        log_dir=tmp_path,
    )

    assert mock_rmtree.called


@patch("mad_world.__main__.amain", new_callable=AsyncMock)
def test_run_game_defaults(mock_amain: AsyncMock, tmp_path: Path) -> None:
    run_game()
    assert mock_amain.called

    kwargs = mock_amain.call_args.kwargs
    alpha = kwargs["alpha_config"]
    omega = kwargs["omega_config"]

    assert isinstance(alpha, LLMPlayerConfig)
    assert isinstance(omega, LLMPlayerConfig)
    assert alpha.persona is not None
    assert omega.persona is not None


@patch("mad_world.__main__.amain", new_callable=AsyncMock)
def test_run_game_custom(mock_amain: AsyncMock, tmp_path: Path) -> None:
    alpha = HumanPlayerConfig(name="Alpha")
    omega = TrivialPlayerConfig(name="Omega", bot_name="CrazyIvan")
    run_game(alpha=alpha, omega=omega, log_dir=tmp_path, verbosity="INFO")
    assert mock_amain.called


def test_cli_parsing(tmp_path: Path) -> None:
    argv = [
        "mad_world",
        "--alpha.kind",
        "human",
        "--alpha.name",
        "HumAlpha",
        "--omega.kind",
        "trivial",
        "--omega.name",
        "TrivOmega",
        "--omega.bot_name",
        "Pacifist",
        "--log_dir",
        str(tmp_path),
    ]
    with (
        patch("mad_world.__main__.amain", new_callable=AsyncMock) as mock_amain,
        patch("sys.argv", argv),
    ):
        main()
        assert mock_amain.called
        kwargs = mock_amain.call_args.kwargs
        alpha = kwargs["alpha_config"]
        omega = kwargs["omega_config"]

        assert isinstance(alpha, HumanPlayerConfig)
        assert alpha.name == "HumAlpha"
        assert isinstance(omega, TrivialPlayerConfig)
        assert omega.name == "TrivOmega"
        assert omega.bot_name == "Pacifist"


def test_cli_parsing_invalid() -> None:
    argv_1 = [
        "mad_world",
        "--alpha.kind",
        "human",
        "--alpha.name",
        "HumAlpha",
        "--alpha.model",
        "gemma3:12b",
    ]
    with (
        patch("sys.argv", argv_1),
        pytest.raises((SystemExit, ArgumentError)),
    ):
        main()


def test_coerce_bool_response() -> None:
    assert coerce_bool_response("", default_val=True) is True
    assert coerce_bool_response("  \n", default_val=False) is False
    assert coerce_bool_response("n", default_val=True) is False
    assert coerce_bool_response("NO", default_val=True) is False
    assert coerce_bool_response("y", default_val=False) is True
    assert coerce_bool_response("YES", default_val=False) is True
    assert coerce_bool_response("YE", default_val=False) is True
    assert coerce_bool_response("invalid", default_val=True) is None


@pytest.mark.asyncio
async def test_prompt_bool_once_success() -> None:
    session = AsyncMock()
    session.prompt_async.return_value = "yes"
    result = await prompt_bool_once("prompt", session, default_val=False)
    assert result is True


@pytest.mark.asyncio
async def test_prompt_bool_once_unrecognized() -> None:
    session = AsyncMock()
    session.prompt_async.return_value = "foo"
    result = await prompt_bool_once("prompt", session, default_val=False)
    assert result == "foo"


@pytest.mark.asyncio
async def test_prompt_bool_retry() -> None:
    patch_target = "mad_world.__main__.prompt_bool_once"
    with patch(patch_target, new_callable=AsyncMock) as mock_pbo:
        mock_pbo.side_effect = ["invalid", True]
        result = await prompt_bool("prompt", default_val=False)
        assert result is True
        assert mock_pbo.call_count == 2
