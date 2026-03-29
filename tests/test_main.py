import logging
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from mad_world.__main__ import (
    amain,
    create_log_session_dir,
    get_player,
    main,
    random_persona,
    setup_logging,
)
from mad_world.enums import GameOverReason
from mad_world.human_player import HumanPlayer
from mad_world.ollama_player import OllamaPlayer
from mad_world.trivial_players import CrazyIvan, Pacifist


def test_random_persona() -> None:
    persona = random_persona()
    assert isinstance(persona, str)
    assert " " in persona
    assert re.match(r"^([A-Z][a-z]+) ([A-Z][a-z]+)$", persona) is not None


def test_create_log_session_dir(tmp_path: Path) -> None:
    timestamp = datetime(2026, 3, 25, 20, 50, 23)
    log_dir = create_log_session_dir(
        tmp_path,
        "Alpha",
        "PersonaA",
        "ModelA",
        "Omega",
        "PersonaB",
        "ModelB",
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

    setup_logging(logging.INFO, log_dir)

    logger = logging.getLogger()
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
        "Alpha", "Omega", "human", "Persona", 0.0, 1, 1, log_dir
    )
    assert isinstance(player_human, HumanPlayer)

    # Test trivial players
    player_crazy = get_player(
        "Alpha", "Omega", "CrazyIvan", "Persona", 0.0, 1, 1, log_dir
    )

    assert isinstance(player_crazy, CrazyIvan)
    assert player_crazy.name == "Alpha"

    player_crazy_snake = get_player(
        "Alpha", "Omega", "crazy_ivan", "Persona", 0.0, 1, 1, log_dir
    )
    assert isinstance(player_crazy_snake, CrazyIvan)
    assert player_crazy_snake.name == "Alpha"

    player_crazy_kebab = get_player(
        "Alpha", "Omega", "crazy-ivan", "Persona", 0.0, 1, 1, log_dir
    )
    assert isinstance(player_crazy_kebab, CrazyIvan)

    player_pacifist = get_player(
        "Alpha", "Omega", "pacifist", "Persona", 0.0, 1, 1, log_dir
    )

    assert isinstance(player_pacifist, Pacifist)

    player_ollama = get_player(
        "Alpha", "Omega", "gemma3:12b", "Persona", 0.0, 1, 1, log_dir
    )
    assert isinstance(player_ollama, OllamaPlayer)
    assert player_ollama.model == "gemma3:12b"


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

    await amain(
        "Alpha",
        "human",
        "PersonaA",
        0.0,
        1,
        1,
        "Omega",
        "human",
        "PersonaB",
        0.0,
        1,
        1,
        log_dir_base=tmp_path,
        verbosity=logging.INFO,
    )

    assert mock_game_loop.called


@pytest.mark.asyncio
@patch("mad_world.__main__.game_loop", new_callable=AsyncMock)
@patch("mad_world.__main__.shutil.rmtree")
async def test_amain_keyboard_interrupt(
    mock_rmtree: MagicMock, mock_game_loop: AsyncMock, tmp_path: Path
) -> None:
    mock_game_loop.side_effect = KeyboardInterrupt()

    await amain(
        "Alpha",
        "human",
        "PersonaA",
        0.0,
        1,
        1,
        "Omega",
        "human",
        "PersonaB",
        0.0,
        1,
        1,
        log_dir_base=tmp_path,
        verbosity=logging.INFO,
    )

    assert mock_rmtree.called


@patch("mad_world.__main__.amain", new_callable=MagicMock)
def test_main_cli(mock_amain: MagicMock, tmp_path: Path) -> None:
    # We need to mock asyncio.run because amain is async
    with patch("mad_world.__main__.asyncio.run") as mock_run:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--alpha-name",
                "A",
                "--omega-name",
                "O",
                "--log-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert mock_run.called
