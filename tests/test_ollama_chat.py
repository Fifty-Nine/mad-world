"""Tests for the Ollama chat script."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import ollama
import pytest
from click.testing import CliRunner
from prompt_toolkit import PromptSession

from mad_world import ollama_chat
from mad_world.config import LLMParams
from mad_world.ollama_chat import (
    QuitProgram,
    exit_loop,
    load_image,
    main,
    print_slash_help,
    process_slash_command,
    prompt_loop,
    run_chat,
)


def test_exit_loop() -> None:
    """Test that exit_loop raises QuitProgram."""
    runner = CliRunner()
    with pytest.raises(QuitProgram) as excinfo:
        runner.invoke(exit_loop, catch_exceptions=False)
    assert excinfo.value.rc == 0


def test_print_slash_help_no_command() -> None:
    """Test print_slash_help with no command provided."""
    runner = CliRunner()
    result = runner.invoke(print_slash_help)
    assert result.exit_code == 0
    assert "quit: Exit the application." in result.output
    assert "help: Print the list of available commands" in result.output


def test_print_slash_help_valid_command() -> None:
    """Test print_slash_help with a valid command."""
    runner = CliRunner()
    result = runner.invoke(print_slash_help, ["quit"])
    assert result.exit_code == 0
    assert "Exit the application." in result.output


def test_print_slash_help_invalid_command() -> None:
    """Test print_slash_help with an invalid command."""
    runner = CliRunner()
    result = runner.invoke(print_slash_help, ["nonexistent_cmd"])
    assert result.exit_code == 0
    assert "Error: No such command: 'nonexistent_cmd'" in result.output


def test_load_image(tmp_path: Path) -> None:
    """Test load_image command."""
    runner = CliRunner()
    img_file = tmp_path / "img.bin"
    img_file.write_bytes(b"dummy image data")

    old_pending = list(ollama_chat.pending_images)
    ollama_chat.pending_images.clear()
    try:
        result = runner.invoke(load_image, [str(img_file)])
        assert result.exit_code == 0
        assert len(ollama_chat.pending_images) == 1
        assert ollama_chat.pending_images[0] == b"dummy image data"
    finally:
        ollama_chat.pending_images = old_pending


def test_process_slash_command_not_slash() -> None:
    """Test process_slash_command with non-slash input."""
    assert process_slash_command("hello world") is False


def test_process_slash_command_valid() -> None:
    """Test process_slash_command with valid command."""
    # We can invoke a valid command, e.g. /help
    with patch("mad_world.ollama_chat.slash_commands.main") as mock_main:
        assert process_slash_command("/help quit") is True
        mock_main.assert_called_once_with(
            args=["help", "quit"], standalone_mode=False
        )


def test_process_slash_command_invalid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test process_slash_command with invalid command."""
    # Calling invalid cmd triggers NoSuchCommand / ClickException
    assert process_slash_command("/invalid_command_xyz") is True
    captured = capsys.readouterr()
    assert "Error: No such command 'invalid_command_xyz'." in captured.out


def test_process_slash_command_quit() -> None:
    """Test process_slash_command with quit command."""
    with pytest.raises(QuitProgram):
        process_slash_command("/quit")


def test_prompt_loop_empty_input() -> None:
    """Test prompt_loop with empty input."""
    session_mock = MagicMock()
    session_mock.prompt.return_value = "  "
    client_mock = MagicMock()
    messages: list[dict[str, str]] = []

    prompt_loop(session_mock, client_mock, "model", messages, LLMParams())
    assert not messages
    client_mock.chat.assert_not_called()


def test_prompt_loop_slash_command() -> None:
    """Test prompt_loop with slash command."""
    session_mock = MagicMock()
    session_mock.prompt.return_value = "/help"
    client_mock = MagicMock()
    messages: list[dict[str, str]] = []

    with patch(
        "mad_world.ollama_chat.process_slash_command", return_value=True
    ):
        prompt_loop(session_mock, client_mock, "model", messages, LLMParams())

    assert not messages
    client_mock.chat.assert_not_called()


def test_prompt_loop_valid_input(capsys: pytest.CaptureFixture[str]) -> None:
    """Test prompt_loop with valid input."""
    session_mock = MagicMock()
    session_mock.prompt.return_value = "Hello"
    client_mock = MagicMock()
    client_mock.chat.return_value = [
        {"message": {"content": "Hi "}},
        {"message": {"content": "there!"}},
    ]
    messages: list[dict[str, str]] = []

    old_pending = list(ollama_chat.pending_images)
    ollama_chat.pending_images.clear()
    ollama_chat.pending_images.append(b"image1")

    try:
        prompt_loop(session_mock, client_mock, "model", messages, LLMParams())
    finally:
        ollama_chat.pending_images = old_pending

    assert len(messages) == 2
    assert messages[0] == {
        "role": "user",
        "content": "Hello",
        "images": [b"image1"],
    }
    assert messages[1] == {"role": "assistant", "content": "Hi there!"}

    # Check that images were cleared
    assert len(ollama_chat.pending_images) == 0

    client_mock.chat.assert_called_once()
    kwargs = client_mock.chat.call_args.kwargs
    assert kwargs["model"] == "model"
    assert kwargs["stream"] is True
    assert kwargs["messages"] is messages

    captured = capsys.readouterr()
    assert "Assistant > " in captured.out
    assert "Hi there!" in captured.out


def test_run_chat_invalid_file() -> None:
    """Test run_chat with invalid file."""
    assert run_chat(Path("non_existent_file.gz"), "model") == 1


def test_run_chat_bad_json(tmp_path: Path) -> None:
    """Test run_chat with bad JSON."""
    log_file = tmp_path / "bad.json.gz"
    with gzip.open(log_file, "wt") as f:
        f.write("{bad json")
    assert run_chat(log_file, "model") == 1


def test_run_chat_not_a_list(tmp_path: Path) -> None:
    """Test run_chat when JSON is not a list."""
    log_file = tmp_path / "dict.json.gz"
    with gzip.open(log_file, "wt") as f:
        json.dump({"role": "user"}, f)
    assert run_chat(log_file, "model") == 1


def test_run_chat_success(tmp_path: Path) -> None:
    """Test run_chat normal loop execution."""
    log_file = tmp_path / "log.json.gz"
    with gzip.open(log_file, "wt") as f:
        json.dump([{"role": "user", "content": "hi"}], f)

    # We patch prompt_loop to raise an exception to exit the infinite loop
    with (
        patch("mad_world.ollama_chat.ollama.Client"),
        patch("mad_world.ollama_chat.PromptSession"),
        patch("mad_world.ollama_chat.prompt_loop", side_effect=QuitProgram(0)),
        pytest.raises(QuitProgram),
    ):
        run_chat(log_file, "model")


def test_main_success(tmp_path: Path) -> None:
    """Test main command."""
    log_file = tmp_path / "log.json.gz"
    with gzip.open(log_file, "wt") as f:
        json.dump([], f)

    runner = CliRunner()
    with patch("mad_world.ollama_chat.run_chat", return_value=0) as mock_run:
        result = runner.invoke(main, [str(log_file), "--model", "mymodel"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            log_file, "mymodel", host=None, settings=None
        )


def test_main_quit_program(tmp_path: Path) -> None:
    """Test main handling QuitProgram."""
    log_file = tmp_path / "log.json.gz"
    with gzip.open(log_file, "wt") as f:
        json.dump([], f)

    runner = CliRunner()
    with patch("mad_world.ollama_chat.run_chat", side_effect=QuitProgram(42)):
        result = runner.invoke(main, [str(log_file)])
        assert result.exit_code == 42


def test_main_keyboard_interrupt(tmp_path: Path) -> None:
    """Test main handling KeyboardInterrupt."""
    log_file = tmp_path / "log.json.gz"
    with gzip.open(log_file, "wt") as f:
        json.dump([], f)

    runner = CliRunner()
    with patch("mad_world.ollama_chat.run_chat", side_effect=KeyboardInterrupt):
        result = runner.invoke(main, [str(log_file)])
        assert result.exit_code == 0


def test_main_eof_error(tmp_path: Path) -> None:
    """Test main handling EOFError."""
    log_file = tmp_path / "log.json.gz"
    with gzip.open(log_file, "wt") as f:
        json.dump([], f)

    runner = CliRunner()
    with patch("mad_world.ollama_chat.run_chat", side_effect=EOFError):
        result = runner.invoke(main, [str(log_file)])
        assert result.exit_code == 0


def test_prompt_loop_response_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Test prompt_loop when ollama.ResponseError is raised."""
    session_mock = MagicMock(spec=PromptSession)
    session_mock.prompt.return_value = "Hello"
    client_mock = MagicMock(spec=ollama.Client)

    # Case 1: Exception raised immediately
    client_mock.chat.side_effect = ollama.ResponseError("err1")
    messages: list[dict[str, Any]] = []

    prompt_loop(session_mock, client_mock, "model", messages, LLMParams())

    captured = capsys.readouterr()
    assert "Failed to communicate with ollama: err1" in captured.err
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    # Case 2: Exception raised after partial success
    def chat_generator_partial_success(*args: Any, **kwargs: Any) -> Any:
        yield {"message": {"content": "Partial "}}
        raise ollama.ResponseError("err2")

    client_mock.chat.side_effect = chat_generator_partial_success
    messages.clear()

    prompt_loop(session_mock, client_mock, "model", messages, LLMParams())

    captured = capsys.readouterr()
    assert "Partial " in captured.out
    assert "Failed to communicate with ollama: err2" in captured.err
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_load_settings_not_found(tmp_path: Path) -> None:
    """Test load_settings when file is not found."""
    settings_file = tmp_path / "missing.json"
    params, model = ollama_chat.load_settings(settings_file)
    assert model is None
    assert params.temperature == 0.8


def test_load_settings_bad_json(tmp_path: Path) -> None:
    """Test load_settings with invalid JSON."""
    settings_file = tmp_path / "bad.json"
    settings_file.write_text("{bad json")
    _params, model = ollama_chat.load_settings(settings_file)
    assert _params == LLMParams()
    assert model is None


def test_load_settings_is_a_dir(tmp_path: Path) -> None:
    settings_file = tmp_path / "dir"
    settings_file.mkdir()
    _params, model = ollama_chat.load_settings(settings_file)
    assert _params == LLMParams()
    assert model is None


def test_load_settings_perms(tmp_path: Path) -> None:
    settings_file = tmp_path / "dir"
    settings_file.write_text("{}")
    settings_file.chmod(0)
    _params, model = ollama_chat.load_settings(settings_file)
    assert _params == LLMParams()
    assert model is None


def test_load_settings_success(tmp_path: Path) -> None:
    """Test load_settings with valid JSON."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        '{"model": "test-model", "params": {"temperature": 0.5}}'
    )
    params, model = ollama_chat.load_settings(settings_file)
    assert model == "test-model"
    assert params.temperature == 0.5


def test_run_chat_derive_settings_path(tmp_path: Path) -> None:
    """Test run_chat derives settings path from .messages.gz."""
    log_file = tmp_path / "game.messages.gz"
    settings_file = tmp_path / "game.model-settings.json"
    settings_file.write_text('{"model": "derived-model", "params": {}}')

    with gzip.open(log_file, "wt") as f:
        json.dump([], f)

    with (
        patch("mad_world.ollama_chat.ollama.Client"),
        patch("mad_world.ollama_chat.PromptSession"),
        patch("mad_world.ollama_chat.prompt_loop", side_effect=QuitProgram(0)),
        pytest.raises(QuitProgram),
    ):
        run_chat(log_file)


def test_run_chat_derive_settings_path_no_gz(tmp_path: Path) -> None:
    """Test run_chat derives settings path from other extension."""
    log_file = tmp_path / "game.log"
    settings_file = tmp_path / "game.model-settings.json"
    settings_file.write_text('{"model": "derived-model", "params": {}}')

    with gzip.open(log_file, "wt") as f:
        json.dump([], f)

    with (
        patch("mad_world.ollama_chat.ollama.Client"),
        patch("mad_world.ollama_chat.PromptSession"),
        patch("mad_world.ollama_chat.prompt_loop", side_effect=QuitProgram(0)),
        pytest.raises(QuitProgram),
    ):
        run_chat(log_file)


def test_run_chat_explicit_settings(tmp_path: Path) -> None:
    """Test run_chat with explicit settings path."""
    log_file = tmp_path / "game.log"
    settings_file = tmp_path / "explicit-settings.json"
    settings_file.write_text('{"model": "derived-model", "params": {}}')

    with gzip.open(log_file, "wt") as f:
        json.dump([], f)

    with (
        patch("mad_world.ollama_chat.ollama.Client"),
        patch("mad_world.ollama_chat.PromptSession"),
        patch("mad_world.ollama_chat.prompt_loop", side_effect=QuitProgram(0)),
        pytest.raises(QuitProgram),
    ):
        run_chat(log_file, settings=settings_file)
