"""Tests for the Mad World log summarizer script."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import litellm
from click.testing import CliRunner

from mad_world.log_summarizer import main, summarize_log

if TYPE_CHECKING:
    import pytest


def test_summarize_log_file_not_found() -> None:
    """Test summarize_log with a non-existent file."""
    assert summarize_log(Path("non_existent_file.txt"), "model") == 1


def test_summarize_log_empty_file(tmp_path: Path) -> None:
    """Test summarize_log with an empty file."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    assert summarize_log(empty_file, "model") == 1


def test_summarize_log_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test summarize_log normal execution."""
    log_file = tmp_path / "log.txt"
    log_file.write_text("Line 1\nLine 2\n")

    client_mock = MagicMock()
    client_mock.return_value = [
        type(
            "obj",
            (),
            {
                "choices": [
                    type(
                        "obj",
                        (),
                        {"delta": type("obj", (), {"content": "Summary "})},
                    )()
                ]
            },
        )(),
        type(
            "obj",
            (),
            {
                "choices": [
                    type(
                        "obj",
                        (),
                        {"delta": type("obj", (), {"content": "part 2."})},
                    )()
                ]
            },
        )(),
    ]

    with patch(
        "mad_world.log_summarizer.litellm.completion", side_effect=client_mock
    ):
        result = summarize_log(log_file, "test_model", api_base="http://mock")

    assert result == 0
    client_mock.assert_called_once()

    # Check that prompt was sent correctly
    kwargs = client_mock.call_args.kwargs
    assert kwargs["model"] == "ollama/test_model"
    assert kwargs["stream"] is True
    assert "0001: Line 1" in kwargs["messages"][0]["content"]
    assert "0002: Line 2" in kwargs["messages"][0]["content"]

    captured = capsys.readouterr()
    assert "Summary part 2." in captured.out


def test_summarize_log_response_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test summarize_log when APIConnectionError is raised."""
    log_file = tmp_path / "log.txt"
    log_file.write_text("Line 1")

    client_mock = MagicMock()
    client_mock.side_effect = litellm.exceptions.APIConnectionError(
        "mock error", "mock", "mock"
    )

    with patch(
        "mad_world.log_summarizer.litellm.completion", side_effect=client_mock
    ):
        result = summarize_log(log_file, "model")

    assert result == 1
    captured = capsys.readouterr()
    assert "Failed to communicate with LLM API: " in captured.err


def test_main_success(tmp_path: Path) -> None:
    """Test main command."""
    log_file = tmp_path / "log.txt"
    log_file.write_text("Test line")

    runner = CliRunner()
    with patch(
        "mad_world.log_summarizer.summarize_log", return_value=0
    ) as mock_summarize:
        result = runner.invoke(
            main,
            [str(log_file), "--model", "mymodel", "--context-size", "1000"],
        )
        assert result.exit_code == 0
        mock_summarize.assert_called_once_with(
            log_file,
            "mymodel",
            None,
            1000,
        )


def test_main_failure(tmp_path: Path) -> None:
    """Test main command when summarizer fails."""
    log_file = tmp_path / "log.txt"
    log_file.write_text("Test line")

    runner = CliRunner()
    with patch("mad_world.log_summarizer.summarize_log", return_value=42):
        result = runner.invoke(main, [str(log_file)])
        assert result.exit_code == 42


def test_main_keyboard_interrupt(tmp_path: Path) -> None:
    """Test main handling KeyboardInterrupt."""
    log_file = tmp_path / "log.txt"
    log_file.write_text("Test line")

    runner = CliRunner()
    with patch(
        "mad_world.log_summarizer.summarize_log", side_effect=KeyboardInterrupt
    ):
        result = runner.invoke(main, [str(log_file)])
        assert result.exit_code == 0


def test_main_eof_error(tmp_path: Path) -> None:
    """Test main handling EOFError."""
    log_file = tmp_path / "log.txt"
    log_file.write_text("Test line")

    runner = CliRunner()
    with patch("mad_world.log_summarizer.summarize_log", side_effect=EOFError):
        result = runner.invoke(main, [str(log_file)])
        assert result.exit_code == 0
