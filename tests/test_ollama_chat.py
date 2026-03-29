"""Tests for the Ollama chat script."""

from __future__ import annotations

from pathlib import Path

from mad_world.ollama_chat import run_chat


def test_run_chat_invalid_file() -> None:
    """Test that run_chat exits on invalid file."""
    assert (run_chat(Path("non_existent_file.gz"), "model")) == 1
