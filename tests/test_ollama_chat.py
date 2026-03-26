"""Tests for the Ollama chat script."""

from pathlib import Path

import pytest

from mad_world.ollama_chat import run_chat


@pytest.mark.asyncio
async def test_run_chat_invalid_file() -> None:
    """Test that run_chat exits on invalid file."""
    with pytest.raises(SystemExit):
        await run_chat(Path("non_existent_file.gz"), "model")
