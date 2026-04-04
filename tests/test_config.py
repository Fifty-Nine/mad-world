"""Tests for the mad_world.config module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from mad_world.config import (
    HumanPlayerConfig,
    LLMPlayerConfig,
    TrivialPlayerConfig,
    _load_model_defaults,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def llm_config() -> LLMPlayerConfig:
    return LLMPlayerConfig(name="name", persona="persona", model="model")


@pytest.fixture
def human_config() -> HumanPlayerConfig:
    return HumanPlayerConfig(name="da human guy")


@pytest.fixture
def trivial_config() -> TrivialPlayerConfig:
    return TrivialPlayerConfig(name="robot", bot_name="CrazyIvan")


def test_llm_player_config_summary(llm_config: LLMPlayerConfig) -> None:
    assert llm_config.summarize() == "name - persona (model)"
    assert llm_config.file_name() == "name-persona-model"


def test_human_player_config_summary(human_config: HumanPlayerConfig) -> None:
    assert human_config.summarize() == "da human guy (Human)"
    assert human_config.file_name() == "da human guy-Human"


def test_trivial_player_config_summary(
    trivial_config: TrivialPlayerConfig,
) -> None:
    assert trivial_config.summarize() == "robot (CrazyIvan bot)"
    assert trivial_config.file_name() == "robot-Bot-CrazyIvan"


@pytest.fixture(autouse=True)
def model_defaults_cache_cleanup() -> Generator[None, None, None]:
    """Fixture that ensures the model defaults cache is
    reloaded before each test in this file."""
    _load_model_defaults.cache_clear()
    yield
    _load_model_defaults.cache_clear()


def test_load_model_defaults_file_not_found() -> None:
    with patch("mad_world.config.files") as mock_files:
        mock_files.return_value.joinpath.return_value.read_text.side_effect = (
            FileNotFoundError
        )
        assert _load_model_defaults() == {}


@patch("mad_world.config.files")
def test_load_model_defaults_json_decode_error(mock_files: MagicMock) -> None:
    mock_files.return_value.joinpath.return_value.read_text.side_effect = (
        json.JSONDecodeError("msg", "doc", 0)
    )
    with pytest.raises(json.JSONDecodeError):
        _load_model_defaults()


@patch("mad_world.config.files")
def test_load_model_defaults_not_dict(mock_files: MagicMock) -> None:
    mock_files.return_value.joinpath.return_value.read_text.return_value = "[]"
    with pytest.raises(ValidationError):
        _load_model_defaults()


def test_llm_player_config_no_defaults() -> None:
    config = LLMPlayerConfig(name="name", model="unknown_model")
    assert config.params.temperature == 0.8
    assert config.params.context_size == 2**15
    assert config.params.token_limit == 2**13
    assert config.params.repeat_penalty == 1.1
    assert config.params.repeat_last_n == 64


@patch("mad_world.config.files")
def test_llm_player_config_custom_defaults(mock_files: MagicMock) -> None:
    mock_files.return_value.joinpath.return_value.read_text.return_value = (
        '{"my_model": {"temperature":-1.0}}'
    )
    config = LLMPlayerConfig(name="name", model="my_model")
    assert config.params.temperature == -1.0
    assert config.params.context_size == 2**15
    assert config.params.token_limit == 2**13
    assert config.params.repeat_penalty == 1.1
    assert config.params.repeat_last_n == 64


@patch("mad_world.config.files")
def test_load_model_defaults_invalid_schema(mock_files: MagicMock) -> None:
    mock_files.return_value.joinpath.return_value.read_text.return_value = (
        '{"gemma": {"temperature": "not a float"}}'
    )
    with pytest.raises(ValidationError):
        _load_model_defaults()
