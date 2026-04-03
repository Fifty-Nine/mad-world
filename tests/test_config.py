"""Tests for the mad_world.config module."""

from __future__ import annotations

import pytest

from mad_world.config import (
    HumanPlayerConfig,
    LLMPlayerConfig,
    TrivialPlayerConfig,
)


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
