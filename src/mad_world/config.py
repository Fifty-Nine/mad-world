"""Models and logic relating to command-line (or other) configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PlayerKind(StrEnum):
    LLM = "llm"
    HUMAN = "human"
    TRIVIAL = "trivial"


class LLMPlayerConfig(BaseModel):
    kind: Literal[PlayerKind.LLM] = PlayerKind.LLM
    name: str
    model: str
    persona: str | None = None
    temperature: float = 0.8
    context_size: int = 2**17
    token_limit: int = 2**13
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64


class HumanPlayerConfig(BaseModel):
    kind: Literal[PlayerKind.HUMAN] = PlayerKind.HUMAN
    name: str


class TrivialPlayerConfig(BaseModel):
    kind: Literal[PlayerKind.TRIVIAL] = PlayerKind.TRIVIAL
    name: str
    bot_name: str


PlayerConfig = Annotated[
    LLMPlayerConfig | HumanPlayerConfig | TrivialPlayerConfig,
    Field(discriminator="kind"),
]
