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

    def _simple_persona(self) -> str:
        return (self.persona or "").partition("\n")[0].strip()

    def summarize(self) -> str:
        return f"{self.name} - {self._simple_persona()} ({self.model})"

    def file_name(self) -> str:
        return f"{self.name}-{self._simple_persona()}-{self.model}"


class HumanPlayerConfig(BaseModel):
    kind: Literal[PlayerKind.HUMAN] = PlayerKind.HUMAN
    name: str

    def summarize(self) -> str:
        return f"{self.name} (Human)"

    def file_name(self) -> str:
        return f"{self.name}-Human"


class TrivialPlayerConfig(BaseModel):
    kind: Literal[PlayerKind.TRIVIAL] = PlayerKind.TRIVIAL
    name: str
    bot_name: str

    def summarize(self) -> str:
        return f"{self.name} ({self.bot_name} bot)"

    def file_name(self) -> str:
        return f"{self.name}-Bot-{self.bot_name}"


PlayerConfig = Annotated[
    LLMPlayerConfig | HumanPlayerConfig | TrivialPlayerConfig,
    Field(discriminator="kind"),
]
