"""Models and logic relating to command-line (or other) configuration."""

from __future__ import annotations

import logging
from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, TypeAdapter


@cache
def _load_model_defaults() -> dict[str, LLMParams]:
    try:
        text = files("mad_world").joinpath("model_defaults.json").read_text()

        return TypeAdapter(dict[str, LLMParams]).validate_json(text)
    except FileNotFoundError:
        logging.getLogger(__name__).exception(
            "Error occured loading model defaults"
        )
        return {}


class PlayerKind(StrEnum):
    LLM = "llm"
    HUMAN = "human"
    TRIVIAL = "trivial"


class LLMParams(BaseModel):
    temperature: float = Field(
        default=0.8, description="Temperature for the LLM."
    )
    context_size: int = Field(default=2**15, description="Context window size.")
    token_limit: int = Field(
        default=2**13, description="Token limit per completion."
    )
    repeat_penalty: float = Field(
        default=1.1, description="Repetition penalty."
    )
    api_base: str | None = Field(
        default=None, description="The base URL for the LLM API."
    )
    repeat_last_n: int = Field(
        default=64, description="Repetition context size."
    )

    @staticmethod
    def defaults_for_model(model: str) -> LLMParams:
        return _load_model_defaults().get(model) or LLMParams()


class LLMPlayerConfig(BaseModel):
    kind: Literal[PlayerKind.LLM] = PlayerKind.LLM
    name: str
    model: str = Field(description="The LLM model being used.")
    persona: str | None = None
    params: LLMParams = Field(default_factory=LLMParams)

    def with_model_defaults(
        self, explicit_params: set[str] | None = None
    ) -> Self:
        explicit_params = (
            self.params.model_fields_set
            if explicit_params is None
            else explicit_params
        )

        universal_defaults = LLMParams().model_dump()
        model_defaults = LLMParams.defaults_for_model(self.model).model_dump(
            exclude_unset=True
        )
        user_options = {
            k: v
            for k, v in self.params.model_dump().items()
            if k in explicit_params
        }

        self.params = LLMParams.model_validate(
            universal_defaults | model_defaults | user_options
        )
        return self

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
