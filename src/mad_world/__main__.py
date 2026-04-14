"""Entry point for Mad World."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anyio
from jsonargparse import ActionYesNo, ArgumentParser
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from mad_world import trivial_players
from mad_world.config import (
    LLMPlayerConfig,
    PlayerConfig,
    PlayerKind,
    TrivialPlayerConfig,
)
from mad_world.core import GameState, format_results, game_loop
from mad_world.hooks import GameLoopCallback, GameLoopHook
from mad_world.human_player import HumanPlayer
from mad_world.ollama_player import OllamaPlayer
from mad_world.personas import random_persona
from mad_world.rules import GameRules
from mad_world.util import wrap_text

if TYPE_CHECKING:
    from prompt_toolkit.key_binding.key_processor import KeyPressEvent

    from mad_world.players import GamePlayer


def coerce_bool_response(response: str, *, default_val: bool) -> bool | None:
    response = response.strip().lower()
    if response == "":
        return default_val

    if response in ("n", "no"):
        return False

    if response in ("y", "ye", "yes"):
        return True

    return None


async def prompt_bool_once(
    prompt: str,
    session: PromptSession[str],
    loop_timeout: float = 5.0,
    *,
    default_val: bool,
) -> bool | str:
    """Prompts the user for a boolean value."""
    suffix = " [Y/n]> " if default_val else " [y/N]> "
    try:
        answer = await asyncio.wait_for(
            session.prompt_async(prompt + suffix), timeout=loop_timeout
        )
    except (TimeoutError, KeyboardInterrupt, EOFError):
        return default_val

    result = coerce_bool_response(answer, default_val=default_val)
    return result if result is not None else answer


async def prompt_bool(
    prompt: str, loop_timeout: float = 5.0, *, default_val: bool
) -> bool:
    session: PromptSession[str] = PromptSession()
    prefix = ""
    while True:
        result = await prompt_bool_once(
            prefix + prompt, session, loop_timeout, default_val=default_val
        )

        if isinstance(result, str):
            prefix = f"Unrecognized response: {result}\n"
            continue

        return result


def should_preserve_logs(*, default_choice: bool) -> bool:
    return asyncio.run(
        prompt_bool(
            "Do you want to preserve the logs?", default_val=default_choice
        )
    )


def get_player(
    config: PlayerConfig,
    opponent_name: str,
    log_dir: Path,
    logger: logging.Logger,
) -> GamePlayer:
    """Instantiates a player from a name/URI."""
    if config.kind == PlayerKind.HUMAN:
        return HumanPlayer(config.name)

    logger = logger or logging.getLogger("mad_world")

    if config.kind == PlayerKind.TRIVIAL:
        assert isinstance(config, TrivialPlayerConfig)
        trivial_player = trivial_players.get_trivial_player(
            config.bot_name, config.name
        )
        if not trivial_player:
            msg = f"Unknown trivial player bot name: {config.bot_name}"
            raise ValueError(msg)
        return trivial_player

    assert isinstance(config, LLMPlayerConfig)
    return OllamaPlayer(
        config=config,
        opponent_name=opponent_name,
        log_dir=log_dir,
        compression_threshold=0.75,
        logger=logger,
    )


DEFAULT_ALPHA = LLMPlayerConfig(
    kind=PlayerKind.LLM,
    name="Norlandia",
    model="gemma4:26b",
)
DEFAULT_OMEGA = LLMPlayerConfig(
    kind=PlayerKind.LLM,
    name="Southern Imperium",
    model="gemma4:26b",
)


def _default_config(
    config: PlayerConfig, explicit_params: set[str] | None = None
) -> PlayerConfig:
    updates = {}
    if config.kind == PlayerKind.LLM:
        assert isinstance(config, LLMPlayerConfig)
        if config.persona is None:
            updates["persona"] = random_persona()
        config.with_model_defaults(explicit_params)

    return config.model_copy(update=updates, deep=True)


def _get_explicitly_set_params(
    args: list[str], player: Literal["alpha", "omega"]
) -> set[str]:
    prefix = f"--{player}.params."
    return {
        arg.split("=")[0].replace(prefix, "")
        for arg in args
        if arg.startswith(prefix)
    }


def run_game(
    alpha: PlayerConfig = DEFAULT_ALPHA,
    omega: PlayerConfig = DEFAULT_OMEGA,
    log_dir: Path = Path("./logs"),
    verbosity: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    *,
    single_step: bool = False,
) -> None:
    """
    Mad World CLI

    Args:
        alpha: Configuration for player 1.
        omega: Configuration for player 2.
        log_dir: Base directory for storing session logs.
        verbosity: Logging verbosity level.
        single_step: Whether to pause before advancing each game phase.
    """
    alpha = _default_config(
        alpha, _get_explicitly_set_params(sys.argv, "alpha")
    )
    omega = _default_config(
        omega, _get_explicitly_set_params(sys.argv, "omega")
    )

    specific_log_dir = create_log_session_dir(
        log_dir,
        alpha,
        omega,
    )

    logger = setup_logging(verbosity, specific_log_dir)

    preserve_by_default = True
    try:
        asyncio.run(
            amain(
                alpha_config=alpha,
                omega_config=omega,
                log_dir=specific_log_dir,
                logger=logger,
                single_step=single_step,
            )
        )
    except (KeyboardInterrupt, EOFError):
        preserve_by_default = False

    if (
        not should_preserve_logs(default_choice=preserve_by_default)
        and specific_log_dir.exists()
        and specific_log_dir.is_dir()
    ):
        shutil.rmtree(specific_log_dir)


def main() -> None:
    parser = ArgumentParser(prog="mad_world")
    parser.add_function_arguments(run_game, skip={"single_step"})
    parser.add_argument(
        "--single_step",
        action=ActionYesNo,
        default=False,
        help="Whether to pause before advancing each game phase.",
    )
    args = parser.parse_args()
    instantiated_args = parser.instantiate_classes(args)
    run_game(**instantiated_args.as_dict())


def setup_logging(verbosity: str, log_dir: Path) -> logging.Logger:
    """Configures logging for the game session."""
    logger = logging.getLogger("mad_world")
    log_level = getattr(logging, verbosity)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    debug_log_file = log_dir / "debug.txt"
    log_file = log_dir / "log.txt"

    debug_file_handler = logging.FileHandler(debug_log_file)
    debug_file_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    logger.addHandler(debug_file_handler)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)

    return logger


def create_log_session_dir(
    log_dir_base: Path,
    alpha_config: PlayerConfig,
    omega_config: PlayerConfig,
    timestamp: datetime | None = None,
) -> Path:
    """Creates a unique directory for the game session logs."""
    if timestamp is None:
        timestamp = datetime.now()

    dir_name = (
        (
            f"{alpha_config.file_name()}-vs-{omega_config.file_name()}."
            f"{timestamp.isoformat()}"
        )
        .replace(":", "-")
        .replace(" ", "_")
    )

    log_dir = log_dir_base / dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


async def single_step_callback(game: GameState) -> GameState | None:
    bindings = KeyBindings()

    @bindings.add("<any>")
    def _(event: KeyPressEvent) -> None:
        event.app.exit()

    print(  # noqa: T201
        "\n--- [Press any key to advance to the next phase] ---"
    )
    await PromptSession(key_bindings=bindings).prompt_async("")
    return None


async def amain(
    alpha_config: PlayerConfig,
    omega_config: PlayerConfig,
    logger: logging.Logger,
    log_dir: Path,
    *,
    single_step: bool = False,
) -> None:
    """Main async entrypoint."""

    logger.info(
        "Game starting\nPlayer 1: %s\nPlayer 2: %s",
        alpha_config.summarize(),
        omega_config.summarize(),
    )

    players = [
        get_player(
            alpha_config,
            omega_config.name,
            log_dir,
            logger,
        ),
        get_player(
            omega_config,
            alpha_config.name,
            log_dir,
            logger,
        ),
    ]

    async def autosave_callback(game: GameState) -> GameState | None:
        try:
            await anyio.Path(os.fspath(log_dir / "game_state.json")).write_text(
                game.model_dump_json(indent=2)
            )
        except OSError:
            logger.exception("Failed to write save game to log dir")
        return None

    callbacks: list[GameLoopCallback] = [
        {GameLoopHook.POST_PHASE: autosave_callback}
    ]

    if single_step:
        callbacks.append({GameLoopHook.POST_PHASE: single_step_callback})

    winner, reason, state = await game_loop(
        GameRules(), players, callbacks=callbacks
    )
    logger.info(wrap_text(format_results(winner, reason, state)))


if __name__ == "__main__":
    main()  # pragma: no cover│asyncio.exceptions.CancelledError
