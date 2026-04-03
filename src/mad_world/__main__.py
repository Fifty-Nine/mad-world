"""Entry point for Mad World."""

from __future__ import annotations

import asyncio
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from jsonargparse import CLI
from prompt_toolkit import PromptSession

from mad_world import trivial_players
from mad_world.config import (
    LLMPlayerConfig,
    PlayerConfig,
    PlayerKind,
)
from mad_world.core import format_results, game_loop
from mad_world.human_player import HumanPlayer
from mad_world.ollama_player import OllamaPlayer
from mad_world.rules import GameRules
from mad_world.util import wrap_text

if TYPE_CHECKING:
    from mad_world.players import GamePlayer

PERSONA_ADJECTIVES = (
    "Amateurish",
    "Belligerent",
    "Bloodthirsty",
    "Bureaucratic",
    "Calculating",
    "Careful",
    "Cautious",
    "Cold",
    "Covert",
    "Cynical",
    "Defensive",
    "Delusional",
    "Devouring",
    "Desperate",
    "Dogmatic",
    "Earnest",
    "Elder",
    "Erratic",
    "Fanatical",
    "Friendly",
    "Heartless",
    "Hellish",
    "Holy",
    "Horrid",
    "Ideological",
    "Ineffective",
    "Inflexible",
    "Inscrutable",
    "Insular",
    "Jeffersonian",
    "Kafkaesque",
    "Leninist",
    "Machiavellian",
    "Maoist",
    "Marxist",
    "Nervous",
    "Nixonian",
    "Opportunistic",
    "Paranoid",
    "Pragmatic",
    "Pretentious",
    "Principled",
    "Quixotic",
    "Rational",
    "Reaganesque",
    "Reckless",
    "Reluctant",
    "Silent",
    "Smiling",
    "Spiteful",
    "Stalinist",
    "Stoic",
    "Theatrical",
    "Trotskyist",
    "Uncompromising",
    "Unpredictable",
    "Vengeful",
    "Zealous",
)

PERSONA_NOUNS = (
    "Apparatchik",
    "Appeaser",
    "Architect",
    "Assassin",
    "Autocrat",
    "Backstabber",
    "Bastard",
    "Bore",
    "Bungler",
    "Brinksman",
    "Builder",
    "Bureaucrat",
    "Calculator",
    "Criminal",
    "Crusader",
    "Dictator",
    "Diplomat",
    "Dogmatist",
    "Extremist",
    "General",
    "Idealist",
    "Isolationist",
    "Jackass",
    "Martyr",
    "Mastermind",
    "Mirror",
    "Monarch",
    "Opportunist",
    "Pacifist",
    "Plotter",
    "Predator",
    "Premier",
    "Profiteer",
    "Realist",
    "Rogue",
    "Saboteur",
    "Strongman",
    "Subverter",
    "Survivor",
    "Tactician",
    "Technocrat",
    "Thug",
    "Vanguard",
    "Victor",
    "Zealot",
)


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


def random_persona() -> str:
    return f"{random.choice(PERSONA_ADJECTIVES)} {random.choice(PERSONA_NOUNS)}"


def get_player(
    config: PlayerConfig,
    opponent_name: str,
    log_dir: Path,
    logger: logging.Logger,
) -> GamePlayer:
    if config.kind == PlayerKind.HUMAN:
        return HumanPlayer(config.name)

    logger = logger or logging.getLogger("mad_world")

    if config.kind == PlayerKind.TRIVIAL:
        trivial_player = trivial_players.get_trivial_player(
            config.bot_name, config.name
        )
        if not trivial_player:
            msg = f"Unknown trivial player bot name: {config.bot_name}"
            raise ValueError(msg)
        return trivial_player

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
    model="gemma3:12b",
)
DEFAULT_OMEGA = LLMPlayerConfig(
    kind=PlayerKind.LLM,
    name="Southern Imperium",
    model="gemma3:12b",
)


def _default_config(config: PlayerConfig) -> PlayerConfig:
    updates = {}
    if config.kind == PlayerKind.LLM and config.persona is None:
        updates["persona"] = random_persona()

    return config.model_copy(update=updates, deep=True)


def run_game(
    alpha: PlayerConfig = DEFAULT_ALPHA,
    omega: PlayerConfig = DEFAULT_OMEGA,
    log_dir: Path = Path("./logs"),
    verbosity: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> None:
    """
    Mad World CLI

    Args:
        alpha: Configuration for player 1.
        omega: Configuration for player 2.
        log_dir: Base directory for storing session logs.
        verbosity: Logging verbosity level.
    """
    alpha = _default_config(alpha)
    omega = _default_config(omega)

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
    CLI(run_game, prog="mad_world")  # type: ignore[no-untyped-call]


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


async def amain(
    alpha_config: PlayerConfig,
    omega_config: PlayerConfig,
    logger: logging.Logger,
    log_dir: Path,
) -> None:

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
    winner, reason, state = await game_loop(GameRules(), players)
    logger.info(wrap_text(format_results(winner, reason, state)))


if __name__ == "__main__":
    main()  # pragma: no cover│asyncio.exceptions.CancelledError
