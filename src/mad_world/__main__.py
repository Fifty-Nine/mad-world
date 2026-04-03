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


def random_persona() -> str:
    return f"{random.choice(PERSONA_ADJECTIVES)} {random.choice(PERSONA_NOUNS)}"


def get_player(
    config: PlayerConfig,
    opponent_name: str,
    log_dir: Path,
    logger: logging.Logger | None = None,
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


def _ensure_persona(config: PlayerConfig) -> PlayerConfig:
    if config.kind == PlayerKind.LLM and config.persona is None:
        new_config = config.model_copy(deep=True)
        new_config.persona = random_persona()
        return new_config
    return config


def run_game(
    alpha: PlayerConfig = DEFAULT_ALPHA,
    omega: PlayerConfig = DEFAULT_OMEGA,
    log_dir: Path = Path("./logs"),
    verbosity: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG",
) -> None:
    """
    Mad World CLI

    Args:
        alpha: Configuration for player 1.
        omega: Configuration for player 2.
        log_dir: Base directory for storing session logs.
        verbosity: Logging verbosity level.
    """
    alpha = _ensure_persona(alpha)
    omega = _ensure_persona(omega)

    log_level = getattr(logging, verbosity)

    asyncio.run(
        amain(
            alpha_config=alpha,
            omega_config=omega,
            verbosity=log_level,
            log_dir_base=log_dir,
        )
    )


def main() -> None:
    CLI(run_game)  # type: ignore[no-untyped-call]


def setup_logging(verbosity: int, log_dir: Path) -> logging.Logger:
    """Configures logging for the game session."""
    logger = logging.getLogger("mad_world")
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
    stream_handler.setLevel(verbosity)

    logger.addHandler(debug_file_handler)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)

    return logger


def get_persona(config: PlayerConfig) -> str:
    if config.kind == PlayerKind.LLM:
        return config.persona or ""
    return ""


def get_model_name(config: PlayerConfig) -> str:
    if config.kind == PlayerKind.LLM:
        return config.model
    if config.kind == PlayerKind.TRIVIAL:
        return config.bot_name
    return "human"


def create_log_session_dir(
    log_dir_base: Path,
    alpha_config: PlayerConfig,
    omega_config: PlayerConfig,
    timestamp: datetime | None = None,
) -> Path:
    """Creates a unique directory for the game session logs."""
    if timestamp is None:
        timestamp = datetime.now()

    alpha_persona = get_persona(alpha_config).partition("\n")[0]
    omega_persona = get_persona(omega_config).partition("\n")[0]
    alpha_persona = f"-{alpha_persona}" if alpha_persona else ""
    omega_persona = f"-{omega_persona}" if omega_persona else ""

    alpha_model = get_model_name(alpha_config)
    omega_model = get_model_name(omega_config)

    dir_name = (
        (
            f"{alpha_config.name}{alpha_persona}-{alpha_model}-vs-"
            f"{omega_config.name}{omega_persona}-{omega_model}."
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
    verbosity: int,
    log_dir_base: Path = Path("./logs"),
) -> None:

    log_dir = create_log_session_dir(
        log_dir_base,
        alpha_config,
        omega_config,
    )

    logger = setup_logging(verbosity, log_dir)

    logger.info(
        "Game starting\nPlayer 1: %s %s (%s)\nPlayer 2: %s %s (%s)",
        alpha_config.name,
        get_persona(alpha_config),
        get_model_name(alpha_config),
        omega_config.name,
        get_persona(omega_config),
        get_model_name(omega_config),
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
    try:
        winner, reason, state = await game_loop(GameRules(), players)
        logger.info(wrap_text(format_results(winner, reason, state)))
    except (KeyboardInterrupt, EOFError):
        if log_dir.exists() and log_dir.is_dir():
            shutil.rmtree(log_dir)


if __name__ == "__main__":
    main()  # pragma: no cover
