"""Entry point for Mad World."""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
from click_loglevel import LogLevel

from mad_world import trivial_players
from mad_world.core import RANDOM, format_results, game_loop
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
    return f"{RANDOM.choice(PERSONA_ADJECTIVES)} {RANDOM.choice(PERSONA_NOUNS)}"


def get_player(
    name: str,
    opponent_name: str,
    model: str,
    persona: str,
    temperature: float,
    context: int,
    tokens: int,
    log_dir: Path,
) -> GamePlayer:
    if model == "human":
        return HumanPlayer(name)

    if trivial_player := trivial_players.get_trivial_player(model, name):
        return trivial_player

    return OllamaPlayer(
        name=name,
        opponent_name=opponent_name,
        model=model,
        persona=persona,
        context_size=context,
        token_limit=tokens,
        temperature=temperature,
        log_dir=log_dir,
    )


@click.command()
@click.option("--alpha-name", default="Norlandia", help="Name of player 1.")
@click.option(
    "--alpha-model",
    default="gemma3:12b",
    help="Ollama model used for player 1.",
)
@click.option(
    "--alpha-persona",
    default=None,
    help="Persona prompt for player 1.",
)
@click.option(
    "--alpha-temperature",
    default=0.0,
    help="Temperature for player 1 model.",
)
@click.option(
    "--alpha-context",
    default=2**17,
    help="Context window size for player 1 model.",
)
@click.option(
    "--alpha-tokens",
    default=2**13,
    help="Output token budget for player 1 model.",
)
@click.option(
    "--omega-name",
    default="Southern Imperium",
    help="Name of player 2.",
)
@click.option(
    "--omega-model",
    default="gemma3:12b",
    help="Ollama model used for player 2.",
)
@click.option(
    "--omega-persona",
    default=None,
    help="Persona prompt for player 2.",
)
@click.option(
    "--omega-temperature",
    default=0.0,
    help="Temperature for player 2 model.",
)
@click.option(
    "--omega-context",
    default=2**17,
    help="Context window size for player 2 model.",
)
@click.option(
    "--omega-tokens",
    default=2**13,
    help="Output token budget for player 2 model.",
)
@click.option(
    "--log-dir",
    default="./logs",
    type=click.Path(path_type=Path),
    help="Base directory for logs.",
)
@click.option(
    "-v",
    "--verbosity",
    type=LogLevel(),
    default="DEBUG",
    help="Set verbosity level.",
    show_default=True,
)
def main(
    alpha_name: str,
    alpha_model: str,
    alpha_persona: str | None,
    alpha_temperature: float,
    alpha_context: int,
    alpha_tokens: int,
    omega_name: str,
    omega_model: str,
    omega_persona: str | None,
    omega_temperature: float,
    omega_context: int,
    omega_tokens: int,
    verbosity: int,
    log_dir: Path,
) -> None:
    asyncio.run(
        amain(
            alpha_name,
            alpha_model,
            alpha_persona,
            alpha_temperature,
            alpha_context,
            alpha_tokens,
            omega_name,
            omega_model,
            omega_persona,
            omega_temperature,
            omega_context,
            omega_tokens,
            verbosity,
            log_dir_base=log_dir,
        ),
    )


def setup_logging(verbosity: int, log_dir: Path) -> None:
    """Configures logging for the game session."""
    logger = logging.getLogger()
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


def create_log_session_dir(
    log_dir_base: Path,
    alpha_name: str,
    alpha_persona: str,
    alpha_model: str,
    omega_name: str,
    omega_persona: str,
    omega_model: str,
    timestamp: datetime | None = None,
) -> Path:
    """Creates a unique directory for the game session logs."""
    if timestamp is None:
        timestamp = datetime.now()

    dir_name = (
        (
            f"{alpha_name}-{alpha_persona}-{alpha_model}-vs-"
            f"{omega_name}-{omega_persona}-{omega_model}."
            f"{timestamp.isoformat()}"
        )
        .replace(":", "-")
        .replace(" ", "_")
    )

    log_dir = log_dir_base / dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


async def amain(
    alpha_name: str,
    alpha_model: str,
    alpha_persona: str | None,
    alpha_temperature: float,
    alpha_context: int,
    alpha_tokens: int,
    omega_name: str,
    omega_model: str,
    omega_persona: str | None,
    omega_temperature: float,
    omega_context: int,
    omega_tokens: int,
    verbosity: int,
    log_dir_base: Path = Path("./logs"),
) -> None:

    alpha_persona = alpha_persona or random_persona()
    omega_persona = omega_persona or random_persona()

    log_dir = create_log_session_dir(
        log_dir_base,
        alpha_name,
        alpha_persona,
        alpha_model,
        omega_name,
        omega_persona,
        omega_model,
    )

    setup_logging(verbosity, log_dir)

    logging.info(
        "Game starting\n"
        f"Player 1: {alpha_name}, {alpha_persona} ({alpha_model})\n"
        f"Player 2: {omega_name}, {omega_persona} ({omega_model})",
    )

    players = [
        get_player(
            alpha_name,
            omega_name,
            alpha_model,
            alpha_persona,
            alpha_temperature,
            alpha_context,
            alpha_tokens,
            log_dir,
        ),
        get_player(
            omega_name,
            alpha_name,
            omega_model,
            omega_persona,
            omega_temperature,
            omega_context,
            omega_tokens,
            log_dir,
        ),
    ]
    try:
        winner, reason, state = await game_loop(GameRules(), players)
        logging.info(wrap_text(format_results(winner, reason, state)))
    except (KeyboardInterrupt, EOFError):
        if log_dir.exists() and log_dir.is_dir():
            shutil.rmtree(log_dir)


if __name__ == "__main__":
    main()  # pragma: no cover
