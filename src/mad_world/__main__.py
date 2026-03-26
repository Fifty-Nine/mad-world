import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path

import click

from mad_world.core import RANDOM, GamePlayer, format_results, game_loop
from mad_world.human_player import HumanPlayer
from mad_world.ollama_player import OllamaPlayer, debug_schemas
from mad_world.rules import GameRules
from mad_world.util import wrap_text

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
    "Desperate",
    "Dogmatic",
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
    "Principled",
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
    "Crusader",
    "Dictator",
    "Diplomat",
    "Dogmatist",
    "Extremist",
    "General",
    "Idealist",
    "Isolationist",
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
    "Vanguard",
    "Victor",
    "Zealot",
)


def random_persona() -> str:
    return f"{RANDOM.choice(PERSONA_ADJECTIVES)} {RANDOM.choice(PERSONA_NOUNS)}"


def get_player(
    name: str, opponent_name: str, model: str, persona: str, log_dir: Path
) -> GamePlayer:
    if model == "human":
        return HumanPlayer(name)

    return OllamaPlayer(
        name=name,
        opponent_name=opponent_name,
        model=model,
        persona=persona,
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
    "--omega-name", default="Southern Imperium", help="Name of player 2."
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
def main(
    alpha_name: str,
    alpha_model: str,
    alpha_persona: str | None,
    omega_name: str,
    omega_model: str,
    omega_persona: str | None,
) -> None:
    asyncio.run(
        amain(
            alpha_name,
            alpha_model,
            alpha_persona,
            omega_name,
            omega_model,
            omega_persona,
        )
    )


async def amain(
    alpha_name: str,
    alpha_model: str,
    alpha_persona: str | None,
    omega_name: str,
    omega_model: str,
    omega_persona: str | None,
) -> None:

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    alpha_persona = alpha_persona or random_persona()
    omega_persona = omega_persona or random_persona()

    log_dir_base = Path("./logs")
    log_dir = log_dir_base / (
        f"{alpha_name}-{alpha_persona}-{alpha_model}-vs-"
        f"{omega_name}-{omega_persona}-{omega_model}."
        f"{datetime.now().isoformat()}"
    ).replace(":", "-").replace(" ", "_")
    log_dir.mkdir(parents=True, exist_ok=True)
    debug_log_file = log_dir / "debug.txt"
    log_file = log_dir / "log.txt"

    debug_file_handler = logging.FileHandler(debug_log_file)
    debug_file_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    logger.addHandler(debug_file_handler)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)

    logging.info(
        "Game starting\n"
        f"Player 1: {alpha_name}, {alpha_persona} ({alpha_model})\n"
        f"Player 2: {omega_name}, {omega_persona} ({omega_model})"
    )

    debug_schemas()

    players = [
        get_player(alpha_name, omega_name, alpha_model, alpha_persona, log_dir),
        get_player(omega_name, alpha_name, omega_model, omega_persona, log_dir),
    ]
    try:
        winner, reason, state = await game_loop(GameRules(), players)
        logging.info(wrap_text(format_results(winner, reason, state)))
    except KeyboardInterrupt:
        if log_dir.exists() and log_dir.is_dir():
            shutil.rmtree(log_dir)


if __name__ == "__main__":
    main()
