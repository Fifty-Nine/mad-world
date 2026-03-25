import logging
from datetime import datetime
from pathlib import Path

import click

from mad_world.core import format_results, game_loop
from mad_world.ollama_player import OllamaPlayer, debug_schemas
from mad_world.rules import GameRules
from mad_world.util import wrap_text


@click.command()
@click.option("--alpha-name", default="Norlandia", help="Name of player 1.")
@click.option(
    "--alpha-model",
    default="gemma3:12b",
    help="Ollama model used for player 1.",
)
@click.option(
    "--alpha-persona",
    default="Friendly Backstabber",
    help="Persona prompt for player 1.",
)
@click.option(
    "--omega-name", default="Southern Imperium", help="Name of player 2."
)
@click.option(
    "--omega-model",
    default="qwen3.5:9b",
    help="Ollama model used for player 2.",
)
@click.option(
    "--omega-persona",
    default="Ruthless Calculator",
    help="Persona prompt for player 2.",
)
def main(
    alpha_name: str,
    alpha_model: str,
    alpha_persona: str,
    omega_name: str,
    omega_model: str,
    omega_persona: str,
) -> None:

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_base = log_dir / (
        f"{alpha_name}-{alpha_persona}-{alpha_model}-vs-"
        f"{omega_name}-{omega_persona}-{omega_model}."
        f"{datetime.now().isoformat()}"
    ).replace(":", "-").replace(" ", "_")
    debug_log_file = log_file_base.with_suffix(".debug.txt")
    log_file = log_file_base.with_suffix(".txt")

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

    try:
        logging.info(
            wrap_text(
                format_results(
                    *game_loop(
                        GameRules(),
                        [
                            OllamaPlayer(
                                name=alpha_name,
                                opponent_name=omega_name,
                                persona=alpha_persona,
                                model=alpha_model,
                            ),
                            OllamaPlayer(
                                name=omega_name,
                                opponent_name=alpha_name,
                                persona=omega_persona,
                                model=omega_model,
                            ),
                        ],
                    )
                )
            )
        )
    except KeyboardInterrupt:
        debug_log_file.unlink(missing_ok=True)
        log_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
