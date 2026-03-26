"""Ollama chat script for Mad World."""

import asyncio
import gzip
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import ollama
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

# Define styles for the prompt
style = Style.from_dict(
    {
        "user": "#ansicyan bold",
        "assistant": "#ansigreen bold",
    }
)


SLASH_COMMANDS: dict[str, tuple[str, Callable[[], None]]] = {
    "/exit": ("Exit the shell.", lambda: sys.exit(0)),
    "/quit": ("Exit the shell.", lambda: sys.exit(0)),
    "/help": ("Print this help output.", lambda: print_help()),
}


def print_help() -> None:
    click.secho("Supported slash commands:", fg="yellow")
    for key, (desc, _cb) in SLASH_COMMANDS.items():
        click.secho(f"  {key}: {desc}", fg="yellow")


async def run_chat(log_file: Path, model: str) -> None:
    """Run the interactive chat session."""
    try:
        with gzip.open(log_file, "rt", encoding="utf-8") as f:
            messages: list[dict[str, Any]] = json.load(f)
    except Exception as e:
        click.secho(f"Error loading log file: {e}", fg="red", err=True)
        sys.exit(1)

    if not isinstance(messages, list):
        click.secho(
            "Error: Log file must contain a list of messages.",
            fg="red",
            err=True,
        )
        sys.exit(1)

    click.secho(f"Loaded {len(messages)} messages from {log_file}", fg="yellow")
    click.secho(f"Using model: {model}", fg="yellow")
    click.echo("Type '/exit' or '/quit' to end the session.\n")

    client = ollama.AsyncClient()

    # Setup history file for the prompt_toolkit session
    history_path = Path.home() / ".mad_world_chat_history"
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path))
    )

    while True:
        try:
            # Using prompt_toolkit to get user input
            user_input = await session.prompt_async(
                [("class:user", "User > ")],
                style=style,
                completer=WordCompleter(list(SLASH_COMMANDS.keys())),
                complete_while_typing=False,
                enable_history_search=True,
            )
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        cmd_ent = SLASH_COMMANDS.get(user_input)
        if cmd_ent is not None:
            cmd_ent[1]()
            continue

        if user_input.lower() in ("/exit", "/quit"):
            break

        messages.append({"role": "user", "content": user_input})

        click.secho("Assistant > ", fg="green", bold=True, nl=False)

        full_response = ""
        try:
            async for part in await client.chat(
                model=model,
                messages=messages,
                stream=True,
            ):
                content = part["message"]["content"]
                click.echo(content, nl=False)
                full_response += content
            click.echo("\n")
        except Exception as e:
            click.secho(
                f"\nError communicating with Ollama: {e}", fg="red", err=True
            )
            continue

        messages.append({"role": "assistant", "content": full_response})


@click.command()
@click.argument(
    "log_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--model",
    default="gemma3:12b",
    help="The name of the Ollama model to use.",
)
def main(log_file: Path, model: str) -> None:
    """
    Chat with an Ollama model using history from a gzipped JSON log file.

    LOG_FILE is the path to a .gz file containing a JSON list of messages.
    """
    asyncio.run(run_chat(log_file, model))


if __name__ == "__main__":
    main()
