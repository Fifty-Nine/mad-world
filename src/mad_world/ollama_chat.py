"""Ollama chat script for Mad World."""

from __future__ import annotations

import copy
import gzip
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

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
    },
)


@dataclass
class QuitProgram(Exception):
    rc: int


@click.group()
def slash_commands() -> None:
    pass


@slash_commands.command(
    name="quit",
    help="Exit the application.",
    add_help_option=False,
)
def exit_loop() -> None:
    raise QuitProgram(0)


@slash_commands.command(
    name="help",
    help=(
        "Print the list of available commands or, with an argument, "
        "get the help for the given command."
    ),
    add_help_option=False,
)
@click.argument("command_name", default=None)
def print_slash_help(command_name: str | None) -> None:
    if command_name:
        ctx = click.Context(slash_commands, info_name=command_name)

        if cmd := slash_commands.get_command(ctx, command_name):
            click.secho(cmd.get_help(ctx), fg="yellow")
        else:
            click.secho(f"Error: No such command: '{command_name}'", fg="red")

    else:
        for name, cmd in slash_commands.commands.items():
            click.secho(f"{name}: {cmd.get_short_help_str()}", fg="yellow")


pending_images: list[bytes] = []


@slash_commands.command(
    name="image",
    help="Include an IMAGE in your next prompt.",
    add_help_option=False,
)
@click.argument("image", type=click.File("rb"))
def load_image(image: BinaryIO) -> None:
    global pending_images
    pending_images += (image.read(),)


def process_slash_command(user_input: str) -> bool:
    if not user_input.startswith("/"):
        return False

    args = shlex.split(user_input[1:])
    try:
        slash_commands.main(args=args, standalone_mode=False)

    except click.ClickException as e:
        click.secho(f"Error: {e.format_message()}", fg="red")

    return True


def prompt_loop(
    session: PromptSession[str],
    client: ollama.Client,
    model: str,
    messages: list[dict[str, Any]],
) -> None:
    user_input = session.prompt(
        [("class:user", "User > ")],
        style=style,
        completer=WordCompleter(list(slash_commands.commands.keys())),
        complete_while_typing=False,
        enable_history_search=True,
    ).strip()

    if not user_input or process_slash_command(user_input):
        return

    messages.append(
        {
            "role": "user",
            "content": user_input,
            "images": copy.deepcopy(pending_images),
        },
    )
    pending_images.clear()

    click.secho("Assistant > ", fg="green", bold=True, nl=False)

    full_response = ""
    for part in client.chat(
        model=model,
        messages=messages,
        stream=True,
    ):
        content = part["message"]["content"]
        click.echo(content, nl=False)
        full_response += content
    click.echo("\n")
    messages.append({"role": "assistant", "content": full_response})


def run_chat(log_file: Path, model: str) -> int:
    """Run the interactive chat session."""
    try:
        with gzip.open(log_file, "rt", encoding="utf-8") as f:
            messages: list[dict[str, Any]] = json.load(f)
    except OSError as e:
        click.secho(f"Error loading log file: {e}", fg="red", err=True)
        return 1

    if not isinstance(messages, list):
        click.secho(
            "Error: Log file must contain a list of messages.",
            fg="red",
            err=True,
        )
        return 1

    click.secho(f"Loaded {len(messages)} messages from {log_file}", fg="yellow")
    click.secho(f"Using model: {model}", fg="yellow")
    click.echo("Type '/quit' to end the session.\n")

    client = ollama.Client()

    # Setup history file for the prompt_toolkit session
    history_path = Path.home() / ".mad_world_chat_history"
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
    )

    while True:
        prompt_loop(session, client, model, messages)


@click.command()
@click.argument(
    "log_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
    try:
        sys.exit(run_chat(log_file, model))
    except QuitProgram as qp:
        sys.exit(qp.rc)
    except (KeyboardInterrupt, EOFError):
        click.echo()
        sys.exit(0)


if __name__ == "__main__":
    main()
