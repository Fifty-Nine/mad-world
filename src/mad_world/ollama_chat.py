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
from pydantic import ValidationError

from mad_world.config import LLMParams

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
def slash_commands() -> None: ...


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
pending_system_messages: list[str] = []


@slash_commands.command(
    name="image",
    help="Include an IMAGE in your next prompt.",
    add_help_option=False,
)
@click.argument("image", type=click.File("rb"))
def load_image(image: BinaryIO) -> None:
    global pending_images  # noqa: PLW0603
    pending_images += (image.read(),)


@slash_commands.command(
    name="system",
    help="Insert a system-role message into the prompt.",
    add_help_option=False,
)
@click.argument("text", nargs=-1)
def system_message(text: tuple[str, ...]) -> None:
    pending_system_messages.append(" ".join(text))


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
    llm_params: LLMParams,
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

    if pending_system_messages:
        messages.extend(
            {
                "role": "system",
                "content": message,
            }
            for message in pending_system_messages
        )
        pending_system_messages.clear()

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
    try:
        prompt_options = {
            "num_predict": llm_params.token_limit,
            "num_ctx": llm_params.context_size,
            "temperature": llm_params.temperature,
            "repeat_penalty": llm_params.repeat_penalty,
            "repeat_last_n": llm_params.repeat_last_n,
        }
        for part in client.chat(
            model=model,
            messages=messages,
            stream=True,
            options=prompt_options,
            think=True,
        ):
            content = part["message"]["content"]
            click.echo(content, nl=False)
            full_response += content

    except ollama.ResponseError as e:
        click.secho(
            f"Failed to communicate with ollama: {e}", fg="red", err=True
        )

    else:
        messages.append({"role": "assistant", "content": full_response})

    click.echo("\n")


def load_settings_from_file(settings_path: Path) -> tuple[LLMParams, str]:
    with settings_path.open("rt", encoding="utf-8") as f:
        settings_data = json.load(f)

    params = LLMParams()
    if params_obj := settings_data.get("params"):
        params = LLMParams.model_validate(params_obj)

    return params, settings_data.get("model")


def load_settings(settings_path: Path) -> tuple[LLMParams, str | None]:
    def warn(msg: str) -> None:
        click.secho(
            f"Warning: {msg} Using default LLM parameters.",
            fg="yellow",
            err=True,
        )

    try:
        return load_settings_from_file(settings_path)

    except FileNotFoundError:
        warn(f"Settings file {settings_path} not found.")

    except PermissionError:
        warn(f"Permissions do not allow reading from {settings_path}.")

    except IsADirectoryError:
        warn(f"{settings_path} is a directory.")

    except (json.JSONDecodeError, ValidationError) as e:
        warn(f"Failed to load settings: {e}")

    return LLMParams(), None


def run_chat(
    log_file: Path,
    model: str | None = None,
    host: str | None = None,
    gm_prompt: str | None = None,
    settings: Path | None = None,
) -> int:
    """Run the interactive chat session."""
    try:
        with gzip.open(log_file, "rt", encoding="utf-8") as f:
            messages: list[dict[str, Any]] = json.load(f)
    except (OSError, ValueError, gzip.BadGzipFile) as e:
        click.secho(f"Error loading log file: {e}", fg="red", err=True)
        return 1

    if not isinstance(messages, list):
        click.secho(
            "Error: Log file must contain a list of messages.",
            fg="red",
            err=True,
        )
        return 1

    if settings is None:
        if log_file.name.endswith(".messages.gz"):
            settings_name = (
                log_file.name[: -len(".messages.gz")] + ".model-settings.json"
            )
            settings_path = log_file.with_name(settings_name)
        else:
            settings_path = log_file.with_suffix(".model-settings.json")
    else:
        settings_path = settings

    llm_params, loaded_model = load_settings(settings_path)

    final_model = model or loaded_model or "gemma3:12b"

    if gm_prompt is not None:
        messages.append({"role": "system", "content": gm_prompt})

    click.secho(f"Loaded {len(messages)} messages from {log_file}", fg="yellow")
    click.secho(f"Using model: {final_model}", fg="yellow")
    click.echo("Type '/quit' to end the session.\n")

    client = ollama.Client(host=host)

    # Setup history file for the prompt_toolkit session
    history_path = Path.home() / ".mad_world_chat_history"
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
    )

    while True:
        prompt_loop(session, client, final_model, messages, llm_params)


@click.command()
@click.argument(
    "log_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--model",
    default=None,
    help=(
        "The name of the Ollama model to use. Defaults to the one in "
        "settings, or gemma3:12b."
    ),
)
@click.option(
    "-h",
    "--ollama-host",
    default=None,
    help="The URL for the ollama instance.",
)
@click.option(
    "--settings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to the model-settings.json file. Defaults to deriving from "
        "log_file."
    ),
)
@click.option(
    "--gm-prompt",
    "gm_prompt",
    default=(
        "The game master (GM) is joining the conversation for a post-match "
        "debrief. You should stay in character for this conversation unless "
        "directed otherwise by the GM. Answer all questions honestly without "
        "deception. Keep your responses brief unless directed otherwise."
    ),
    help=(
        "Set the additional system prompt that will be passed to the model "
        "before the conversation. This helps align the model with the intent "
        "of the upcoming conversation."
    ),
)
@click.option(
    "--no-gm-prompt",
    "gm_prompt",
    flag_value=None,
    help=(
        "Explicitly disable the GM prompt. Useful if you want to keep the "
        "model in the exact same state it was before its context was saved."
    ),
)
def main(
    log_file: Path,
    gm_prompt: str | None,
    model: str | None = None,
    ollama_host: str | None = None,
    settings: Path | None = None,
) -> None:
    """
    Chat with an Ollama model using history from a gzipped JSON log file.

    LOG_FILE is the path to a .gz file containing a JSON list of messages.
    """
    try:
        sys.exit(
            run_chat(
                log_file,
                model,
                host=ollama_host,
                settings=settings,
                gm_prompt=gm_prompt,
            )
        )
    except QuitProgram as qp:
        sys.exit(qp.rc)
    except (KeyboardInterrupt, EOFError):
        click.echo()
        sys.exit(0)


if __name__ == "__main__":
    main()  # pragma: no cover
