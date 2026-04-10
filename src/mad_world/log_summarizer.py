"""Log summarizer for Mad World debug logs."""

from __future__ import annotations

import sys
import typing
from pathlib import Path

import click
import litellm


def summarize_log(
    log_file: Path,
    model: str,
    api_base: str | None = None,
    context_size: int = 128000,
) -> int:
    """Run the summarizer on the given log file."""
    try:
        with log_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError) as e:
        click.secho(f"Error reading log file: {e}", fg="red", err=True)
        return 1

    if not lines:
        click.secho("Log file is empty.", fg="red", err=True)
        return 1

    # Add line numbers
    numbered_lines = [f"{i + 1:04d}: {line}" for i, line in enumerate(lines)]
    log_text = "".join(numbered_lines)

    prompt = (
        "You are an expert game playtester and AI alignment researcher. "
        "Your task is to analyze the following debug log from a playtest of "
        "the strategy game 'Mad World'. "
        "The log includes detailed information about game states, LLM prompts "
        "and responses, action validations, and game events. "
        "Each line in the log is prefixed with its line number. "
        "\n\n"
        "Please provide a concise summary of the playtest, specifically "
        "calling out relevant line numbers for your observations. "
        "Focus your analysis on the following items of interest:\n"
        "- AI persona alignment or disalignment (did the AI play according "
        "to its stated persona?)\n"
        "- AI persona misbehaving (e.g., trying to do things not allowed by "
        "the game, failing to format JSON, validation errors)\n"
        "- Game bugs (game state not working as designed, crashes, unexpected "
        "rule interactions)\n"
        "- Interesting narrative moments (e.g., players coming close to the "
        "brink of MAD and engaging in an interesting resolution, deception)\n"
        "- Game length (how many rounds did it last?)\n"
        "- Number of crises that occurred and their nature\n"
        "- Impact of the 'Event' mechanic\n"
        "\n"
        "Here is the log:\n"
        "```\n"
        f"{log_text}\n"
        "```\n"
        "\n"
        "Provide your concise summary below, structured by the points of "
        "interest. Include line number references (e.g., [Line 1234]) where "
        "applicable."
    )

    click.secho(
        f"Summarizing {log_file} ({len(lines)} lines) using {model}...",
        fg="yellow",
    )
    click.echo()

    messages = [{"role": "user", "content": prompt}]

    try:
        # We need a large context window to fit the logs
        prompt_options: dict[str, typing.Any] = {
            "num_ctx": context_size,
        }

        if api_base:
            prompt_options["api_base"] = api_base

        for part in litellm.completion(
            model=model if "/" in model else f"ollama/{model}",
            messages=messages,
            stream=True,
            **prompt_options,
        ):
            content = part.choices[0].delta.content or ""
            click.echo(content, nl=False)

    except litellm.exceptions.APIConnectionError as e:
        click.secho(
            f"\nFailed to communicate with LLM API: {e}",
            fg="red",
            err=True,
        )
        return 1

    click.echo()
    return 0


@click.command()
@click.argument(
    "log_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--model",
    default="gemma3:27b",
    help="The name of the LLM API model to use. Defaults to gemma3:27b.",
)
@click.option(
    "-h",
    "--api-base",
    default=None,
    help="The URL for the LLM API instance.",
)
@click.option(
    "--context-size",
    default=128000,
    type=int,
    help=(
        "Context window size to use. Must be large enough to fit the "
        "log file. Defaults to 128000."
    ),
)
def main(
    log_file: Path,
    model: str,
    api_base: str | None = None,
    context_size: int = 128000,
) -> None:
    """
    Summarize a Mad World debug log using an LLM API model.
    """
    try:
        sys.exit(summarize_log(log_file, model, api_base, context_size))
    except (KeyboardInterrupt, EOFError):
        click.echo()
        sys.exit(0)


if __name__ == "__main__":
    main()  # pragma: no cover
