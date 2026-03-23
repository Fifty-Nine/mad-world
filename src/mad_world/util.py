"""Utility functions for Mad World."""

import textwrap


def wrap_text(text: str, indent: str = "", width: int = 80) -> str:
    """
    Wraps text while preserving existing newlines.

    Args:
        text: The text to wrap.
        indent: The indent string to apply to each line.
        width: The maximum width of each line.

    Returns:
        The wrapped text as a single string.
    """
    lines = text.splitlines()
    wrapped_lines = []
    for line in lines:
        if not line.strip():
            wrapped_lines.append(indent)
            continue

        wrapped = textwrap.wrap(
            line,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
        )
        wrapped_lines.extend(wrapped)

    return "\n".join(wrapped_lines)
