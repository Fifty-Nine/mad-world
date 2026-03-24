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
    if not text:
        return ""

    lines = text.splitlines()
    wrapped_lines = []
    for line in lines:
        if not line.strip():
            wrapped_lines.append(indent)
            continue

        existing_indent = line[: len(line) - len(line.lstrip())]
        combined_indent = indent + existing_indent

        wrapped = textwrap.wrap(
            line.lstrip(),
            width=width,
            initial_indent=combined_indent,
            subsequent_indent=combined_indent,
        )
        wrapped_lines.extend(wrapped)

    result = "\n".join(wrapped_lines)
    if text.endswith("\n"):
        result += "\n"

    return result
