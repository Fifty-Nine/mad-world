"""Utility functions for Mad World."""

import re
import textwrap
from typing import Any


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


def escalation_budget(clock: int, max_clock: int) -> int:
    """Return the escalation budget for the given clock value and doomsday
    limit. This is the total amount of clock headroom divided by 2 and
    represents the pareto-optimal escalation. Clock impacts below this
    amount are non-optimal, but may help defuse tensions. Clock impacts
    above this amount are inherently dangerous as they may risk triggering
    MAD even if the opponent bids conservatively.
    """
    return (max_clock - 1 - clock) // 2


def pareto_optimal_bid(
    clock: int, max_clock: int, allowed_bids: list[int]
) -> int:
    """Similar to the above but specific to the bidding phase. This returns
    the highest allowed bid within the escalation budget. Because bid values are
    constrained, the optimal bid may be significantly smaller than the
    actual pareto-optimal escalation budget."""

    return max(
        (
            bid
            for bid in allowed_bids
            if bid < escalation_budget(clock, max_clock)
        ),
        default=0,
    )


def get_class_name(name: str) -> str:
    """Converts snake_case, kebab-case, camelCase or PascalCase to
    PascalCase.
    """
    # Handle all-caps or mixed case by normalizing to a readable format first
    # Insert space between lowercase and uppercase (for camelCase)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # Replace separators with spaces
    s = s.replace("_", " ").replace("-", " ")
    # Capitalize each word and join, ensuring rest of word is lowercase
    return "".join(
        word[0].upper() + word[1:].lower() if word else "" for word in s.split()
    )


def get_attr_by_type[T](
    namespace: Any, expected_type: Any, name: str
) -> type[T] | None:
    """
    Finds a matching attribute by name (normalized to PascalCase) in a
    namespace that is a subclass of the expected type.

    Args:
        namespace: The object/module to search in.
        expected_type: The base class that the attribute must be a subclass of.
        name: The name to look for.

    Returns:
        The matching class if found, else None.
    """
    class_name = get_class_name(name)
    attr = getattr(namespace, class_name, None)
    if (
        isinstance(attr, type)
        and issubclass(attr, expected_type)
        and attr is not expected_type
    ):
        return attr
    return None
