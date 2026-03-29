"""Utility functions for Mad World."""

from __future__ import annotations

import re
import textwrap
from abc import ABC
from functools import singledispatch
from typing import Any, cast

from more_itertools import partition


def increase_or_decrease(val: int) -> str:
    return "increase" if val >= 0 else "decrease"


def cost_or_gain(val: int) -> str:
    return "gain" if val >= 0 else "cost"


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


def get_subclass_by_name[T: ABC](
    module_name: str,
    type_name: str,
    base_class: Any,
    *args: Any,
    **kwargs: Any,
) -> T | None:
    """
    Finds a matching subclass of the given base class that matches the
    normalized version of the given name in the given module.

    Args:
        module_name: The name of the module to search in.
        type_name: The name of the subclass to look for.
        base_class: The base class that the attribute must be a subclass of.
        *args: Positional arguments to pass to the class constructor.
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        An instance of the matching class constructed via cls(*args, **kwargs)
        or None if no match is found.
    """
    class_name = get_class_name(type_name)

    if cls := next(
        (
            c
            for c in base_class.__subclasses__()
            if c.__name__ == class_name and c.__module__ == module_name
        ),
        None,
    ):
        return cast("T", cls(*args, **kwargs))

    return None


def get_doomsday_bids(
    clock: int, limit: int, zero_bid_impact: int, bids: list[int]
) -> tuple[list[tuple[int, int]], list[int]]:
    """Compute bids that are risky or deadly given the current clock.

    Args:
        clock: The current doomsday clock value.
        limit: The maximum doomsday clock value.
        zero_bid_impact: The net effect of a zero bid.
        bids: The allowed bids.

    Returns:
        A tuple of (risky_bids, deadly_bids).
        risky_bids is a list of (bid, obid) where bid + obid >= limit.
        deadly_bids is a list of bids that unilaterally >= limit.
    """
    max_bid = max(bids)

    def bid_impact(bid: int) -> int:
        return bid or zero_bid_impact

    def bids_trigger_mad(bid: int, obid: int) -> bool:
        return clock + bid_impact(bid) + bid_impact(obid) >= limit

    if not bids_trigger_mad(max_bid, max_bid):
        return [], []

    non_deadly, deadly = partition(lambda b: bids_trigger_mad(b, 0), bids)
    risky = [
        (bid, min(obid for obid in bids if bids_trigger_mad(bid, obid)))
        for bid in non_deadly
        if bids_trigger_mad(bid, max_bid)
    ]
    return risky, list(deadly)


def reorder_schema_properties(
    schema: dict[str, Any], last_key: str
) -> dict[str, Any]:
    """Given the JSON Schema for a model, reorder the properties
    so that the `last_key` field always comes last. We also prefix the
    property keys with numbers (e.g. 00_, 01_) because the underlying
    llama.cpp grammar engine forcefully alphabetizes schema properties.
    Use remove_ordering_prefix to convert an instance matching this schema
    into one matching the original schema.
    """

    def process_obj(obj: dict[str, Any]) -> None:
        if "properties" not in obj:
            return

        old_props = obj["properties"]
        last_obj = old_props.pop(last_key, None)
        required = obj.get("required", [])
        new_props = {}

        for i, field in enumerate(old_props.keys()):
            field_obj = old_props[field]
            new_key = f"{i:02d}_{field}"

            new_props[new_key] = field_obj

            # I want to fix this but it's currently more trouble than
            # it's worth.
            # ast-grep-ignore: python-excessive-nesting
            if field in required:
                required.remove(field)
                required.append(new_key)

        if last_obj is not None:
            new_props[f"99_{last_key}"] = last_obj

        if last_key in required:
            required.remove(last_key)
            required.append(f"99_{last_key}")

        obj["properties"] = new_props

    process_obj(schema)
    for def_schema in schema.get("$defs", {}).values():
        process_obj(def_schema)

    return schema


ORDERING_PREFIX_RE = re.compile(r"^\d\d_")


@singledispatch
def remove_ordering_prefix(obj: Any, *, is_key: bool = False) -> Any:
    return obj


@remove_ordering_prefix.register
def _(obj: str, *, is_key: bool = False) -> str:
    return ORDERING_PREFIX_RE.sub("", obj) if is_key else obj


@remove_ordering_prefix.register(dict)
def _(obj: dict[Any, Any], *, is_key: bool = False) -> dict[Any, Any]:
    return {
        remove_ordering_prefix(k, is_key=True): remove_ordering_prefix(
            v, is_key=False
        )
        for k, v in obj.items()
    }


@remove_ordering_prefix.register(list)
def _(obj: list[Any], *, is_key: bool = False) -> list[Any]:
    return [remove_ordering_prefix(o, is_key=False) for o in obj]
