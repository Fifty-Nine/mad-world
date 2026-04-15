"""Functions for working with random numbers/sequences/etc."""

from __future__ import annotations

import random
from typing import Annotated, Any

from pydantic import PlainSerializer, PlainValidator
from pydantic.json_schema import SkipJsonSchema


class ComparableRandom(random.Random):
    """A random.Random subclass that supports equality based on state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __eq__(self, other: Any) -> bool:
        if hasattr(other, "getstate"):
            # mypy and ruff disagree on the type, so we can't satisfy both
            return self.getstate() == other.getstate()  # type: ignore[no-any-return]
        return NotImplemented  # pragma: no cover

    def __hash__(self) -> int:
        raise TypeError("ComparableRandom is unhashable")  # pragma: no cover


def _to_tuple(obj: Any) -> Any:
    if isinstance(obj, list):
        return tuple(_to_tuple(x) for x in obj)
    return obj


def deserialize_random_state(state: Any) -> ComparableRandom:
    if isinstance(state, ComparableRandom):
        return state

    result = ComparableRandom()
    if hasattr(state, "getstate"):
        result.setstate(state.getstate())
    else:
        result.setstate(_to_tuple(state))
    return result


def serialize_random_state(rng: random.Random) -> Any:
    return rng.getstate()


SerializableRandom = Annotated[
    ComparableRandom,
    PlainValidator(deserialize_random_state),
    PlainSerializer(serialize_random_state, return_type=Any),
    SkipJsonSchema,
]
