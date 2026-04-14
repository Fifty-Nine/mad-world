"""Functions for working with random numbers/sequences/etc."""

from __future__ import annotations

import random
from typing import Annotated, Any

from pydantic import PlainSerializer, PlainValidator
from pydantic.json_schema import SkipJsonSchema


def _to_tuple(obj: Any) -> Any:
    if isinstance(obj, list):
        return tuple(_to_tuple(x) for x in obj)
    return obj


def deserialize_random_state(state: Any) -> random.Random:
    if isinstance(state, random.Random):
        return state

    result = random.Random()
    result.setstate(_to_tuple(state))
    return result


def serialize_random_state(rng: random.Random) -> Any:
    return rng.getstate()


SerializableRandom = Annotated[
    random.Random,
    PlainValidator(deserialize_random_state),
    PlainSerializer(serialize_random_state, return_type=Any),
    SkipJsonSchema,
]
