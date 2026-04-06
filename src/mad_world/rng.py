"""Functions for working with random numbers/sequences/etc."""

from __future__ import annotations

import random
from typing import Annotated, Any, cast

from pydantic import PlainSerializer, PlainValidator
from pydantic.json_schema import SkipJsonSchema


def deserialize_random_state(state: Any) -> random.Random:
    if getattr(state, "seed", None) is not None and callable(
        getattr(state, "getstate", None)
    ):
        return cast("random.Random", state)

    result = random.Random()
    result.setstate(state)
    return result


def serialize_random_state(rng: random.Random) -> Any:
    return rng.getstate()


SerializableRandom = Annotated[
    random.Random,
    PlainValidator(deserialize_random_state),
    PlainSerializer(serialize_random_state, return_type=Any),
    SkipJsonSchema,
]
