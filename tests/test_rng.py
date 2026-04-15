"""Tests for serializable RNG module."""

from __future__ import annotations

import copy
import random

from pydantic import BaseModel, Field

import mad_world.rng
from mad_world.rng import ComparableRandom, deserialize_random_state


class DummyModel(BaseModel):
    rng: mad_world.rng.SerializableRandom = Field(description="The RNG.")


def test_rng_roundtrip(seeded_rng: mad_world.rng.ComparableRandom) -> None:
    dup_rng = copy.deepcopy(seeded_rng)

    model = DummyModel(rng=seeded_rng)
    assert model.rng.randint(0, 1000) == dup_rng.randint(0, 1000)

    model_copy = DummyModel.model_validate(model.model_dump())

    assert model.rng.randint(0, 1000) == model_copy.rng.randint(0, 1000)
    assert id(model.rng) != id(model_copy.rng)


def test_deserialize_from_random() -> None:

    std_rng = random.Random(42)

    # If we pass a random object directly (not its state)
    restored = deserialize_random_state(std_rng)

    assert isinstance(restored, ComparableRandom)
    assert restored == std_rng
