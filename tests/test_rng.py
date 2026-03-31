"""Tests for serializable RNG module."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

import mad_world.rng

if TYPE_CHECKING:
    import random


class DummyModel(BaseModel):
    rng: mad_world.rng.SerializableRandom = Field(description="The RNG.")


def test_rng_roundtrip(seeded_rng: random.Random) -> None:
    dup_rng = copy.deepcopy(seeded_rng)

    model = DummyModel(rng=seeded_rng)
    assert model.rng.randint(0, 1000) == dup_rng.randint(0, 1000)

    model_copy = DummyModel.model_validate(model.model_dump())

    assert model.rng.randint(0, 1000) == model_copy.rng.randint(0, 1000)
    assert id(model.rng) != id(model_copy.rng)
