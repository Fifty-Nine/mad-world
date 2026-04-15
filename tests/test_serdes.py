from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field, TypeAdapter, create_model

import mad_world.enums
from mad_world.events import ActorKind, PlayerActor, SystemActor


@pytest.fixture
def lookup_test_value(request: pytest.FixtureRequest) -> Any:
    if isinstance(request.param, str):
        result = request.getfixturevalue(request.param)
        assert isinstance(result, BaseModel)
        return result

    return request.param


def literal_field(val: Any, type_hint: Any = None) -> Any:
    if type_hint is None:
        type_hint = type(val)
    dynamic_model = create_model("DynamicModel", m=(type_hint, Field()))
    return dynamic_model(m=val)


@pytest.mark.parametrize(
    "lookup_test_value",
    [
        pytest.param("basic_game", marks=pytest.mark.xfail),
        "basic_player",
        literal_field(mad_world.enums.GamePhase.BIDDING),
        literal_field(mad_world.enums.GameOverReason.ECONOMIC_VICTORY),
        literal_field(mad_world.enums.StandoffPosture.STAND_FIRM),
        literal_field(mad_world.enums.BlameGamePosture.DEFLECT),
        literal_field(mad_world.enums.OpenChannelPreference.ACCEPT),
        literal_field(ActorKind.SYSTEM),
        literal_field(
            [SystemActor(), PlayerActor(name="Foo"), None],
            type_hint=list[SystemActor | PlayerActor | None],
        ),
        literal_field(SystemActor()),
        literal_field(PlayerActor(name="bar")),
    ],
    indirect=True,
)
def test_round_trip_serialization(lookup_test_value: Any) -> None:

    if isinstance(lookup_test_value, BaseModel):
        json_data = lookup_test_value.model_dump_json()

    else:
        json_bytes = TypeAdapter(type(lookup_test_value)).dump_json(
            lookup_test_value
        )
        json_data = json_bytes.decode("utf-8")

    data = json.loads(json_data)

    if isinstance(lookup_test_value, BaseModel):
        restored = type(lookup_test_value).model_validate(data)

    else:
        TypeAdapter(type(lookup_test_value)).validate_json(json_data)
        restored = lookup_test_value

    assert restored == lookup_test_value
