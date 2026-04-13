"""Game loop callback and hook definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import pydash

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mad_world.core import GameState


class GameLoopHook(StrEnum):
    POST_PHASE = "post_phase"
    PRE_DESTROY_WORLD = "pre_destroy_world"


type GameLoopCallback = dict[
    GameLoopHook, Callable[["GameState"], Awaitable["GameState | None"]]
]


async def run_callbacks(
    callbacks: list[GameLoopCallback], game: GameState, hook: GameLoopHook
) -> GameState:
    async def reducer(
        state_coro: Awaitable[GameState], cb_map: GameLoopCallback
    ) -> GameState:
        state = await state_coro
        cb = cb_map.get(hook)
        if cb is not None:
            return (await cb(state)) or state
        return state

    async def initial() -> GameState:
        return game

    return await pydash.reduce_(callbacks, reducer, initial())
