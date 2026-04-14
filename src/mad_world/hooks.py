"""Game loop callback and hook definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mad_world.core import GameState


import asyncio


class GameLoopHook(StrEnum):
    PRE_GAME = "pre_game"
    POST_PHASE = "post_phase"
    PRE_DESTROY_WORLD = "pre_destroy_world"
    POST_GAME = "post_game"


type GameLoopCallback = dict[
    GameLoopHook, Callable[["GameState"], Awaitable["GameState | None"]]
]


async def run_callbacks(
    callbacks: list[GameLoopCallback],
    game: GameState,
    hook: GameLoopHook,
    *,
    concurrent: bool = False,
) -> GameState:
    if concurrent:
        await asyncio.gather(
            *(cb(game) for cb_map in callbacks if (cb := cb_map.get(hook)))
        )
        return game

    state = game
    for cb_map in callbacks:
        if cb := cb_map.get(hook):
            state = (await cb(state)) or state
    return state
