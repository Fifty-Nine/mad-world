"""Utility UnimplementedPlayer class for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from mad_world.players import GamePlayer

if TYPE_CHECKING:
    from mad_world.actions import (
        BaseAction,
        BiddingAction,
        ChatAction,
        InitialMessageAction,
        MessagingAction,
        OperationsAction,
    )
    from mad_world.core import GameState
    from mad_world.crises import GenericCrisis


class UnimplementedPlayer(GamePlayer):
    """A non-abstract version of GamePlayer with every method marked as
    unimplemented. This helps reduce boilerplate in tests that need to
    construct a GamePlayer subclass but don't need to cover all
    methods."""

    @override
    async def chat(
        self, game: GameState, remaining_messages: int, last_message: str | None
    ) -> ChatAction:
        raise NotImplementedError

    @override
    async def operations(self, game: GameState) -> OperationsAction:
        raise NotImplementedError

    @override
    async def bid(self, game: GameState) -> BiddingAction:
        raise NotImplementedError

    @override
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        raise NotImplementedError

    @override
    async def message(self, game: GameState) -> MessagingAction:
        raise NotImplementedError

    @override
    async def crisis[T: BaseAction](
        self, game: GameState, crisis: GenericCrisis[T]
    ) -> T:
        raise NotImplementedError
