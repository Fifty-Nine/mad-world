"""Base GamePlayer types and logic for the game."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mad_world.actions import (
        BaseAction,
        BiddingAction,
        InitialMessageAction,
        MessagingAction,
        OperationsAction,
    )
    from mad_world.core import GameState
    from mad_world.crises import GenericCrisis
    from mad_world.enums import GameOverReason
    from mad_world.rules import (
        GameRules,
    )


class GamePlayer(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    def start_game(self, game: GameRules) -> None:  # noqa: B027
        """Called with the rules for the current game
        at the start of the game.
        """

    @abstractmethod
    async def initial_message(self, game: GameState) -> InitialMessageAction:
        """Get the initial message for your opponent. This will be provided
        to them in the bidding phase of round 1.
        """

    @abstractmethod
    async def message(self, game: GameState) -> MessagingAction:
        """Get a message for your opponent before an action phase."""

    @abstractmethod
    async def bid(self, game: GameState) -> BiddingAction:
        """Get the player's input for the bidding phase, given the current
        game state."""

    @abstractmethod
    async def operations(self, game: GameState) -> OperationsAction:
        """Get the player's input for the operations phase."""

    @abstractmethod
    async def crisis[T: BaseAction](
        self,
        game: GameState,
        crisis: GenericCrisis[T],
    ) -> T:
        """Get the player's input for a pending crisis."""

    async def game_over(  # noqa: B027
        self,
        game: GameState,
        winner: str | None,
        reason: GameOverReason,
    ) -> None:
        """Called when the game is over."""
