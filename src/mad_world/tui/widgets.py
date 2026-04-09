"""Custom widgets for the Mad World TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Input, Log, Static

if TYPE_CHECKING:
    from mad_world.core import GameState


class GameStateDisplay(Static):  # pragma: no cover
    """Displays the current state of the game."""

    def update_state(self, game: GameState) -> None:
        """Update the displayed game state."""
        self.update(game.describe_state())


class EventLogDisplay(Log):  # pragma: no cover
    """Displays the event log."""

    def append_events(self, events: list[str]) -> None:
        """Append a list of text events to the log."""
        for event in events:
            self.write(event)
            self.write("")  # Blank line separator


class CommandShell(Input):
    """Command shell for the GM."""
