"""Mad World TUI Application."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

from mad_world.__main__ import DEFAULT_ALPHA, DEFAULT_OMEGA, get_player
from mad_world.core import GameManager
from mad_world.rules import GameRules
from mad_world.tui.widgets import (
    CommandShell,
    EventLogDisplay,
    GameStateDisplay,
)

if TYPE_CHECKING:
    from textual.worker import Worker

    from mad_world.config import PlayerConfig


class MadWorldApp(App[None]):  # pragma: no cover
    """The main TUI App for Mad World."""

    CSS = """
    GameStateDisplay {
        height: 10;
        border: solid green;
    }
    EventLogDisplay {
        height: 1fr;
        border: solid blue;
    }
    CommandShell {
        dock: bottom;
    }
    """

    def __init__(
        self,
        alpha_config: PlayerConfig | None = None,
        omega_config: PlayerConfig | None = None,
    ) -> None:
        super().__init__()
        self.manager: GameManager | None = None
        self.alpha_config = alpha_config or DEFAULT_ALPHA
        self.omega_config = omega_config or DEFAULT_OMEGA
        self._game_worker: Worker[None] | None = None
        self._last_event_count: int = 0

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()
        yield GameStateDisplay(id="state")
        yield EventLogDisplay(id="log")
        yield CommandShell(
            placeholder="Enter command (new, step, run, pause, quit)",
            id="shell",
        )
        yield Footer()

    async def on_mount(self) -> None:
        """Called when app is mounted."""
        self.query_one(CommandShell).focus()
        await self.start_new_game()

    async def start_new_game(self) -> None:
        """Initialize a new game."""
        log_dir = Path("./logs")  # Simplification for now
        logger = logging.getLogger("mad_world")
        players = [
            get_player(
                self.alpha_config, self.omega_config.name, log_dir, logger
            ),
            get_player(
                self.omega_config, self.alpha_config.name, log_dir, logger
            ),
        ]
        self.manager = GameManager(GameRules(), players)
        await self.manager.start()
        self._last_event_count = 0
        self.query_one(EventLogDisplay).clear()
        self.query_one(EventLogDisplay).write("--- NEW GAME ---")
        self.update_display()

    def update_display(self) -> None:
        """Update the UI with current state and new events."""
        if self.manager is None:  # pragma: no cover
            return

        # Update State
        self.query_one(GameStateDisplay).update_state(self.manager.game)

        # Append new events
        events = self.manager.game.event_log
        if len(events) > self._last_event_count:
            new_events = events[self._last_event_count :]
            log_display = self.query_one(EventLogDisplay)
            log_display.append_events([e.description for e in new_events])
            self._last_event_count = len(events)

    async def on_input_submitted(self, message: CommandShell.Submitted) -> None:
        """Handle shell commands."""
        command = message.value.strip().lower()
        message.input.value = ""

        if command == "quit":
            self.exit()
        elif command == "new":
            self.pause_game()
            await self.start_new_game()
        elif command == "step":
            if self.manager and not self.manager.game_over:
                await self.manager.step()
                self.update_display()
        elif command == "run":
            if self._game_worker is None or not self._game_worker.is_running:
                self._game_worker = self.run_game_loop()
        elif command == "pause":
            self.pause_game()

    def pause_game(self) -> None:
        if self._game_worker and self._game_worker.is_running:
            self._game_worker.cancel()  # pragma: no cover

    @work(exclusive=True)
    async def run_game_loop(self) -> None:
        """Asynchronous worker to run the game continuously."""
        while self.manager and not self.manager.game_over:
            await self.manager.step()
            self.update_display()
            await asyncio.sleep(0.0)  # Yield control
