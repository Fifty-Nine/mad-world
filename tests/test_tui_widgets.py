from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from mad_world.core import GameState
from mad_world.rules import GameRules
from mad_world.tui.widgets import (
    CommandShell,
    EventLogDisplay,
    GameStateDisplay,
)


class DummyApp(App[None]):
    def compose(self) -> ComposeResult:
        yield GameStateDisplay()
        yield EventLogDisplay()
        yield CommandShell()


@pytest.mark.asyncio
async def test_widgets() -> None:
    app = DummyApp()
    async with app.run_test():
        gsd = app.query_one(GameStateDisplay)
        rules = GameRules()
        game = GameState.new_game(rules=rules, players=["P1", "P2"])
        gsd.update_state(game)

        eld = app.query_one(EventLogDisplay)
        eld.append_events(["Event 1", "Event 2"])

        cs = app.query_one(CommandShell)
        assert cs is not None
