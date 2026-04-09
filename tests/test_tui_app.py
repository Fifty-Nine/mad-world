from __future__ import annotations

import pytest

from mad_world.config import PlayerKind, TrivialPlayerConfig
from mad_world.tui.app import MadWorldApp
from mad_world.tui.widgets import (
    CommandShell,
    EventLogDisplay,
    GameStateDisplay,
)


@pytest.mark.asyncio
async def test_app_basic() -> None:
    # Use trivial players for testing
    alpha = TrivialPlayerConfig(
        kind=PlayerKind.TRIVIAL, name="Alpha", bot_name="crazy-ivan"
    )
    omega = TrivialPlayerConfig(
        kind=PlayerKind.TRIVIAL, name="Omega", bot_name="crazy-ivan"
    )
    app = MadWorldApp(alpha_config=alpha, omega_config=omega)

    async with app.run_test() as pilot:
        # Check layout
        assert app.query_one(GameStateDisplay) is not None
        assert app.query_one(EventLogDisplay) is not None

        # Test step command
        shell = app.query_one(CommandShell)
        shell.value = "step"
        await pilot.press("enter")
        await pilot.pause(0.1)  # allow async step to finish

        # Test pause command
        shell.value = "pause"
        await pilot.press("enter")

        # Test new command
        shell.value = "new"
        await pilot.press("enter")

        # Test run command
        shell.value = "run"
        await pilot.press("enter")
        await pilot.pause(0.5)
        shell.value = "pause"
        await pilot.press("enter")

        # Quit
        shell.value = "quit"
        await pilot.press("enter")
