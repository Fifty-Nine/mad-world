from __future__ import annotations

from unittest.mock import patch

from mad_world.config import PlayerKind, TrivialPlayerConfig
from mad_world.tui.__main__ import main, run_tui


def test_run_tui() -> None:
    alpha = TrivialPlayerConfig(
        kind=PlayerKind.TRIVIAL, name="Alpha", bot_name="crazy-ivan"
    )
    omega = TrivialPlayerConfig(
        kind=PlayerKind.TRIVIAL, name="Omega", bot_name="crazy-ivan"
    )

    with patch("mad_world.tui.app.MadWorldApp.run") as mock_run:
        run_tui(alpha, omega)
        mock_run.assert_called_once()


def test_main() -> None:
    with patch("mad_world.tui.__main__.CLI") as mock_cli:
        main()
        mock_cli.assert_called_once()
