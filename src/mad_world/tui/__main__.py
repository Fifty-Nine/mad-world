"""Entry point for the Mad World TUI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from jsonargparse import CLI

from mad_world.__main__ import (
    DEFAULT_ALPHA,
    DEFAULT_OMEGA,
    _default_config,
    _get_explicitly_set_params,
)
from mad_world.tui.app import MadWorldApp

if TYPE_CHECKING:
    from mad_world.config import PlayerConfig


def run_tui(  # pragma: no cover
    alpha: PlayerConfig = DEFAULT_ALPHA,
    omega: PlayerConfig = DEFAULT_OMEGA,
) -> None:
    """Run the Mad World TUI."""
    alpha = _default_config(
        alpha, _get_explicitly_set_params(sys.argv, "alpha")
    )
    omega = _default_config(
        omega, _get_explicitly_set_params(sys.argv, "omega")
    )

    app = MadWorldApp(alpha_config=alpha, omega_config=omega)
    app.run()


def main() -> None:  # pragma: no cover
    CLI(run_tui, prog="mad_world_tui")  # type: ignore[no-untyped-call]


if __name__ == "__main__":  # pragma: no cover
    main()
