import logging
import re
from datetime import datetime
from pathlib import Path

from mad_world.__main__ import (
    create_log_session_dir,
    random_persona,
    setup_logging,
)


def test_random_persona() -> None:
    persona = random_persona()
    assert isinstance(persona, str)
    assert " " in persona
    assert re.match(r"^([A-Z][a-z]+) ([A-Z][a-z]+)$", persona) is not None


def test_create_log_session_dir(tmp_path: Path) -> None:
    timestamp = datetime(2026, 3, 25, 20, 50, 23)
    log_dir = create_log_session_dir(
        tmp_path,
        "Alpha",
        "PersonaA",
        "ModelA",
        "Omega",
        "PersonaB",
        "ModelB",
        timestamp=timestamp,
    )

    expected_name = (
        "Alpha-PersonaA-ModelA-vs-Omega-PersonaB-ModelB.2026-03-25T20-50-23"
    )
    assert log_dir.name == expected_name
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_setup_logging(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    setup_logging(log_dir)

    logger = logging.getLogger()
    # Check if handlers were added
    handler_types = [type(h) for h in logger.handlers]
    assert logging.FileHandler in handler_types
    assert logging.StreamHandler in handler_types

    debug_file = log_dir / "debug.txt"
    log_file = log_dir / "log.txt"
    assert debug_file.exists()
    assert log_file.exists()
