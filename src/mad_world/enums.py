"""Enum definitions for the game."""

from __future__ import annotations

from enum import Enum


class GamePhase(Enum):
    OPENING = 1
    BIDDING_MESSAGING = 2
    BIDDING = 3
    OPERATIONS_MESSAGING = 4
    OPERATIONS = 5
    END = 6


class GameOverReason(Enum):
    WORLD_DESTROYED = 1
    ECONOMIC_VICTORY = 2
    STALEMATE = 3


class StandoffPosture(Enum):
    BACK_DOWN = (1,)
    STAND_FIRM = 2
