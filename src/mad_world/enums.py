"""Enum definitions for the game."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class GamePhase(IntEnum):
    OPENING = 1
    BIDDING_MESSAGING = 2
    BIDDING = 3
    OPERATIONS_MESSAGING = 4
    OPERATIONS = 5
    CRISIS_MESSAGING = 6
    CRISIS = 7
    END = 8
    ROUND_EVENTS = 9

    def is_messaging(self) -> bool:
        return self in (
            self.BIDDING_MESSAGING,
            self.OPERATIONS_MESSAGING,
            self.CRISIS_MESSAGING,
        )

    def is_crisis(self) -> bool:
        return self in (self.CRISIS_MESSAGING, self.CRISIS)


class GameOverReason(IntEnum):
    WORLD_DESTROYED = 1
    ECONOMIC_VICTORY = 2
    STALEMATE = 3


class StandoffPosture(StrEnum):
    BACK_DOWN = "back down"
    STAND_FIRM = "stand firm"


class BlameGamePosture(StrEnum):
    SHOULDER = "shoulder"
    DEFLECT = "deflect"


class OpenChannelPreference(StrEnum):
    REQUEST = "request"
    ACCEPT = "accept"
    REJECT = "reject"
