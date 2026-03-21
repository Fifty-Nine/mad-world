"""Tests for the trivial player implementations."""

from mad_world.core import GameRules, init_game
from mad_world.trivial_players import CrazyIvan


def test_crazy_ivan_initial_message() -> None:
    """Test Crazy Ivan's initial message."""
    player = CrazyIvan("TestIvan")
    game_state = init_game([player])
    assert (
        player.initial_message(game_state) == "I'm crazy Ivan. Prepare to die!"
    )


def test_crazy_ivan_bid() -> None:
    """Test Crazy Ivan's bidding logic."""
    player = CrazyIvan("TestIvan")
    rules = GameRules()
    game_state = init_game([player], rules)

    action = player.bid(game_state, message_from_opponent=None)

    # Ivan always bids the maximum allowed
    assert action.bid == max(rules.allowed_bids)
    assert action.message_to_opponent is None
    assert action.internal_monologue == "No thoughts, head empty."


def test_crazy_ivan_operations() -> None:
    """Test Crazy Ivan's operation logic."""
    player = CrazyIvan("TestIvan")
    game_state = init_game([player])

    action = player.operations(game_state, message_to_opponent=None)

    assert action.operations == ["first-strike"]
    assert action.message_to_opponent is None
    assert action.internal_monologue == "I'm crazy!"
