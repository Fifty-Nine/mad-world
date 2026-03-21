"""Tests for the trivial player implementations."""

from mad_world.core import GameRules, init_game
from mad_world.trivial_players import Capitalist, CrazyIvan, Pacifist


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


def test_pacifist_initial_message() -> None:
    """Test Pacifist's initial message."""
    player = Pacifist("TestPacifist")
    game_state = init_game([player])
    assert (
        player.initial_message(game_state)
        == "I seek only peace and prosperity for all."
    )


def test_pacifist_bid() -> None:
    """Test Pacifist's bidding logic."""
    player = Pacifist("TestPacifist")
    game_state = init_game([player])

    action = player.bid(game_state, message_from_opponent=None)

    # Pacifist always bids 0
    assert action.bid == 0
    assert (
        action.message_to_opponent
        == "Let us de-escalate tensions and work together."
    )
    assert (
        action.internal_monologue
        == "I must reduce the doomsday clock at all costs."
    )


def test_pacifist_operations() -> None:
    """Test Pacifist's operation logic."""
    player = Pacifist("TestPacifist")
    game_state = init_game([player])

    action = player.operations(game_state, message_to_opponent=None)

    assert action.operations == []
    assert action.message_to_opponent == "I offer you the hand of friendship."
    assert (
        action.internal_monologue
        == "I will not participate in these violent games."
    )


def test_capitalist_initial_message() -> None:
    """Test Capitalist's initial message."""
    player = Capitalist("TestCap")
    game_state = init_game([player])
    assert (
        player.initial_message(game_state)
        == "Greed is good. I am here to maximize shareholder value."
    )


def test_capitalist_bid() -> None:
    """Test Capitalist's bidding logic."""
    player = Capitalist("TestCap")
    game_state = init_game([player])

    action = player.bid(game_state, message_from_opponent=None)

    assert action.bid == 3
    assert action.message_to_opponent == "A rising tide lifts all boats."
    assert action.internal_monologue == "Securing capital for expansion."


def test_capitalist_operations() -> None:
    """Test Capitalist's operation logic."""
    player = Capitalist("TestCap")
    game_state = init_game([player])

    action = player.operations(game_state, message_to_opponent=None)

    assert action.operations == ["domestic-investment"]
    assert action.message_to_opponent == "Building a better tomorrow."
    assert (
        action.internal_monologue
        == "Reinvesting dividends for compound growth."
    )
