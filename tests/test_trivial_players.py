"""Tests for the trivial player implementations."""

from mad_world.core import GamePhase, init_game
from mad_world.rules import GameRules
from mad_world.trivial_players import (
    Capitalist,
    CrazyIvan,
    Diplomat,
    Pacifist,
    Saboteur,
)


def test_crazy_ivan_initial_message() -> None:
    """Test Crazy Ivan's initial message."""
    player = CrazyIvan("TestIvan")
    game_state = init_game([player])
    assert (
        player.initial_message(game_state).message_to_opponent
        == "I'm crazy Ivan. Prepare to die!"
    )


def test_crazy_ivan_bid() -> None:
    """Test Crazy Ivan's bidding logic."""
    player = CrazyIvan("TestIvan")
    rules = GameRules()
    game_state = init_game([player], rules)

    action = player.bid(game_state)

    # Ivan always bids the maximum allowed
    assert action.bid == max(rules.allowed_bids)


def test_crazy_ivan_operations() -> None:
    """Test Crazy Ivan's operation logic."""
    player = CrazyIvan("TestIvan")
    game_state = init_game([player])

    action = player.operations(game_state)

    assert action.operations == ["first-strike"]


def test_crazy_ivan_message() -> None:
    """Test Crazy Ivan's message logic."""
    player = CrazyIvan("TestIvan")
    game_state = init_game([player])

    action = player.message(game_state)
    assert action.message_to_opponent is None


def test_pacifist_initial_message() -> None:
    """Test Pacifist's initial message."""
    player = Pacifist("TestPacifist")
    game_state = init_game([player])
    assert (
        player.initial_message(game_state).message_to_opponent
        == "I seek only peace and prosperity for all."
    )


def test_pacifist_bid() -> None:
    """Test Pacifist's bidding logic."""
    player = Pacifist("TestPacifist")
    game_state = init_game([player])

    action = player.bid(game_state)

    # Pacifist always bids 0
    assert action.bid == 0


def test_pacifist_operations() -> None:
    """Test Pacifist's operation logic."""
    player = Pacifist("TestPacifist")
    game_state = init_game([player])

    action = player.operations(game_state)

    assert action.operations == []


def test_pacifist_message() -> None:
    """Test Pacifist's message logic."""
    player = Pacifist("TestPacifist")
    game_state = init_game([player])

    game_state.current_phase = GamePhase.BIDDING_MESSAGING
    action = player.message(game_state)
    assert (
        action.message_to_opponent
        == "Let us de-escalate tensions and work together."
    )

    game_state.current_phase = GamePhase.OPERATIONS_MESSAGING
    action = player.message(game_state)
    assert action.message_to_opponent == "I offer you the hand of friendship."


def test_capitalist_initial_message() -> None:
    """Test Capitalist's initial message."""
    player = Capitalist("TestCap")
    game_state = init_game([player])
    assert (
        player.initial_message(game_state).message_to_opponent
        == "Greed is good. I am here to maximize shareholder value."
    )


def test_capitalist_bid() -> None:
    """Test Capitalist's bidding logic."""
    player = Capitalist("TestCap")
    game_state = init_game([player])

    action = player.bid(game_state)

    assert action.bid == 3


def test_capitalist_operations() -> None:
    """Test Capitalist's operation logic."""
    player = Capitalist("TestCap")
    game_state = init_game([player])

    action = player.operations(game_state)

    assert action.operations == ["domestic-investment"]


def test_capitalist_message() -> None:
    """Test Capitalist's message logic."""
    player = Capitalist("TestCap")
    game_state = init_game([player])

    game_state.current_phase = GamePhase.BIDDING_MESSAGING
    action = player.message(game_state)
    assert action.message_to_opponent == "A rising tide lifts all boats."

    game_state.current_phase = GamePhase.OPERATIONS_MESSAGING
    action = player.message(game_state)
    assert action.message_to_opponent == "Building a better tomorrow."


def test_saboteur_initial_message() -> None:
    """Test Saboteur's initial message."""
    player = Saboteur("TestSaboteur")
    game_state = init_game([player])
    assert (
        player.initial_message(game_state).message_to_opponent
        == "We look forward to a long and mutually beneficial relationship..."
    )


def test_saboteur_bid() -> None:
    """Test Saboteur's bidding logic."""
    player = Saboteur("TestSaboteur")
    game_state = init_game([player])

    action = player.bid(game_state)

    assert action.bid == 1


def test_saboteur_operations_insufficient_influence() -> None:
    """Test Saboteur's operation logic without enough influence."""
    player = Saboteur("TestSaboteur")
    game_state = init_game([player])

    # Set influence to 3 (needs 4 for proxy-subversion)
    game_state.players["TestSaboteur"].influence = 3

    action = player.operations(game_state)

    assert action.operations == []


def test_saboteur_operations_sufficient_influence() -> None:
    """Test Saboteur's operation logic with enough influence."""
    player = Saboteur("TestSaboteur")
    game_state = init_game([player])

    # Set influence to 4 (needs 4 for proxy-subversion)
    game_state.players["TestSaboteur"].influence = 4

    action = player.operations(game_state)

    assert action.operations == ["proxy-subversion"]


def test_saboteur_message() -> None:
    """Test Saboteur's message logic."""
    player = Saboteur("TestSaboteur")
    game_state = init_game([player])

    game_state.current_phase = GamePhase.BIDDING_MESSAGING
    action = player.message(game_state)
    assert (
        action.message_to_opponent
        == "Just moving some paperwork around. Administrative things."
    )

    game_state.current_phase = GamePhase.OPERATIONS_MESSAGING
    game_state.players["TestSaboteur"].influence = 4
    action = player.message(game_state)
    assert action.message_to_opponent == (
        "Oh, did your infrastructure spontaneously combust? "
        "Must be the weather."
    )

    game_state.players["TestSaboteur"].influence = 3
    action = player.message(game_state)
    assert (
        action.message_to_opponent
        == "Everything is quiet on the western front."
    )


def test_diplomat_initial_message() -> None:
    """Test Diplomat's initial message."""
    player = Diplomat("TestDiplomat")
    game_state = init_game([player])
    assert (
        player.initial_message(game_state).message_to_opponent
        == "I believe we can resolve our differences through dialogue."
    )


def test_diplomat_bid() -> None:
    """Test Diplomat's bidding logic."""
    player = Diplomat("TestDiplomat")
    game_state = init_game([player])

    action = player.bid(game_state)

    assert action.bid == 1


def test_diplomat_operations_insufficient_influence() -> None:
    """Test Diplomat's operation logic without enough influence."""
    player = Diplomat("TestDiplomat")
    game_state = init_game([player])

    # Set influence to 4 (needs 5 for unilateral-drawdown)
    game_state.players["TestDiplomat"].influence = 4

    action = player.operations(game_state)

    assert action.operations == []


def test_diplomat_operations_sufficient_influence() -> None:
    """Test Diplomat's operation logic with enough influence."""
    player = Diplomat("TestDiplomat")
    game_state = init_game([player])

    # Set influence to 5 (needs 5 for unilateral-drawdown)
    game_state.players["TestDiplomat"].influence = 5

    action = player.operations(game_state)

    assert action.operations == ["unilateral-drawdown"]


def test_diplomat_message() -> None:
    """Test Diplomat's message logic."""
    player = Diplomat("TestDiplomat")
    game_state = init_game([player])

    game_state.current_phase = GamePhase.BIDDING_MESSAGING
    action = player.message(game_state)
    assert (
        action.message_to_opponent
        == "Let us keep the channels of communication open."
    )

    game_state.current_phase = GamePhase.OPERATIONS_MESSAGING
    game_state.players["TestDiplomat"].influence = 5
    action = player.message(game_state)
    assert action.message_to_opponent == (
        "I invite you to the negotiating table. "
        "Let us step back from the brink."
    )

    game_state.players["TestDiplomat"].influence = 4
    action = player.message(game_state)
    assert (
        action.message_to_opponent == "We must continue our diplomatic efforts."
    )
