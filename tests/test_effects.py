"""Tests for game effects and event cards that apply them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mad_world.effects import (
    ArmsControlEffect,
    NoDomesticInvestmentEffect,
    NoZeroBidsEffect,
    ProxyWarEscalationEffect,
    SupplyChainShockEffect,
    UNPeacekeepingEffect,
)
from mad_world.enums import GamePhase
from mad_world.event_cards import (
    ArmsControlTreatyEvent,
    BanDomesticInvestmentEvent,
    BanZeroBidsEvent,
    ProxyWarEscalationEvent,
    SupplyChainShockEvent,
)
from mad_world.events import ActorKind

if TYPE_CHECKING:
    from mad_world.core import GameState


def test_base_effect(basic_game: GameState) -> None:
    # Use NoDomesticInvestmentEffect since it uses default modify_bids
    effect = NoDomesticInvestmentEffect(duration=None)
    effect.start_round = basic_game.current_round

    # default modify_bids should not modify bids
    bids = basic_game.allowed_bids
    assert effect.modify_bids(bids) == bids

    # Never expires if no expiration_round
    assert not effect.is_expired(basic_game)

    # Test on_expire
    events = effect.on_expire(basic_game)
    assert len(events) == 1
    assert "has expired" in events[0].description
    assert events[0].actor.actor_kind == ActorKind.SYSTEM


def test_expiration_logic(basic_game: GameState) -> None:
    effect = NoZeroBidsEffect(
        duration=2,
    )
    basic_game.current_round = 1

    effect.start_round = basic_game.current_round
    assert not effect.is_expired(basic_game)

    basic_game.current_round = 2
    assert not effect.is_expired(basic_game)

    basic_game.current_round = 3
    assert effect.is_expired(basic_game)


def test_no_zero_bids_effect(basic_game: GameState) -> None:
    effect = NoZeroBidsEffect(duration=None)
    bids = effect.modify_bids([0, 1, 3, 5])
    assert bids == [1, 3, 5]


def test_arms_control_effect(basic_game: GameState) -> None:
    effect = ArmsControlEffect(duration=None)
    bids = effect.modify_bids([0, 1, 3, 5, 10])
    assert bids == [0, 1, 3]


def test_un_peacekeeping_effect(basic_game: GameState) -> None:
    effect = UNPeacekeepingEffect(duration=None)
    ops = basic_game.allowed_operations
    assert "proxy-subversion" in ops
    assert "conventional-offensive" in ops

    modified_ops = effect.modify_operations(ops)
    assert "proxy-subversion" not in modified_ops
    assert "conventional-offensive" not in modified_ops
    assert "aggressive-extraction" in modified_ops


def test_no_domestic_investment_effect(basic_game: GameState) -> None:
    effect = NoDomesticInvestmentEffect(duration=None)
    ops = basic_game.allowed_operations
    assert "domestic-investment" in ops

    modified_ops = effect.modify_operations(ops)
    assert "domestic-investment" not in modified_ops
    assert "aggressive-extraction" in modified_ops


def test_supply_chain_shock_effect(basic_game: GameState) -> None:
    effect = SupplyChainShockEffect(duration=None)
    ops = basic_game.allowed_operations

    modified_ops = effect.modify_operations(ops)

    assert (
        modified_ops["domestic-investment"].influence_cost
        == ops["domestic-investment"].influence_cost + 1
    )
    assert (
        modified_ops["proxy-subversion"].influence_cost
        == ops["proxy-subversion"].influence_cost + 1
    )

    assert (
        modified_ops["first-strike"].influence_cost
        == ops["first-strike"].influence_cost
    )


def test_proxy_war_escalation_effect(basic_game: GameState) -> None:
    effect = ProxyWarEscalationEffect(duration=None)
    ops = basic_game.allowed_operations

    modified_ops = effect.modify_operations(ops)

    assert (
        modified_ops["proxy-subversion"].clock_effect
        == ops["proxy-subversion"].clock_effect + 1
    )
    assert (
        modified_ops["conventional-offensive"].clock_effect
        == ops["conventional-offensive"].clock_effect + 1
    )

    assert (
        modified_ops["domestic-investment"].clock_effect
        == ops["domestic-investment"].clock_effect
    )


def test_ban_zero_bids_event(basic_game: GameState) -> None:
    event = BanZeroBidsEvent()
    basic_game.current_round = 5

    events = event.run(basic_game)
    assert len(events) == 1
    assert "has been applied" in events[0].description

    basic_game.apply_event(events[0])

    assert len(basic_game.active_effects) == 1
    effect = basic_game.active_effects[0]
    assert isinstance(effect, NoZeroBidsEffect)
    assert effect.duration == 2


def test_ban_domestic_investment_event(basic_game: GameState) -> None:
    event = BanDomesticInvestmentEvent()
    basic_game.current_round = 2

    events = event.run(basic_game)
    assert len(events) == 1

    basic_game.apply_event(events[0])

    assert len(basic_game.active_effects) == 1
    effect = basic_game.active_effects[0]
    assert isinstance(effect, NoDomesticInvestmentEffect)
    assert effect.duration == 2


def test_arms_control_treaty_event(basic_game: GameState) -> None:
    event = ArmsControlTreatyEvent()
    basic_game.current_round = 2

    events = event.run(basic_game)
    assert len(events) == 1
    assert "has been applied" in events[0].description

    basic_game.apply_event(events[0])

    assert len(basic_game.active_effects) == 1
    effect = basic_game.active_effects[0]
    assert isinstance(effect, ArmsControlEffect)
    assert effect.duration == 2


def test_supply_chain_shock_event(basic_game: GameState) -> None:
    event = SupplyChainShockEvent()
    basic_game.current_round = 2

    events = event.run(basic_game)
    assert len(events) == 1
    assert "has been applied" in events[0].description

    basic_game.apply_event(events[0])

    assert len(basic_game.active_effects) == 1
    effect = basic_game.active_effects[0]
    assert isinstance(effect, SupplyChainShockEffect)
    assert effect.duration == 2


def test_proxy_war_escalation_event(basic_game: GameState) -> None:
    event = ProxyWarEscalationEvent()
    basic_game.current_round = 2

    events = event.run(basic_game)
    assert len(events) == 1
    assert "has been applied" in events[0].description

    basic_game.apply_event(events[0])

    assert len(basic_game.active_effects) == 1
    effect = basic_game.active_effects[0]
    assert isinstance(effect, ProxyWarEscalationEffect)
    assert effect.duration == 2


def test_active_effects_integration(basic_game: GameState) -> None:
    """Test that properties actually use active_effects properly."""
    effect = NoZeroBidsEffect(duration=None)
    basic_game.active_effects.append(effect)

    assert 0 not in basic_game.allowed_bids

    # Check that ops are unmodified
    assert "domestic-investment" in basic_game.allowed_operations


def test_advance_phase_expiration_integration(basic_game: GameState) -> None:
    """Test that advance_phase successfully removes expired effects."""
    basic_game.current_phase = GamePhase.OPERATIONS
    basic_game.current_round = 1

    # Adding an effect that expires after round 1
    effect = NoZeroBidsEffect(duration=0)
    effect.start_round = basic_game.current_round
    basic_game.active_effects.append(effect)

    # Add an effect that expires later (round 3)
    later_effect = NoDomesticInvestmentEffect(duration=2)
    later_effect.start_round = basic_game.current_round
    basic_game.active_effects.append(later_effect)

    assert 0 not in basic_game.allowed_bids

    # Advance phase will transition from OPERATIONS to ROUND_EVENTS
    # and increment round to 2
    basic_game.advance_phase()

    assert basic_game.current_phase == GamePhase.ROUND_EVENTS
    assert basic_game.current_round == 2

    # The effect should be removed, but later_effect should remain
    assert len(basic_game.active_effects) == 1
    assert basic_game.active_effects[0] is later_effect
    assert 0 in basic_game.allowed_bids

    # The expiration event should have been logged
    log = basic_game.event_log
    # Check if the "has expired" event is there
    assert any("has expired" in e.event.description for e in log)
