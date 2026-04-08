from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mad_world.effects import (
    BaseOngoingEffect,
    NoDomesticInvestmentEffect,
    NoZeroBidsEffect,
)

if TYPE_CHECKING:
    from mad_world.core import GameState


def test_base_ongoing_effect_expiration() -> None:
    class DummyEffect(BaseOngoingEffect):
        card_kind: ClassVar[str] = "dummy_effect"
        title: str = "Dummy"
        description: str = "Desc"

    # Lasts indefinitely
    indefinite = DummyEffect(title="", description="")
    assert not indefinite.is_expired(1)
    assert not indefinite.is_expired(100)

    # Expires after round 2
    expiring = DummyEffect(title="", description="", expiration_round=2)
    assert not expiring.is_expired(1)
    assert not expiring.is_expired(2)
    assert expiring.is_expired(3)


def test_no_zero_bids_effect() -> None:
    effect = NoZeroBidsEffect()
    bids = [0, 1, 3, 5, 10]
    modified = effect.modify_allowed_bids(bids)
    assert 0 not in modified
    assert 1 in modified
    assert len(modified) == 4


def test_no_domestic_investment_effect(basic_game: GameState) -> None:
    effect = NoDomesticInvestmentEffect()
    ops = basic_game.rules.allowed_operations
    assert "domestic-investment" in ops

    modified = effect.modify_allowed_operations(ops)
    assert "domestic-investment" not in modified
    assert "proxy-subversion" in modified


def test_base_hooks() -> None:
    class DummyEffect(BaseOngoingEffect):
        card_kind: ClassVar[str] = "dummy_effect_2"
        title: str = ""
        description: str = ""

    effect = DummyEffect()
    assert effect.modify_allowed_bids([1, 2]) == [1, 2]
    assert effect.modify_allowed_operations({}) == {}


def test_active_effects_property_hooks(basic_game: GameState) -> None:
    effect = NoZeroBidsEffect()
    basic_game.active_effects.append(effect)

    assert 0 not in basic_game.allowed_bids

    effect_ops = NoDomesticInvestmentEffect()
    basic_game.active_effects.append(effect_ops)

    assert "domestic-investment" not in basic_game.allowed_operations


def test_advance_phase_retains_unexpired_effects(basic_game: GameState) -> None:
    effect = NoZeroBidsEffect(expiration_round=100)
    basic_game.active_effects.append(effect)
    basic_game.current_round = 1
    basic_game.advance_phase()

    assert effect in basic_game.active_effects
