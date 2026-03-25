import pytest

from mad_world.rules import (
    GameRules,
    OperationDefinition,
    cost_or_gain,
    increase_or_decrease,
)


@pytest.mark.parametrize(
    "clock,max_clock,allowed_bids,risky,deadly",
    [
        (0, 25, (0, 1, 3, 5, 8), [], []),
        (24, 25, (0, 1, 3, 5, 8), [(0, 3)], [1, 3, 5, 8]),
        (20, 25, (0, 1, 3, 5, 8), [(0, 8), (1, 5), (3, 3)], [5, 8]),
        (10, 25, (0, 1, 3, 5, 8), [(8, 8)], []),
        (0, 1, (0, 1, 3, 5, 8), [(0, 3)], [1, 3, 5, 8]),
        (0, 1, (0,), [], []),
    ],
)
def test_get_doomsday_bids(
    clock: int,
    max_clock: int,
    allowed_bids: list[int],
    risky: list[tuple[int, int]],
    deadly: list[int],
) -> None:
    rules = GameRules()
    rules.max_clock_state = max_clock
    rules.allowed_bids = allowed_bids

    assert rules.get_doomsday_bids(clock) == (risky, deadly)


@pytest.mark.parametrize(
    "value,inc_dec,cost_gain",
    [
        (100, "increase", "gain"),
        (-100, "decrease", "cost"),
        (0, "increase", "gain"),
        (-1, "decrease", "cost"),
        (1, "increase", "gain"),
    ],
)
def test_increase_or_decrease(value: int, inc_dec: str, cost_gain: str) -> None:
    assert increase_or_decrease(value) == inc_dec
    assert cost_or_gain(value) == cost_gain


def test_operation_effect_formatting() -> None:
    assert OperationDefinition.format_one(0, "foo", lambda i: "") == ""
    assert (
        OperationDefinition.format_one(1, "bar", increase_or_decrease)
        == "  bar increase: 1\n"
    )
    assert (
        OperationDefinition.format_one(-1, "baz", cost_or_gain)
        == "  baz cost: 1\n"
    )
    assert (
        OperationDefinition.format_one(-1, "quux", increase_or_decrease)
        == "  quux decrease: 1\n"
    )


def test_operation_formatting() -> None:
    test_op = OperationDefinition(
        name="foo",
        description="bar",
        influence_cost=1,
        enemy_influence_effect=999,
        clock_effect=-1,
        friendly_gdp_effect=2,
        enemy_gdp_effect=-3,
    )

    assert test_op.format(verbose=True) == (
        "foo:\n"
        "  Description:\n"
        "    bar\n"
        "  Inf cost: 1\n"
        "  Opponent Inf increase: 999\n"
        "  Clock decrease: 1\n"
        "  GDP increase: 2\n"
        "  Opponent GDP decrease: 3\n"
    )

    assert test_op.format(verbose=False, indent="x") == (
        "xfoo:\n"
        "x  Inf cost: 1\n"
        "x  Opponent Inf increase: 999\n"
        "x  Clock decrease: 1\n"
        "x  GDP increase: 2\n"
        "x  Opponent GDP decrease: 3\n"
    )

    test_op.name = "first-strike"
    assert test_op.format(verbose=True) == (
        "first-strike:\n"
        "  Description:\n"
        "    bar\n"
        "  Cost: everything\n"
        "  Gain: a legacy of ashes, but at "
        "least your opponent doesn't win.\n"
    )

    test_op = OperationDefinition(
        name="do-nothing", description="?", influence_cost=0
    )
    assert test_op.format(verbose=False, indent="y") == ("ydo-nothing:\n")
    assert test_op.format(verbose=True) == (
        "do-nothing:\n  Description:\n    ?\n"
    )
