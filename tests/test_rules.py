from mad_world.rules import GameRules


def test_get_doomsday_bids_safe_clock() -> None:
    rules = GameRules()
    risky, deadly = rules.get_doomsday_bids(0)
    assert risky == []
    assert deadly == []


def test_get_doomsday_bids_clock_20() -> None:
    # allowed_bids = [0, 1, 3, 5, 8], max_clock_state = 25
    rules = GameRules()
    risky, deadly = rules.get_doomsday_bids(20)

    # 20 + 8 = 28 >= 25 (deadly)
    # 20 + 5 = 25 >= 25 (deadly)
    assert deadly == [5, 8]

    # 20 + 3 + 3 = 26 >= 25 -> obid 3
    # 20 + 1 + 5 = 26 >= 25 -> obid 5
    # 20 + 0 + 5 = 25 >= 25 -> obid 5, but 0 => -1
    assert risky == [(0, 8), (1, 5), (3, 3)]


def test_get_doomsday_bids_clock_23() -> None:
    # allowed_bids = [0, 1, 3, 5, 8], max_clock_state = 25
    rules = GameRules()
    risky, deadly = rules.get_doomsday_bids(23)

    assert deadly == [3, 5, 8]

    # 23 + 1 + 1 = 25 >= 25 -> obid 1
    # 23 + 0 + 3 = 26 >= 25 -> obid 3
    assert risky == [(0, 3), (1, 1)]


def test_get_doomsday_bids_clock_24() -> None:
    rules = GameRules()
    risky, deadly = rules.get_doomsday_bids(24)

    # At 24, anything but 0 is deadly.
    assert deadly == [1, 3, 5, 8]
    assert risky == [(0, 3)]
