from engine.board import Board, Color
from engine.moves import generate_move_sequences


def _white_home_only(dist: dict) -> Board:
    """Board with all white in home per `dist` (point -> count), black parked
    deep in its own home so it does not interfere."""
    b = Board()
    b.points[24].count = 0
    b.points[24].color = None
    b.points[12].count = 0
    b.points[12].color = None
    for pt, cnt in dist.items():
        b.points[pt].count = cnt
        b.points[pt].color = Color.WHITE
    # Park black at 13 (its head); ensures opponents exist.
    b.points[13].count = 15
    b.points[13].color = Color.BLACK
    return b


class TestBearOff:
    def test_exact_bear_off(self):
        b = _white_home_only({6: 15})
        seqs = generate_move_sequences(b, Color.WHITE, (6, 6), is_first_roll=False)
        assert seqs, "expected at least one sequence"
        # All 4 dice bear off from 6.
        for seq in seqs:
            assert all(m.is_bear_off for m in seq)

    def test_overshoot_when_higher_empty(self):
        b = _white_home_only({3: 15})
        seqs = generate_move_sequences(b, Color.WHITE, (6, 5), is_first_roll=False)
        # Each die overshoots — bear off from pt 3 (highest occupied).
        assert any(any(m.is_bear_off and m.from_point == 3 for m in s) for s in seqs)

    def test_overshoot_blocked_when_higher_occupied(self):
        # With pt 5 occupied and pt 3 occupied: die=6 from pt 3 must not
        # bear off (overshoot is blocked because pt 5 is higher than pt 3
        # in step terms and still has a checker). die=1 from pt 3 lands
        # on pt 2 (not a bear-off). So pt 3 must never appear as a
        # bear-off source in any legal sequence here.
        b = _white_home_only({5: 1, 3: 14})
        seqs = generate_move_sequences(b, Color.WHITE, (6, 1), is_first_roll=False)
        for s in seqs:
            for m in s:
                assert not (m.is_bear_off and m.from_point == 3)

    def test_cannot_bear_off_before_all_in_home(self):
        # White has one checker at pt 24 (head), rest in home. With dice
        # (6, 5) — only two moves — the head checker cannot reach home in
        # a single turn, so no bear-off may appear in any sequence.
        b = Board()
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        b.points[6].count = 14
        b.points[6].color = Color.WHITE
        b.points[12].count = 0
        b.points[12].color = None
        b.points[13].count = 15
        b.points[13].color = Color.BLACK
        seqs = generate_move_sequences(b, Color.WHITE, (6, 5), is_first_roll=False)
        for s in seqs:
            for m in s:
                assert not m.is_bear_off
