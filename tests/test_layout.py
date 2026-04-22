from ui.layout import (
    BoardLayout, checker_positions, dice_slot_count, dice_slot_offsets,
    dice_slot_values,
)


class TestBoardLayout:
    def test_board_centered(self):
        L = BoardLayout(screen_w=1280, screen_h=720)
        assert L.board_left == 128
        assert L.board_top == 40
        assert L.board_width == 1024
        assert L.board_height == 640

    def test_point_rects_top_row(self):
        L = BoardLayout()
        r24 = L.point_rect(24)
        r19 = L.point_rect(19)
        assert r24.y == L.board_top
        assert r19.y == L.board_top
        assert r24.x > r19.x  # 24 is to the right of 19 in the top row

    def test_point_rects_bottom_row(self):
        L = BoardLayout()
        r1 = L.point_rect(1)
        r6 = L.point_rect(6)
        assert r1.y > L.board_top
        assert r1.x > r6.x  # pt 1 is rightmost in bottom row

    def test_bar_between_halves(self):
        L = BoardLayout()
        r18 = L.point_rect(18)
        r19 = L.point_rect(19)
        assert r19.x > r18.x + r18.w


class TestCheckerPositions:
    def test_six_checkers_stacked(self):
        L = BoardLayout()
        positions = checker_positions(point=24, count=6, layout=L, top_row=True)
        assert len(positions) == 6
        ys = [p[1] for p in positions]
        assert ys == sorted(ys)

    def test_bottom_row_stacks_upward(self):
        L = BoardLayout()
        positions = checker_positions(point=1, count=6, layout=L, top_row=False)
        ys = [p[1] for p in positions]
        assert ys == sorted(ys, reverse=True)

    def test_overflow_overlaps(self):
        L = BoardLayout()
        positions = checker_positions(point=24, count=15, layout=L, top_row=True)
        assert len(positions) == 15


class TestDiceSlotCount:
    def test_non_double_has_two_slots(self):
        assert dice_slot_count((2, 5)) == 2

    def test_double_has_four_slots(self):
        assert dice_slot_count((3, 3)) == 4
        assert dice_slot_count((6, 6)) == 4


class TestDiceSlotOffsets:
    def test_non_double_is_vertical_stack(self):
        # Two slots stacked vertically: same x, increasing y.
        offs = dice_slot_offsets((2, 5))
        assert len(offs) == 2
        assert offs[0][0] == offs[1][0]
        assert offs[1][1] > offs[0][1]

    def test_double_is_2x2_grid(self):
        # 4 slots: two columns, two rows. So we get exactly two distinct
        # x-values and two distinct y-values.
        offs = dice_slot_offsets((4, 4))
        assert len(offs) == 4
        xs = {x for x, _ in offs}
        ys = {y for _, y in offs}
        assert len(xs) == 2
        assert len(ys) == 2


class TestDiceSlotValues:
    def test_non_double_shows_both_values(self):
        # Non-double: slot 0 shows dice[0], slot 1 shows dice[1].
        assert dice_slot_values((3, 5)) == [3, 5]

    def test_double_repeats_value_four_times(self):
        # Double: all four slots show the same value.
        assert dice_slot_values((4, 4)) == [4, 4, 4, 4]
