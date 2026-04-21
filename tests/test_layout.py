from ui.layout import BoardLayout, checker_positions


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
