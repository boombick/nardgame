import pytest
from engine.board import Board, Color, point_to_step, step_to_point


class TestBoardInit:
    def test_initial_white_checkers(self):
        board = Board()
        assert board.points[24].count == 15
        assert board.points[24].color == Color.WHITE

    def test_initial_black_checkers(self):
        board = Board()
        assert board.points[12].count == 15
        assert board.points[12].color == Color.BLACK

    def test_initial_empty_points(self):
        board = Board()
        for pt in range(1, 25):
            if pt not in (12, 24):
                assert board.points[pt].count == 0

    def test_initial_borne_off(self):
        board = Board()
        assert board.borne_off[Color.WHITE] == 0
        assert board.borne_off[Color.BLACK] == 0

    def test_total_white_checkers(self):
        board = Board()
        on_board = sum(p.count for p in board.points[1:] if p.color == Color.WHITE)
        assert on_board + board.borne_off[Color.WHITE] == 15

    def test_total_black_checkers(self):
        board = Board()
        on_board = sum(p.count for p in board.points[1:] if p.color == Color.BLACK)
        assert on_board + board.borne_off[Color.BLACK] == 15


class TestStepConversion:
    def test_white_head_is_step_0(self):
        assert point_to_step(24, Color.WHITE) == 0

    def test_white_home_edge_is_step_18(self):
        assert point_to_step(6, Color.WHITE) == 18

    def test_white_deepest_home_is_step_23(self):
        assert point_to_step(1, Color.WHITE) == 23

    def test_black_head_is_step_0(self):
        assert point_to_step(12, Color.BLACK) == 0

    def test_black_home_edge_is_step_18(self):
        assert point_to_step(18, Color.BLACK) == 18

    def test_black_deepest_home_is_step_23(self):
        assert point_to_step(13, Color.BLACK) == 23

    def test_roundtrip_white(self):
        for pt in range(1, 25):
            assert step_to_point(point_to_step(pt, Color.WHITE), Color.WHITE) == pt

    def test_roundtrip_black(self):
        for pt in range(1, 25):
            assert step_to_point(point_to_step(pt, Color.BLACK), Color.BLACK) == pt


class TestHomeAndHead:
    def test_white_head(self):
        assert Board().is_head(24, Color.WHITE) is True
        assert Board().is_head(12, Color.WHITE) is False

    def test_black_head(self):
        assert Board().is_head(12, Color.BLACK) is True
        assert Board().is_head(24, Color.BLACK) is False

    def test_white_home_range(self):
        b = Board()
        for pt in range(1, 7):
            assert b.is_home(pt, Color.WHITE) is True
        for pt in range(7, 25):
            assert b.is_home(pt, Color.WHITE) is False

    def test_black_home_range(self):
        b = Board()
        for pt in range(13, 19):
            assert b.is_home(pt, Color.BLACK) is True
        for pt in list(range(1, 13)) + list(range(19, 25)):
            assert b.is_home(pt, Color.BLACK) is False

    def test_all_in_home_initial_false(self):
        b = Board()
        assert b.all_in_home(Color.WHITE) is False
        assert b.all_in_home(Color.BLACK) is False


class TestMutators:
    def test_remove_one_decrements_count(self):
        b = Board()
        b.remove_one(24, Color.WHITE)
        assert b.points[24].count == 14
        assert b.points[24].color == Color.WHITE

    def test_remove_one_clears_color_when_count_reaches_zero(self):
        b = Board()
        # Move all but one checker away so we can drain the last one
        b.points[24].count = 1
        b.remove_one(24, Color.WHITE)
        assert b.points[24].count == 0
        assert b.points[24].color is None

    def test_place_one_sets_color_on_empty_point(self):
        b = Board()
        b.place_one(1, Color.WHITE)
        assert b.points[1].color == Color.WHITE
        assert b.points[1].count == 1

    def test_place_one_stacks_on_own_color(self):
        b = Board()
        b.place_one(1, Color.WHITE)
        b.place_one(1, Color.WHITE)
        assert b.points[1].count == 2
        assert b.points[1].color == Color.WHITE

    def test_bear_off_one_increments_borne_off_and_removes_from_board(self):
        b = Board()
        initial_count = b.points[24].count
        b.bear_off_one(24, Color.WHITE)
        assert b.borne_off[Color.WHITE] == 1
        assert b.points[24].count == initial_count - 1

    def test_clone_is_independent_mutating_original_does_not_affect_clone(self):
        b = Board()
        c = b.clone()
        b.remove_one(24, Color.WHITE)
        assert c.points[24].count == 15  # clone unaffected

    def test_clone_is_independent_mutating_clone_does_not_affect_original(self):
        b = Board()
        c = b.clone()
        c.remove_one(12, Color.BLACK)
        assert b.points[12].count == 15  # original unaffected

    def test_remove_one_raises_when_no_own_checker(self):
        b = Board()
        with pytest.raises(AssertionError):
            b.remove_one(1, Color.WHITE)  # point 1 is empty

    def test_place_one_raises_when_occupied_by_opponent(self):
        b = Board()
        # Point 24 already has WHITE checkers; placing BLACK there must fail
        with pytest.raises(AssertionError):
            b.place_one(24, Color.BLACK)
