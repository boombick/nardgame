import pytest
from engine.board import Board, Color
from engine.moves import Move, is_legal_single, apply_single


class TestMoveDataclass:
    def test_move_fields(self):
        m = Move(from_point=24, to_point=19, is_bear_off=False)
        assert m.from_point == 24 and m.to_point == 19
        assert m.is_bear_off is False

    def test_bear_off_move(self):
        m = Move(from_point=3, to_point=0, is_bear_off=True)
        assert m.is_bear_off is True
        assert m.to_point == 0


class TestIsLegalSingle:
    def test_white_from_head(self):
        assert is_legal_single(Board(), Color.WHITE, 24, 5) is True

    def test_not_own_checker(self):
        b = Board()
        assert is_legal_single(b, Color.BLACK, 24, 5) is False

    def test_empty_from_point(self):
        assert is_legal_single(Board(), Color.WHITE, 10, 3) is False

    def test_blocked_by_opponent(self):
        b = Board()
        b.points[19].count = 1
        b.points[19].color = Color.BLACK
        assert is_legal_single(b, Color.WHITE, 24, 5) is False

    def test_own_pile_is_ok(self):
        b = Board()
        b.points[19].count = 2
        b.points[19].color = Color.WHITE
        assert is_legal_single(b, Color.WHITE, 24, 5) is True

    def test_cannot_bear_off_when_not_home(self):
        assert is_legal_single(Board(), Color.WHITE, 24, 5) is True
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[6].count = 15
        b.points[6].color = Color.WHITE
        # from pt 24 with no checker -> illegal
        assert is_legal_single(b, Color.WHITE, 24, 6) is False

    def test_bear_off_exact(self):
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[6].count = 15
        b.points[6].color = Color.WHITE
        assert is_legal_single(b, Color.WHITE, 6, 6) is True

    def test_bear_off_over_when_higher_empty(self):
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[3].count = 15
        b.points[3].color = Color.WHITE
        # die=5 overshoots; highest occupied is pt 3 -> allowed
        assert is_legal_single(b, Color.WHITE, 3, 5) is True

    def test_bear_off_over_blocked_when_higher_occupied(self):
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[5].count = 1
        b.points[5].color = Color.WHITE
        b.points[3].count = 14
        b.points[3].color = Color.WHITE
        # die=6 overshoots from 5; from 5 OK (highest), from 3 NO (5 still occupied)
        assert is_legal_single(b, Color.WHITE, 5, 6) is True
        assert is_legal_single(b, Color.WHITE, 3, 6) is False


class TestApplySingle:
    def test_apply_regular(self):
        b = Board()
        apply_single(b, Color.WHITE, Move(24, 19, False))
        assert b.points[24].count == 14
        assert b.points[19].count == 1
        assert b.points[19].color == Color.WHITE

    def test_apply_bear_off(self):
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[6].count = 15
        b.points[6].color = Color.WHITE
        apply_single(b, Color.WHITE, Move(6, 0, True))
        assert b.points[6].count == 14
        assert b.borne_off[Color.WHITE] == 1
