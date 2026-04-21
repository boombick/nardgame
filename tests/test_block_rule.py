from engine.board import Board, Color
from engine.moves import forms_illegal_block, Move


def _white_blocks_at(points):
    b = Board()
    b.points[24].count = 0
    b.points[24].color = None
    for pt in points:
        b.points[pt].count = 1
        b.points[pt].color = Color.WHITE
    return b


class TestBlockRule:
    def test_no_block_when_five_in_row(self):
        # White owns pts 20-23; checker at 24 moves 24 -> 19 would create 19-24 run of 6.
        # Black at 15 is ahead (< 19 in white-direction) => legal.
        b = _white_blocks_at([20, 21, 22, 23])
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        b.points[15].count = 1
        b.points[15].color = Color.BLACK
        move = Move(24, 19, False)
        assert forms_illegal_block(b, Color.WHITE, move) is False

    def test_block_illegal_no_opponent_ahead(self):
        # White owns 19..23 and stacks 2 at 24; moving one 24 -> 19 leaves a
        # checker at 24 so the run becomes 19..24 (six consecutive).
        # Black sits only at its head (pt 12): no non-head opponent checker
        # exists anywhere => block is illegal.
        b = _white_blocks_at([19, 20, 21, 22, 23])
        b.points[24].count = 2
        b.points[24].color = Color.WHITE
        b.points[12].count = 1
        b.points[12].color = Color.BLACK
        move = Move(24, 19, False)
        assert forms_illegal_block(b, Color.WHITE, move) is True

    def test_block_legal_with_opponent_ahead(self):
        # White owns 20..24 and moves a checker from 13 -> 19 forming 19..24 prime.
        # Black at 5 is ahead (< 19 in white-direction) => legal.
        b = _white_blocks_at([20, 21, 22, 23, 24])
        b.points[13].count = 1
        b.points[13].color = Color.WHITE
        b.points[5].count = 1
        b.points[5].color = Color.BLACK
        move = Move(13, 19, False)
        assert forms_illegal_block(b, Color.WHITE, move) is False

    def test_no_block_when_move_does_not_complete_six(self):
        b = _white_blocks_at([19, 20, 21, 22])  # only 4 in a row
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        b.points[12].count = 1
        b.points[12].color = Color.BLACK
        move = Move(24, 23, False)  # run becomes 19-23 = 5 long
        assert forms_illegal_block(b, Color.WHITE, move) is False
