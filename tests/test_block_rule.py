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
        # Black at 15 is in black's home (13..18) => block is legal per spec.
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

    def test_block_legal_with_opponent_in_home(self):
        # White owns 20..24 and moves a checker from 13 -> 19 forming 19..24 prime.
        # Per spec, a block is legal only if at least one opponent checker is
        # in the opponent's home (pts 13..18 for black). Black at 14 is in
        # home => block legal.
        b = _white_blocks_at([20, 21, 22, 23, 24])
        b.points[13].count = 1
        b.points[13].color = Color.WHITE
        b.points[14].count = 1
        b.points[14].color = Color.BLACK
        move = Move(13, 19, False)
        assert forms_illegal_block(b, Color.WHITE, move) is False

    def test_block_illegal_opponent_past_block_but_not_in_home(self):
        # Regression for the user-reported position: white builds a 6-prime
        # (7..12) while black has a checker at pt 5 (past the block in
        # white's direction) but no black checker is in black's home.
        # Per spec this is still illegal — "хотя бы одна шашка противника
        # в его доме" is the binding requirement.
        b = _white_blocks_at([7, 8, 9, 10, 11, 12])
        b.points[13].count = 1
        b.points[13].color = Color.WHITE
        # Move 13 -> 12 is a no-op on the run (pt 12 already white), so use
        # a different completing move: 18 -> 12 (die 6) with pt 12 temporarily
        # empty before the move.
        b.points[12].count = 0
        b.points[12].color = None
        b.points[18].count = 1
        b.points[18].color = Color.WHITE
        b.points[5].count = 1
        b.points[5].color = Color.BLACK    # past block in white-direction
        b.points[20].count = 1             # not in black's home (13..18)
        b.points[20].color = Color.BLACK
        move = Move(18, 12, False)
        assert forms_illegal_block(b, Color.WHITE, move) is True

    def test_no_block_when_move_does_not_complete_six(self):
        b = _white_blocks_at([19, 20, 21, 22])  # only 4 in a row
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        b.points[12].count = 1
        b.points[12].color = Color.BLACK
        move = Move(24, 23, False)  # run becomes 19-23 = 5 long
        assert forms_illegal_block(b, Color.WHITE, move) is False
