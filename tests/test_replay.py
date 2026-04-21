from engine.board import Color
from engine.moves import Move
from notation.parser import ParsedGame, ParsedTurn
from notation.replay import Replay


def _parsed_opening_game():
    return ParsedGame(
        headers={"White": "A", "Black": "B"},
        turns=[
            ParsedTurn(1, Color.WHITE, (3, 5),
                       [Move(24, 21, False), Move(24, 19, False)], False),
            ParsedTurn(2, Color.BLACK, (6, 2),
                       [Move(12, 6, False), Move(12, 10, False)], False),
        ],
        result_score=(0, 0),
    )


class TestReplayStepping:
    def test_initial_state(self):
        r = Replay(_parsed_opening_game())
        assert r.current_move_index == -1
        assert r.board.points[24].count == 15

    def test_step_forward_one_move(self):
        r = Replay(_parsed_opening_game())
        r.step_forward()
        assert r.board.points[24].count == 14
        assert r.board.points[21].count == 1

    def test_step_forward_to_end(self):
        r = Replay(_parsed_opening_game())
        while not r.is_at_end():
            r.step_forward()
        # White: 13 on 24, 1 on 21, 1 on 19. Black: 13 on 12, 1 on 6, 1 on 10.
        assert r.board.points[24].count == 13
        assert r.board.points[12].count == 13
        assert r.board.points[6].count == 1
        assert r.board.points[6].color == Color.BLACK

    def test_step_backward(self):
        r = Replay(_parsed_opening_game())
        r.step_forward()
        r.step_forward()
        r.step_backward()
        assert r.board.points[19].count == 0
        assert r.board.points[24].count == 14

    def test_cannot_step_before_start(self):
        r = Replay(_parsed_opening_game())
        r.step_backward()
        assert r.current_move_index == -1

    def test_current_dice_and_player(self):
        r = Replay(_parsed_opening_game())
        r.step_forward()
        assert r.current_player == Color.WHITE
        assert r.current_dice == (3, 5)
        r.step_forward()
        r.step_forward()
        assert r.current_player == Color.BLACK
        assert r.current_dice == (6, 2)
