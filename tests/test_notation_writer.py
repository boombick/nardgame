from engine.board import Color
from engine.moves import Move
from engine.game import TurnRecord
from notation.writer import format_move, format_turn, format_game


class TestFormatMove:
    def test_regular_move(self):
        assert format_move(Move(24, 19, False)) == "24/19"

    def test_bear_off(self):
        assert format_move(Move(6, 0, True)) == "6/off"


class TestFormatTurn:
    def test_regular_turn(self):
        rec = TurnRecord(
            player=Color.WHITE, dice=(3, 5), is_first_roll=False,
            sequence=[Move(24, 21, False), Move(24, 19, False)],
        )
        assert format_turn(1, rec) == "1. W 3-5: 24/21 24/19"

    def test_doubles_compress_repeats(self):
        rec = TurnRecord(
            player=Color.WHITE, dice=(1, 1), is_first_roll=False,
            sequence=[Move(24, 23, False)] * 4,
        )
        assert format_turn(3, rec) == "3. W 1-1: 24/23(x4)"

    def test_skip(self):
        rec = TurnRecord(
            player=Color.WHITE, dice=(4, 4), is_first_roll=False,
            sequence=[],
        )
        assert format_turn(5, rec) == "5. W 4-4: --"

    def test_bear_off_in_notation(self):
        rec = TurnRecord(
            player=Color.BLACK, dice=(5, 3), is_first_roll=False,
            sequence=[Move(10, 5, False), Move(5, 0, True)],
        )
        assert format_turn(4, rec) == "4. B 5-3: 10/5 5/off"


class TestFormatGame:
    def test_headers_and_result_mars(self):
        history = [
            TurnRecord(Color.WHITE, (3, 5),
                       [Move(24, 21, False), Move(24, 19, False)], True),
        ]
        text = format_game(
            history=history,
            event="Test", date="2026-04-21",
            white="A", black="B",
            score=(2, 0),
        )
        assert '[Event "Test"]' in text
        assert '[Date "2026-04-21"]' in text
        assert '[White "A"]' in text
        assert '[Black "B"]' in text
        assert "1. W 3-5: 24/21 24/19" in text
        assert "Result: 2-0 (Марс)" in text

    def test_result_oyn(self):
        text = format_game(history=[], event="E", date="D", white="A", black="B",
                           score=(1, 0))
        assert "Result: 1-0 (Ойн)" in text

    def test_result_draw(self):
        text = format_game(history=[], event="E", date="D", white="A", black="B",
                           score=(0, 0), draw=True)
        assert "Result: 1/2-1/2" in text
