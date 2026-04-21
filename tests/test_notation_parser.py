from engine.board import Color
from engine.moves import Move
from notation.parser import ParsedGame, parse_game, parse_turn_line


class TestParseTurnLine:
    def test_regular(self):
        player, dice, moves, is_skip = parse_turn_line("1. W 3-5: 24/21 24/19")
        assert player == Color.WHITE
        assert dice == (3, 5)
        assert moves == [Move(24, 21, False), Move(24, 19, False)]
        assert is_skip is False

    def test_bear_off(self):
        player, dice, moves, _ = parse_turn_line("4. B 5-3: 10/5 5/off")
        assert player == Color.BLACK
        assert dice == (5, 3)
        assert moves == [Move(10, 5, False), Move(5, 0, True)]

    def test_doubles_expansion(self):
        _, _, moves, _ = parse_turn_line("3. W 1-1: 24/23(x4)")
        assert moves == [Move(24, 23, False)] * 4

    def test_skip(self):
        _, _, moves, is_skip = parse_turn_line("5. W 4-4: --")
        assert moves == []
        assert is_skip is True


class TestParseGame:
    def test_full_game(self):
        text = (
            '[Event "T"]\n'
            '[Date "2026-04-21"]\n'
            '[White "A"]\n'
            '[Black "B"]\n'
            "\n"
            "1. W 3-5: 24/21 24/19\n"
            "2. B 6-2: 12/6 12/10\n"
            "Result: 2-0 (Марс)\n"
        )
        g = parse_game(text)
        assert isinstance(g, ParsedGame)
        assert g.headers["Event"] == "T"
        assert g.headers["White"] == "A"
        assert g.headers["Black"] == "B"
        assert len(g.turns) == 2
        assert g.turns[0].player == Color.WHITE
        assert g.turns[1].player == Color.BLACK
        assert g.result_score == (2, 0)
        assert g.result_note == "Марс"

    def test_result_draw(self):
        text = '[Event "T"]\n\nResult: 1/2-1/2\n'
        g = parse_game(text)
        assert g.is_draw is True


class TestRoundTrip:
    def test_writer_parser_roundtrip(self):
        from engine.game import TurnRecord
        from notation.writer import format_game
        history = [
            TurnRecord(Color.WHITE, (3, 5),
                       [Move(24, 21, False), Move(24, 19, False)], True),
            TurnRecord(Color.BLACK, (1, 1),
                       [Move(12, 11, False)] * 4, True),
        ]
        text = format_game(history=history, event="E", date="D",
                           white="A", black="B", score=(1, 0))
        g = parse_game(text)
        assert len(g.turns) == 2
        assert g.turns[1].moves == [Move(12, 11, False)] * 4
