from engine.board import Color
from engine.game import Game, TurnRecord


class StubRng:
    def __init__(self, values):
        self.values = list(values)

    def randint(self, a, b):
        return self.values.pop(0)


class TestDetermineStarter:
    def test_white_wins_starter_roll(self):
        g = Game("A", "B", rng=StubRng([5, 3]))
        assert g.determine_starter() == Color.WHITE

    def test_black_wins_starter_roll(self):
        g = Game("A", "B", rng=StubRng([2, 6]))
        assert g.determine_starter() == Color.BLACK

    def test_tie_reroll(self):
        g = Game("A", "B", rng=StubRng([4, 4, 6, 2]))
        assert g.determine_starter() == Color.WHITE


class TestTurnFlow:
    def test_roll_and_play_advances_turn(self):
        # starter=W (5>3), first roll 3-5
        g = Game("A", "B", rng=StubRng([5, 3, 3, 5]))
        g.determine_starter()
        dice = g.roll()
        assert dice == (3, 5)
        seqs = g.legal_sequences()
        assert seqs
        g.play(seqs[0])
        assert g.current_player == Color.BLACK
        assert len(g.history) == 1

    def test_first_roll_flag(self):
        # starter=W (5>3); W rolls 3-5; B rolls 2-1; W rolls 4-2
        g = Game("A", "B", rng=StubRng([5, 3, 3, 5, 2, 1, 4, 2]))
        g.determine_starter()
        g.roll()
        assert g.is_first_roll_for_current is True
        g.play(g.legal_sequences()[0])
        # After W plays first turn, B is about to roll their first turn too.
        assert g.is_first_roll_for_current is True
        g.roll()
        g.play(g.legal_sequences()[0])
        # Now W is on their SECOND turn.
        assert g.is_first_roll_for_current is False


class TestGameOver:
    def test_oyn_score_1(self):
        g = Game("A", "B")
        g.board.borne_off[Color.WHITE] = 15
        g.board.borne_off[Color.BLACK] = 3
        assert g.is_over() is True
        assert g.winner() == Color.WHITE
        assert g.score() == (1, 0)  # (white_pts, black_pts)

    def test_mars_score_2(self):
        g = Game("A", "B")
        g.board.borne_off[Color.WHITE] = 15
        g.board.borne_off[Color.BLACK] = 0
        assert g.score() == (2, 0)

    def test_not_over(self):
        g = Game("A", "B")
        assert g.is_over() is False


class TestTurnRecord:
    def test_record_has_player_dice_sequence(self):
        g = Game("A", "B", rng=StubRng([5, 3, 3, 5]))
        g.determine_starter()
        g.roll()
        seq = g.legal_sequences()[0]
        g.play(seq)
        rec = g.history[0]
        assert isinstance(rec, TurnRecord)
        assert rec.player == Color.WHITE
        assert rec.dice == (3, 5)
        assert rec.sequence == seq
        assert rec.is_first_roll is True
